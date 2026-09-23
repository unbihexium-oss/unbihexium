# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/training.py
# Title       : Training and evaluation of model zoo models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Trains the starter models of the model zoo on labelled data:
#
#   ChipDataset   PyTorch dataset over a FolderDataset, a SyntheticDataset or
#                 any sequence of Sample records; cuts random training chips
#                 or a regular grid of validation chips, normalises the
#                 bands, augments and encodes the task targets
#   TrainConfig   hyperparameters (epochs, batch size, learning rate, ...)
#   Trainer       AdamW with linear warm-up and cosine decay, gradient
#                 clipping, optional mixed precision, validation after every
#                 epoch, best and last checkpoints, early stopping and a
#                 JSON history
#   train         end-to-end entry point used by `unbihexium train`
#   evaluate      metrics of a model on a dataset split
#
# Checkpoints are written with zoo.checkpoint.save_checkpoint. The per-band
# normalisation statistics estimated from the training data are stored in
# the model configuration (extra["normalization"]), so that inference,
# ONNX exports and the task APIs apply exactly the same scaling.
#
# Usage
# -----
#   from unbihexium.ai.training import TrainConfig, train
#   result = train("ship_detector_base", "data/ships", TrainConfig(epochs=50))
#   print(result.best_checkpoint, result.best_metrics)
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON history files.
import json

# Cosine learning rate schedule.
import math

# Wall-clock timing of epochs.
import time

# Configuration and result records.
from dataclasses import asdict, dataclass, field

# Represent file paths.
from pathlib import Path

# Type of loosely structured values and callback type.
from typing import Any, Callable

# Arrays.
import numpy as np

# Tensor library.
import torch

# Data loading utilities.
from torch.utils.data import DataLoader, Dataset

# Datasets on disk and synthetic data, CenterNet target encoding.
from unbihexium.ai.data import FolderDataset, SyntheticDataset, encode_centernet

# Metric accumulators.
from unbihexium.ai.evaluation import TaskEvaluator

# Training losses.
from unbihexium.ai.losses import TaskLoss

# Model construction.
from unbihexium.ai.models.factory import ZooModel, build_model

# Samples, normalisation and augmentation.
from unbihexium.ai.transforms import Augmenter, Normalization, Sample, crop, pad, random_crop

# Task of a model.
from unbihexium.zoo.catalog import Task

# Checkpoint files.
from unbihexium.zoo.checkpoint import load_checkpoint, save_checkpoint

# Model configuration.
from unbihexium.zoo.config import BuildConfig

# Number of images used to estimate the normalisation statistics.
NORMALIZATION_IMAGES = 32


# Hyperparameters of a training run.
@dataclass
class TrainConfig:
    # Number of passes over the training chips.
    epochs: int = 50
    # Chips per optimisation step.
    batch_size: int = 8
    # Peak learning rate of AdamW.
    learning_rate: float = 1e-3
    # Decoupled weight decay of AdamW.
    weight_decay: float = 1e-4
    # Epochs of linear learning rate warm-up.
    warmup_epochs: float = 1.0
    # Side length of the training chips; default is the model tile size.
    chip_size: int | None = None
    # Random chips per epoch; default is one per training image.
    samples_per_epoch: int | None = None
    # Worker processes of the data loaders.
    num_workers: int = 0
    # Device: auto, cpu, cuda, cuda:1 or mps.
    device: str = "auto"
    # Seed of the random number generators.
    seed: int = 0
    # Mixed precision on CUDA devices.
    amp: bool = False
    # Maximum norm of the gradients; zero disables clipping.
    grad_clip: float = 10.0
    # Stop after this many epochs without improvement; None disables it.
    patience: int | None = None
    # Loss of regression targets: l1, mse or huber.
    regression_loss: str = "l1"
    # Random rotations and mirrors.
    augment: bool = True
    # Random brightness, contrast and noise; default for classification tasks.
    photometric: bool | None = None
    # Directory of checkpoints and history.
    output_dir: str = "runs"
    # Score threshold of detections during validation.
    detection_threshold: float = 0.3
    # Optional class weights of the segmentation loss.
    class_weights: list[float] | None = None
    # Print a line per epoch.
    verbose: bool = True


# Outcome of a training run.
@dataclass
class TrainingResult:
    # Metrics of every epoch.
    history: list[dict[str, Any]] = field(default_factory=list)
    # Epoch of the best checkpoint.
    best_epoch: int = 0
    # Validation metrics of the best checkpoint.
    best_metrics: dict[str, Any] = field(default_factory=dict)
    # Path of the best checkpoint.
    best_checkpoint: Path | None = None
    # Path of the last checkpoint.
    last_checkpoint: Path | None = None


# Resolve "auto" to the best available device.
def resolve_device(device: str) -> torch.device:
    # Explicit devices are used as given.
    if device != "auto":
        # Parsed device.
        return torch.device(device)
    # CUDA when available.
    if torch.cuda.is_available():
        # First GPU.
        return torch.device("cuda")
    # Apple GPUs when available.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Metal backend.
        return torch.device("mps")
    # CPU otherwise.
    return torch.device("cpu")


# PyTorch dataset of encoded training or validation chips.
class ChipDataset(Dataset):
    # Create the dataset.
    def __init__(
        self,  # The dataset.
        source: Any,  # FolderDataset, SyntheticDataset or a sequence of samples.
        config: BuildConfig,  # Model configuration.
        chip_size: int,  # Side length of the chips.
        mode: str = "random",  # random chips for training, grid chips for validation.
        normalization: Normalization | None = None,  # Band standardisation.
        augmenter: Augmenter | None = None,  # Training augmentation.
        length: int | None = None,  # Chips per epoch in random mode.
        seed: int = 0,  # Seed of the random chips.
    ) -> None:  # The constructor returns nothing.
        # Samples to cut chips from.
        self.source = source
        # Model configuration.
        self.config = config
        # Chips are multiples of the network size multiple.
        m = config.size_multiple
        # Rounded chip size.
        self.chip = int(math.ceil(chip_size / m) * m)
        # Chip selection mode.
        self.mode = mode
        # Band standardisation.
        self.normalization = normalization
        # Augmentation, only used in random mode.
        self.augmenter = augmenter if mode == "random" else None
        # Seed of the random chips.
        self.seed = seed
        # Epoch counter, mixed into the random chips.
        self.epoch = 0
        # Scene targets describe whole images and are never tiled.
        self.whole = config.task is Task.SCENE_REGRESSION
        # Grid windows (image, top, left) for validation.
        self.windows: list[tuple[int, int, int]] = []
        # Build the grid.
        if mode == "grid":
            # Every image.
            for i in range(len(source)):
                # Image size without reading pixels when possible.
                _, h, w = self._shape(i)
                # Whole images for scene targets.
                if self.whole:
                    # One window per image.
                    self.windows.append((i, 0, 0))
                    # Next image.
                    continue
                # Non-overlapping windows covering the image.
                for top in range(0, max(h - self.chip, 0) + self.chip, self.chip):
                    # Windows along the row.
                    for left in range(0, max(w - self.chip, 0) + self.chip, self.chip):
                        # Record the window if it starts inside the image.
                        if top < h and left < w:
                            # Image index and window origin.
                            self.windows.append((i, top, left))
        # Number of chips.
        self.length = len(self.windows) if mode == "grid" else (length or len(source))

    # Shape of an image of the source.
    def _shape(self, index: int) -> tuple[int, int, int]:
        # Folder datasets read the file header only.
        if hasattr(self.source, "shape"):
            # Header shape.
            return self.source.shape(index)
        # Other sources are loaded.
        return tuple(self.source[index].image.shape)  # type: ignore[return-value]

    # Load a window of an image, reading only that window from files.
    def _load(self, index: int, window: tuple[int, int, int, int] | None) -> Sample:
        # Windowed reading of folder datasets.
        if hasattr(self.source, "load"):
            # Clip the window to the image.
            if window is not None:
                # Image size.
                _, h, w = self._shape(index)
                # Window origin and size.
                top, left, hh, ww = window
                # Clipped window.
                window = (top, left, min(hh, h - top), min(ww, w - left))
            # Read the window.
            return self.source.load(index, window)
        # In-memory samples are cropped.
        sample = self.source[index]
        # Crop the requested window.
        return crop(sample, *window) if window is not None else sample

    # Number of chips.
    def __len__(self) -> int:
        # Fixed per epoch.
        return self.length

    # Cut, augment, normalise and encode chip `index`.
    def __getitem__(self, index: int) -> dict[str, Any]:
        # Generator per chip and epoch.
        rng = np.random.default_rng([self.seed, self.epoch, index])
        # Validation chips come from the grid.
        if self.mode == "grid":
            # Image and window of the chip.
            i, top, left = self.windows[index]
            # Whole images for scene targets.
            window = None if self.whole else (top, left, self.chip, self.chip)
            # Load the window.
            sample = self._load(i, window)
        # Training chips are random.
        else:
            # Image of the chip.
            i = index % len(self.source)
            # Image size.
            _, h, w = self._shape(i)
            # Random window origin.
            top = int(rng.integers(0, max(h - self.chip, 0) + 1))
            # Random column.
            left = int(rng.integers(0, max(w - self.chip, 0) + 1))
            # Whole images for scene targets.
            window = None if self.whole else (top, left, self.chip, self.chip)
            # Load the window.
            sample = self._load(i, window)
            # Scene images larger than a chip are cropped at random.
            if self.whole and max(sample.height, sample.width) > self.chip:
                # Random chip.
                sample = random_crop(sample, self.chip, rng)
        # Scene images are cut to the chip size at the centre in grid mode.
        if self.whole and max(sample.height, sample.width) > self.chip:
            # Centre window.
            top, left = (sample.height - self.chip) // 2, (sample.width - self.chip) // 2
            # Crop the centre.
            sample = crop(sample, max(top, 0), max(left, 0), self.chip, self.chip)
        # Pad chips at the image border to the full chip size.
        sample = pad(sample, self.chip, self.chip)
        # Random rotations, mirrors and radiometric jitter.
        if self.augmenter is not None:
            # Augment the chip.
            sample = self.augmenter(sample, rng)
        # Normalise the image; NaN becomes zero.
        norm = self.normalization or np.nan_to_num
        # Standardised image.
        image = norm(sample.image)
        # Encoded item.
        item: dict[str, Any] = {"image": np.ascontiguousarray(image, dtype=np.float32)}
        # Task of the model.
        task = self.config.task
        # Detection targets.
        if task is Task.DETECTION:
            # Heat map, size, offset and weight arrays.
            item.update(
                encode_centernet(  # CenterNet targets.
                    sample.boxes,  # Boxes.
                    sample.labels,  # Classes.
                    self.config.out_channels,  # Number of classes.
                    self.chip,  # Chip height.
                    self.chip,  # Chip width.
                    self.config.output_stride,  # Output stride.
                )  # End of the encoding.
            )  # End of the update.
            # Raw boxes for the metrics.
            item["boxes"] = np.zeros((0, 4), np.float32) if sample.boxes is None else sample.boxes
            # Raw labels for the metrics.
            item["labels"] = np.zeros(0, np.int64) if sample.labels is None else sample.labels
        # Class masks.
        elif task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
            # Mask as int64.
            item["mask"] = np.ascontiguousarray(sample.mask, dtype=np.int64)
        # Scene vectors.
        elif task is Task.SCENE_REGRESSION:
            # Vector as float32.
            item["vector"] = np.asarray(sample.vector, dtype=np.float32)
        # Dense targets.
        else:
            # Values as float32.
            item["values"] = np.ascontiguousarray(sample.values, dtype=np.float32)
        # Return the encoded chip.
        return item


# Stack chips into a batch; boxes and labels stay lists of arrays.
def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    # Batch dictionary.
    batch: dict[str, Any] = {}
    # Every key of the items.
    for key in items[0]:
        # Variable-length detection references.
        if key in ("boxes", "labels"):
            # List of arrays.
            batch[key] = [item[key] for item in items]
        # Fixed-size arrays are stacked into tensors.
        else:
            # Stacked tensor.
            batch[key] = torch.from_numpy(np.stack([item[key] for item in items]))
    # Return the batch.
    return batch


# Estimate normalisation statistics from chips of a dataset.
def estimate_normalization(
    dataset: ChipDataset,  # Chips to sample.
    images: int = NORMALIZATION_IMAGES,  # Number of chips.
) -> Normalization:  # Band statistics.
    # Keep the current normalisation and augmentation.
    saved = (dataset.normalization, dataset.augmenter)
    # Raw, unaugmented chips.
    dataset.normalization, dataset.augmenter = None, None
    # Evenly spaced chip indices.
    indices = np.linspace(0, len(dataset) - 1, min(images, len(dataset))).astype(int)
    # Read the raw chips; missing values stay NaN.
    raw = []
    # Load every selected chip.
    for i in indices:
        # Sample of the chip before normalisation.
        item = dataset[int(i)]
        # Raw image.
        raw.append(item["image"])
    # Restore the dataset settings.
    dataset.normalization, dataset.augmenter = saved
    # Fit the statistics.
    return Normalization.fit(raw)


# Move tensor values of a batch to a device.
def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Tensors move; lists stay on the host.
    return {k: _move(v, device) for k, v in batch.items()}


# Move a tensor to a device; other values are returned unchanged.
def _move(value: Any, device: torch.device) -> Any:
    # Only tensors have a device.
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


# NumPy view of the targets of a batch for the metrics.
def to_numpy(batch: dict[str, Any]) -> dict[str, Any]:
    # Tensors become arrays; lists stay lists.
    return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}


# Optimisation loop with validation and checkpoints.
class Trainer:
    # Create a trainer for a model.
    def __init__(self, model: ZooModel, config: TrainConfig | None = None) -> None:
        # Hyperparameters.
        self.config = config or TrainConfig()
        # Device of the computation.
        self.device = resolve_device(self.config.device)
        # Model on the device.
        self.model = model.to(self.device)
        # Loss of the task.
        loss = TaskLoss(model.config, self.config.regression_loss, self.config.class_weights)
        # Loss buffers on the device.
        self.loss = loss.to(self.device)
        # Output directory.
        self.output_dir = Path(self.config.output_dir)
        # Mixed precision only on CUDA.
        self.amp = self.config.amp and self.device.type == "cuda"

    # Learning rate factor: linear warm-up, then cosine decay to 1 percent.
    @staticmethod
    def lr_factor(step: int, warmup: int, total: int) -> float:
        # Warm-up phase.
        if step < warmup:
            # Linear increase.
            return (step + 1) / max(warmup, 1)
        # Progress through the decay phase.
        progress = (step - warmup) / max(total - warmup, 1)
        # Cosine from 1 to 0.01.
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # Run the model over a loader and return the validation metrics.
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        # Metric accumulator of the task.
        evaluator = TaskEvaluator(self.model.config, self.config.detection_threshold)
        # Sum of the losses.
        total, batches = 0.0, 0
        # Evaluation mode.
        self.model.eval()
        # No gradients.
        with torch.inference_mode():
            # Every batch.
            for batch in loader:
                # Batch on the device.
                batch = to_device(batch, self.device)
                # Forward pass.
                output = self.model(batch["image"])
                # Loss of the batch.
                loss, _ = self.loss(output, batch)
                # Accumulate the loss.
                total, batches = total + float(loss), batches + 1
                # Accumulate the metrics.
                evaluator.update(output.float().cpu().numpy(), to_numpy(batch))
        # Metrics with the mean loss.
        return {"loss": total / max(batches, 1), **evaluator.compute()}

    # Train the model.
    def fit(
        self,  # The trainer.
        train_data: ChipDataset,  # Training chips.
        val_data: ChipDataset | None = None,  # Validation chips.
        callback: Callable[[dict[str, Any]], None] | None = None,  # Called after every epoch.
    ) -> TrainingResult:  # History and checkpoints.
        # Hyperparameters.
        cfg = self.config
        # Seed PyTorch for reproducible initialisation of the loaders.
        torch.manual_seed(cfg.seed)
        # Shuffled training loader.
        train_loader = DataLoader(
            train_data,  # Dataset.
            batch_size=cfg.batch_size,  # Batch size.
            shuffle=True,  # New order every epoch.
            num_workers=cfg.num_workers,  # Worker processes.
            collate_fn=collate,  # Batch assembly.
            drop_last=len(train_data) > cfg.batch_size,  # Avoid tiny last batches.
            generator=torch.Generator().manual_seed(cfg.seed),  # Reproducible order.
        )  # End of the training loader.
        # Validation loader in fixed order.
        val_loader = None
        # Loader over the validation chips, if any.
        if val_data is not None and len(val_data):
            # Fixed order, same batch size.
            val_loader = DataLoader(
                val_data,  # Dataset.
                cfg.batch_size,  # Batch size.
                num_workers=cfg.num_workers,  # Worker processes.
                collate_fn=collate,  # Batch assembly.
            )  # End of the validation loader.
        # AdamW with decoupled weight decay.
        optimiser = torch.optim.AdamW(
            self.model.parameters(),  # Trainable parameters.
            lr=cfg.learning_rate,  # Peak learning rate.
            weight_decay=cfg.weight_decay,  # Decoupled weight decay.
        )  # End of the optimiser.
        # Steps per epoch.
        steps = max(len(train_loader), 1)
        # Total and warm-up steps.
        total, warmup = cfg.epochs * steps, int(cfg.warmup_epochs * steps)
        # Per-step learning rate schedule.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser,  # Optimiser whose rate is scaled.
            lambda step: self.lr_factor(step, warmup, total),  # Factor per step.
        )  # End of the scheduler.
        # Gradient scaler for mixed precision.
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        # Metric that selects the best checkpoint.
        name, larger = TaskEvaluator(self.model.config).monitor
        # Without validation data the training loss is monitored.
        if val_loader is None:
            # Smaller loss is better.
            name, larger = "loss", False
        # Best value so far.
        best = -math.inf if larger else math.inf
        # Result record.
        result = TrainingResult()
        # Epochs without improvement.
        stale = 0
        # Create the output directory.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Epoch loop.
        for epoch in range(1, cfg.epochs + 1):
            # New random chips every epoch.
            train_data.epoch = epoch
            # Training mode.
            self.model.train()
            # Start time of the epoch.
            start = time.perf_counter()
            # Sum of the training losses.
            running, batches = 0.0, 0
            # Batch loop.
            for batch in train_loader:
                # Batch on the device.
                batch = to_device(batch, self.device)
                # Clear the gradients.
                optimiser.zero_grad(set_to_none=True)
                # Forward pass, in mixed precision if enabled.
                with torch.autocast(self.device.type, enabled=self.amp):
                    # Network output.
                    output = self.model(batch["image"])
                # Loss in float32.
                loss, _ = self.loss(output.float(), batch)
                # Skip batches with invalid losses instead of corrupting the weights.
                if not torch.isfinite(loss):
                    # Next batch.
                    continue
                # Backward pass.
                scaler.scale(loss).backward()
                # Gradient clipping.
                if cfg.grad_clip > 0:
                    # Unscale before measuring the norm.
                    scaler.unscale_(optimiser)
                    # Clip the norm.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                # Optimisation step.
                scaler.step(optimiser)
                # Update the loss scale.
                scaler.update()
                # Advance the schedule.
                scheduler.step()
                # Accumulate the loss.
                running, batches = running + float(loss.detach()), batches + 1
            # Record of the epoch.
            record: dict[str, Any] = {
                "epoch": epoch,  # Epoch number.
                "train_loss": running / max(batches, 1),  # Mean training loss.
                "lr": scheduler.get_last_lr()[0],  # Current learning rate.
            }  # End of the record.
            # Validation.
            if val_loader is not None:
                # Metrics on the validation chips.
                record["val"] = self.evaluate(val_loader)
            # Monitored value.
            value = record["val"][name] if val_loader is not None else record["train_loss"]
            # Duration of the epoch.
            record["seconds"] = round(time.perf_counter() - start, 3)
            # Add to the history.
            result.history.append(record)
            # Whether the monitored value improved.
            improved = math.isfinite(value) and (value > best if larger else value < best)
            # Save the best checkpoint.
            if improved:
                # New best value.
                best, stale = value, 0
                # Best epoch and metrics.
                result.best_epoch, result.best_metrics = epoch, record.get("val", {"loss": value})
                # Write the checkpoint.
                result.best_checkpoint = self._save("best.pt", record)
            # Count epochs without improvement.
            else:
                # One more stale epoch.
                stale += 1
            # Write the last checkpoint.
            result.last_checkpoint = self._save("last.pt", record)
            # Write the history.
            self._write_history(result)
            # Progress line.
            if cfg.verbose:
                # Epoch summary.
                print(
                    f"epoch {epoch}/{cfg.epochs} loss {record['train_loss']:.4f} {name} {value:.4f}"
                )
            # User callback.
            if callback is not None:
                # Pass the record.
                callback(record)
            # Early stopping.
            if cfg.patience is not None and stale >= cfg.patience:
                # Stop training.
                break
        # Return the result.
        return result

    # Save a checkpoint with the training state.
    def _save(self, name: str, record: dict[str, Any]) -> Path:
        # Training metadata stored with the weights.
        training = {
            "epoch": record["epoch"],  # Epoch.
            "metrics": _jsonable(record),  # Metrics of the epoch.
            "config": asdict(self.config),  # Hyperparameters.
        }
        # Write the checkpoint.
        save_checkpoint(self.model, self.output_dir / name, training=training)
        # Return its path.
        return self.output_dir / name

    # Write history.json.
    def _write_history(self, result: TrainingResult) -> None:
        # History document.
        data = {
            "model_id": self.model.config.model_id,  # Model.
            "best_epoch": result.best_epoch,  # Best epoch.
            "best_metrics": _jsonable(result.best_metrics),  # Best metrics.
            "history": _jsonable(result.history),  # Every epoch.
        }  # End of the document.
        # Write the file.
        (self.output_dir / "history.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


# Convert NaN and infinities to None for strict JSON.
def _jsonable(value: Any) -> Any:
    # Floats: non-finite values become None.
    if isinstance(value, float):
        # Finite values stay.
        return value if math.isfinite(value) else None
    # Dictionaries: convert the values.
    if isinstance(value, dict):
        # Recursive conversion.
        return {k: _jsonable(v) for k, v in value.items()}
    # Lists and tuples: convert the items.
    if isinstance(value, (list, tuple)):
        # Recursive conversion.
        return [_jsonable(v) for v in value]
    # Other values are unchanged.
    return value


# Model to train: a catalogue model, a checkpoint or a model object.
def _prepare_model(
    model: ZooModel | str | Path,  # Model, model id or checkpoint path.
    variant: str | None,  # Variant for model ids.
    channel_names: list[str] | None,  # Input override.
    outputs: list[str] | None,  # Output override.
) -> ZooModel:  # Model ready for training.
    # Model objects are used as given.
    if isinstance(model, ZooModel):
        # Train in place.
        return model
    # Checkpoints are fine-tuned.
    if Path(str(model)).suffix == ".pt" and Path(str(model)).is_file():
        # Load the checkpoint.
        return load_checkpoint(model)
    # Catalogue models start from their starter weights.
    return build_model(str(model), variant, channel_names=channel_names, outputs=outputs)


# Train a model on a dataset folder or on synthetic data.
def train(
    model: ZooModel | str | Path,  # Model, model id or checkpoint to fine-tune.
    data: str | Path | None = None,  # Dataset root in the folder layout.
    config: TrainConfig | None = None,  # Hyperparameters.
    variant: str | None = None,  # Variant for model ids.
    synthetic: int | None = None,  # Number of synthetic training samples instead of data.
    callback: Callable[[dict[str, Any]], None] | None = None,  # Called after every epoch.
) -> TrainingResult:  # History and checkpoints.
    # Hyperparameters.
    cfg = config or TrainConfig()
    # Settings of the dataset, which may override classes and bands.
    settings: dict[str, Any] = {}
    # Read dataset.yaml for real data.
    if data is not None and (Path(data) / "dataset.yaml").is_file():
        # Imported here because only folder datasets need it.
        import yaml

        # Parse the settings.
        settings = yaml.safe_load((Path(data) / "dataset.yaml").read_text(encoding="utf-8")) or {}
    # Model to train, customised to the dataset classes and bands if given.
    net = _prepare_model(model, variant, settings.get("channel_names"), settings.get("classes"))
    # Model configuration.
    mcfg = net.config
    # Spectral indices have nothing to learn.
    if not mcfg.task.is_trainable:
        # Explain the problem.
        raise ValueError(f"{mcfg.model_id} computes a fixed formula and is not trainable")
    # Chip size of the run.
    chip = cfg.chip_size or mcfg.tile_size
    # Photometric augmentation defaults to on for classification tasks.
    photometric = cfg.photometric
    # Default per task.
    if photometric is None:
        # Detection and segmentation profit from radiometric jitter.
        photometric = mcfg.task in (Task.DETECTION, Task.SEGMENTATION, Task.CHANGE_DETECTION)
    # Displacement fields change sign under rotations, so they are not rotated.
    geometric = cfg.augment and not ({"dx", "dy"} & set(mcfg.outputs))
    # Augmenter of the training chips.
    augmenter = Augmenter(geometric=geometric, photometric=cfg.augment and photometric)
    # Sources of training and validation samples.
    if synthetic:
        # Synthetic training samples.
        train_source: Any = SyntheticDataset(mcfg, length=synthetic, size=chip, seed=cfg.seed)
        # Number of independent synthetic validation samples.
        val_length = max(4, synthetic // 4)
        # Validation samples with another seed.
        val_source: Any = SyntheticDataset(mcfg, length=val_length, size=chip, seed=cfg.seed + 1)
    # Real data in the folder layout.
    elif data is not None:
        # Training split.
        train_source = FolderDataset(data, "train", mcfg)
        # Validation split, if present.
        has_val = (Path(data) / "val" / "images").is_dir()
        # Open the validation split when it exists.
        val_source = FolderDataset(data, "val", mcfg) if has_val else None
    # Some data is required.
    else:
        # Explain the options.
        raise ValueError("pass a dataset directory or a number of synthetic samples")
    # Training chips.
    train_set = ChipDataset(
        train_source,  # Samples.
        mcfg,  # Configuration.
        chip,  # Chip size.
        "random",  # Random chips.
        augmenter=augmenter,  # Augmentation.
        length=cfg.samples_per_epoch,  # Chips per epoch.
        seed=cfg.seed,  # Seed.
    )  # End of the training chips.
    # Statistics recorded with a fine-tuned checkpoint are kept.
    stored = mcfg.normalization
    # Otherwise estimate them from the training chips.
    normalization = Normalization.from_dict(stored) if stored else estimate_normalization(train_set)
    # Use them for the training chips.
    train_set.normalization = normalization
    # Validation chips on a regular grid.
    val_set = (
        ChipDataset(val_source, mcfg, chip, "grid", normalization=normalization)  # Grid chips.
        if val_source is not None  # Only with validation data.
        else None  # No validation.
    )  # End of the validation chips.
    # Record the statistics in the model configuration.
    net.config = mcfg.with_extra(normalization=normalization.to_dict())
    # Default output directory per model.
    if cfg.output_dir == "runs":
        # runs/<model id>.
        cfg.output_dir = str(Path("runs") / mcfg.model_id)
    # Train.
    return Trainer(net, cfg).fit(train_set, val_set, callback)


# Metrics of a model on a split of a dataset folder.
def evaluate(
    model: ZooModel | str | Path,  # Model, model id or checkpoint.
    data: str | Path,  # Dataset root.
    split: str = "val",  # Split to evaluate.
    chip_size: int | None = None,  # Chip size; default is the model tile size.
    batch_size: int = 8,  # Chips per forward pass.
    device: str = "auto",  # Device.
    threshold: float = 0.3,  # Detection score threshold.
) -> dict[str, Any]:  # Metrics.
    # Model to evaluate.
    net = _prepare_model(model, None, None, None)
    # Stored normalisation statistics.
    stored = net.config.normalization
    # Samples of the split.
    source = FolderDataset(data, split, net.config)
    # Grid chips.
    chips = ChipDataset(
        source,  # Samples.
        net.config,  # Configuration.
        chip_size or net.config.tile_size,  # Chip size.
        "grid",  # Regular grid.
        normalization=Normalization.from_dict(stored) if stored else None,  # Statistics.
    )  # End of the chips.
    # Trainer used for its evaluation loop.
    trainer = Trainer(net, TrainConfig(device=device, detection_threshold=threshold, verbose=False))
    # Loader in fixed order.
    loader = DataLoader(chips, batch_size, collate_fn=collate)
    # Metrics.
    return trainer.evaluate(loader)


# =============================================================================
# End of module src/unbihexium/ai/training.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
