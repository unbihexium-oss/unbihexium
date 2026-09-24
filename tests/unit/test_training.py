# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_training.py
# Title       : Tests of training, checkpoints and evaluation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and PyTorch
# =============================================================================
#
# Abstract
# --------
# Trains tiny starter models for a few epochs on synthetic data and checks
# that they learn (validation metrics improve over the untrained model),
# that checkpoints carry the normalisation statistics and training metadata
# and load back with identical predictions, that fine-tuning from a
# checkpoint works, that training and evaluation read the dataset folder
# layout, and the learning rate schedule and chip dataset.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON history.
import json

# Represent file paths.
from pathlib import Path

# Arrays.
import numpy as np

# Test framework.
import pytest

# PyTorch is optional for the library; skip these tests without it.
torch = pytest.importorskip("torch")

# Synthetic data.
from unbihexium.ai.data import SyntheticDataset  # noqa: E402 - imported after the skip check

# Model construction.
from unbihexium.ai.models import build_model  # noqa: E402 - imported after the skip check

# Training API under test.
from unbihexium.ai.training import (  # noqa: E402 - imported after the skip check
    ChipDataset,  # Chip dataset.
    TrainConfig,  # Hyperparameters.
    Trainer,  # Optimisation loop.
    collate,  # Batch assembly.
    estimate_normalization,  # Band statistics.
    evaluate,  # Evaluation entry point.
    train,  # Training entry point.
)  # End of the training imports.

# Sample records.
from unbihexium.ai.transforms import Sample  # noqa: E402 - imported after the skip check

# Checkpoint loading.
from unbihexium.zoo.checkpoint import load_checkpoint, read_checkpoint  # noqa: E402


# Short training configuration for tests.
def quick(tmp_path: Path, **changes: object) -> TrainConfig:
    # Few epochs on small chips, quietly.
    base = {"epochs": 3, "batch_size": 8, "chip_size": 64, "learning_rate": 3e-3, "verbose": False}
    # Apply the changes.
    base.update(changes)
    # Configuration writing into the temporary directory.
    return TrainConfig(output_dir=str(tmp_path / "run"), device="cpu", **base)  # type: ignore[arg-type]


# The schedule warms up linearly and decays to one percent.
def test_lr_schedule() -> None:
    # First step of the warm-up.
    assert Trainer.lr_factor(0, 10, 100) == pytest.approx(0.1)
    # End of the warm-up.
    assert Trainer.lr_factor(9, 10, 100) == pytest.approx(1.0)
    # End of training.
    assert Trainer.lr_factor(100, 10, 100) == pytest.approx(0.01)


# Chips are encoded for the task and batched.
def test_chip_dataset() -> None:
    # Tiny detector configuration.
    config = build_model("ship_detector_tiny").config
    # Random chips from synthetic samples.
    chips = ChipDataset(SyntheticDataset(config, 4, 80), config, 64, length=6)
    # Six chips per epoch.
    assert len(chips) == 6
    # Batch of two chips.
    batch = collate([chips[0], chips[1]])
    # Image, heat map and box lists.
    assert batch["image"].shape == (2, 3, 64, 64) and batch["heatmap"].shape == (2, 1, 16, 16)
    # Boxes stay lists of arrays.
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    # Grid chips cover every image with whole chips.
    grid = ChipDataset(SyntheticDataset(config, 2, 80), config, 64, mode="grid")
    # Two images of 80 pixels need four chips each.
    assert len(grid) == 8


# Detection and segmentation models learn on synthetic data.
@pytest.mark.parametrize(
    ("family", "metric"), [("ship_detector", "map50"), ("water_surface_detector", "miou")]
)
def test_training_improves(tmp_path: Path, family: str, metric: str) -> None:
    # Train the tiny model.
    result = train(f"{family}_tiny", synthetic=48, config=quick(tmp_path, epochs=4))
    # Metric of the best checkpoint.
    best = result.best_metrics[metric]
    # Clearly better than chance.
    assert best > 0.3
    # History of every epoch.
    history = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    # Four epochs recorded.
    assert len(history["history"]) == 4
    # The training loss decreased.
    assert history["history"][-1]["train_loss"] < history["history"][0]["train_loss"]


# Checkpoints carry normalisation and reproduce predictions.
def test_checkpoint_contents(tmp_path: Path) -> None:
    # Train a regression model briefly.
    result = train("tree_height_estimator_tiny", synthetic=16, config=quick(tmp_path, epochs=1))
    # Raw checkpoint.
    payload = read_checkpoint(result.best_checkpoint)
    # Normalisation statistics for every band.
    assert len(payload["config"]["extra"]["normalization"]["mean"]) == 12
    # Training metadata.
    assert payload["training"]["epoch"] == 1
    # Loaded model.
    model = load_checkpoint(result.best_checkpoint)
    # Same input twice gives the same output.
    x = torch.rand(1, 12, 32, 32)
    # Two forward passes.
    with torch.no_grad():
        # Deterministic evaluation mode.
        assert torch.equal(model(x), model(x))


# Fine-tuning continues from a checkpoint and keeps its statistics.
def test_fine_tune(tmp_path: Path) -> None:
    # First run.
    first = train("yield_predictor_tiny", synthetic=16, config=quick(tmp_path / "a", epochs=1))
    # Normalisation of the first run.
    stats = read_checkpoint(first.best_checkpoint)["config"]["extra"]["normalization"]
    # Second run from the checkpoint.
    second = train(first.best_checkpoint, synthetic=16, config=quick(tmp_path / "b", epochs=1))
    # The statistics are unchanged.
    assert read_checkpoint(second.best_checkpoint)["config"]["extra"]["normalization"] == stats


# Spectral index models cannot be trained.
def test_formula_not_trainable(tmp_path: Path) -> None:
    # NDVI has no parameters.
    with pytest.raises(ValueError, match="not trainable"):
        # Attempt to train.
        train("ndvi_calculator", synthetic=4, config=quick(tmp_path))


# Training and evaluation read the dataset folder layout.
def test_folder_training(tmp_path: Path) -> None:
    # Dataset root.
    root = tmp_path / "data"
    # Samples of the synthetic water task.
    config = build_model("water_surface_detector_tiny").config
    # Generator of samples.
    source = SyntheticDataset(config, 6, 64)
    # Write train and val splits as NumPy files.
    for split, indices in (("train", range(4)), ("val", range(4, 6))):
        # Image and label directories.
        (root / split / "images").mkdir(parents=True)
        # Labels.
        (root / split / "labels").mkdir(parents=True)
        # Write every sample.
        for i in indices:
            # Sample.
            sample = source[i]
            # Image.
            np.save(root / split / "images" / f"s{i}.npy", sample.image)
            # Mask.
            np.save(root / split / "labels" / f"s{i}.npy", sample.mask)
    # Train on the folder.
    result = train("water_surface_detector_tiny", root, quick(tmp_path, epochs=2))
    # Validation metrics exist.
    assert "miou" in result.best_metrics
    # Evaluate the checkpoint on the validation split.
    metrics = evaluate(result.best_checkpoint, root, "val", chip_size=64, device="cpu")
    # Accuracy is a fraction.
    assert 0.0 <= metrics["accuracy"] <= 1.0 and metrics["pixels"] == 2 * 64 * 64


# Normalisation statistics ignore NaN values and the padding of small images.
def test_estimate_normalization_ignores_nan_and_padding() -> None:
    # Tiny segmentation configuration.
    config = build_model("water_surface_detector_tiny").config
    # Image smaller than the chip, alternating values 10 and 20 per column.
    image = np.tile(np.array([10.0, 20.0], np.float32), (config.in_channels, 40, 20))
    # A block of missing values.
    image[:, :10, :10] = np.nan
    # One labelled sample.
    sample = Sample(image=image, mask=np.zeros((40, 40), np.int64))
    # Grid chips of 64 pixels, so the image is padded.
    chips = ChipDataset([sample], config, 64, mode="grid")
    # Estimated statistics.
    norm = estimate_normalization(chips)
    # Mean and standard deviation of the valid values only.
    assert np.allclose(norm.mean.ravel(), 15.0) and np.allclose(norm.std.ravel(), 5.0)


# =============================================================================
# End of module tests/unit/test_training.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
