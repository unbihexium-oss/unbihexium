# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/model.py
# Title       : Framework-neutral model wrapper
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; ONNX models need ONNX
#               Runtime and PyTorch models PyTorch (imported lazily)
# =============================================================================
#
# Abstract
# --------
# ModelWrapper runs a model on image arrays of shape (channels, height,
# width) or (batch, channels, height, width) with the same three stages for
# every framework:
#
#   preprocess    float32 conversion and per-channel standardisation
#                 (x - mean) / std
#   forward       ONNX Runtime session, TorchScript or torch.export module,
#                 PyTorch module, scikit-learn style estimator (pixels as
#                 samples) or any callable on the batch
#   postprocess   task-specific decoding of the raw output
#
# Postprocessing by task:
#   segmentation     one channel: probability >= threshold gives 1, else 0;
#                    several channels: arg max over the channels
#   classification   softmax over the last axis (class probabilities)
#   other tasks      the raw output
# With logits=True the output is passed through the logistic function (one
# channel) or the softmax (several channels) first, so thresholds apply to
# probabilities.
#
# The mean and std defaults are the ImageNet statistics of RGB images scaled
# to [0, 1], as used by torchvision (Deng et al., 2009).
#
# References
# ----------
#   Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L. (2009).
#     ImageNet: a large-scale hierarchical image database. CVPR.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Record containers.
from dataclasses import dataclass, field

# Frameworks and tasks.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray


# Frameworks a model can come from.
class ModelFramework(str, Enum):
    # PyTorch modules and TorchScript files.
    PYTORCH = "pytorch"
    # ONNX files run with ONNX Runtime.
    ONNX = "onnx"
    # scikit-learn style estimators with a predict method.
    SKLEARN = "sklearn"
    # Any callable on NumPy batches.
    CUSTOM = "custom"


# Tasks a model can solve.
class ModelTask(str, Enum):
    # Object detection.
    DETECTION = "detection"
    # Per-pixel classes.
    SEGMENTATION = "segmentation"
    # One class per image.
    CLASSIFICATION = "classification"
    # Continuous values.
    REGRESSION = "regression"
    # Higher-resolution images.
    SUPER_RESOLUTION = "super_resolution"
    # Changes between two dates.
    CHANGE_DETECTION = "change_detection"


# Configuration of a model.
@dataclass
class ModelConfig:
    # Identifier.
    model_id: str
    # Human-readable name.
    name: str
    # Task.
    task: ModelTask
    # Framework.
    framework: ModelFramework = ModelFramework.PYTORCH
    # Number of input channels.
    input_channels: int = 3
    # Number of classes or output channels.
    num_classes: int = 1
    # Expected input height and width, when fixed.
    input_size: tuple[int, int] | None = None
    # Whether to standardise the inputs.
    normalize: bool = True
    # Per-channel means (one value or one per channel).
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    # Per-channel standard deviations (one value or one per channel).
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    # Probability threshold of binary segmentation.
    threshold: float = 0.5
    # Whether the model outputs logits rather than probabilities.
    logits: bool = False
    # Version of the model.
    version: str = "1.0.0"
    # Free-form tags.
    tags: dict[str, str] = field(default_factory=dict)

    # Validate the fields.
    def __post_init__(self) -> None:
        # Accept the task and framework as strings.
        self.task = ModelTask(self.task)
        # Framework.
        self.framework = ModelFramework(self.framework)
        # Channel counts are positive.
        if self.input_channels <= 0 or self.num_classes <= 0:
            # Explain the problem.
            raise ValueError("input_channels and num_classes must be positive")
        # Standard deviations must be positive to divide by them.
        if any(s <= 0 for s in self.std):
            # Explain the problem.
            raise ValueError(f"std values must be positive, got {self.std}")
        # One statistic, or one per channel, when standardising.
        for name, values in (("mean", self.mean), ("std", self.std)):
            # Check the length.
            if self.normalize and len(values) not in (1, self.input_channels):
                # Explain the problem.
                raise ValueError(f"{name} needs 1 or {self.input_channels} values")
        # Thresholds are probabilities.
        if not 0.0 <= self.threshold <= 1.0:
            # Explain the problem.
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


# Numerically stable logistic function.
def _sigmoid(x: NDArray[Any]) -> NDArray[np.float32]:
    # 1 / (1 + exp(-x)) written with tanh.
    return (0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))).astype(np.float32)


# Numerically stable softmax along an axis.
def _softmax(x: NDArray[Any], axis: int) -> NDArray[np.float32]:
    # Subtract the maximum so that exp does not overflow.
    z = np.exp(np.asarray(x, dtype=np.float64) - np.max(x, axis=axis, keepdims=True))
    # Normalise.
    return (z / np.sum(z, axis=axis, keepdims=True)).astype(np.float32)


# Model with framework-neutral pre- and postprocessing.
@dataclass
class ModelWrapper:
    # Configuration.
    config: ModelConfig
    # Loaded model: session, module, estimator or callable.
    model: Any = None
    # File the weights were loaded from.
    weights_path: Path | None = None

    # Load the weights when a path is given without a model.
    def __post_init__(self) -> None:
        # Load on construction.
        if self.weights_path is not None and self.model is None:
            # Read the file.
            self.load_weights(self.weights_path)

    # Load an ONNX file, a TorchScript archive or a torch.export program (.pt2).
    def load_weights(self, path: str | Path) -> None:
        # Path object.
        path = Path(path)
        # The file must exist.
        if not path.is_file():
            # Explain the problem.
            raise FileNotFoundError(f"no such model file: {path}")
        # ONNX models.
        if self.config.framework is ModelFramework.ONNX:
            # Imported lazily.
            import onnxruntime as ort

            # CPU session.
            self.model = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        # PyTorch models saved as TorchScript.
        elif self.config.framework is ModelFramework.PYTORCH:
            # Imported lazily.
            import torch

            # Programs saved with torch.export use the .pt2 extension.
            if path.suffix == ".pt2":
                # Module of the exported program.
                self.model = torch.export.load(str(path)).module()
            # Other files are TorchScript archives.
            else:
                # Load on the CPU in evaluation mode.
                self.model = torch.jit.load(str(path), map_location="cpu").eval()
        # Other frameworks cannot be loaded safely from files.
        else:
            # Pickled estimators can execute code, so they are not loaded here.
            raise ValueError(f"pass a fitted {self.config.framework.value} model as `model`")
        # Remember the file.
        self.weights_path = path

    # Batch of shape (N, C, H, W) from one image or a batch.
    def _as_batch(self, data: NDArray[Any]) -> tuple[NDArray[np.float32], bool]:
        # Float32 array.
        batch = np.asarray(data, dtype=np.float32)
        # Single images get a batch axis.
        single = batch.ndim == 3
        # Add it.
        if single:
            # Shape (1, C, H, W).
            batch = batch[np.newaxis]
        # Only images and batches are accepted.
        if batch.ndim != 4:
            # Explain the problem.
            raise ValueError(f"expected (C, H, W) or (N, C, H, W), got shape {np.shape(data)}")
        # Channel count of the model.
        expected = self.config.input_channels
        # The channel count must match.
        if batch.shape[1] != expected:
            # Explain the problem.
            raise ValueError(f"expected {expected} channels, got {batch.shape[1]}")
        # Batch and whether it was a single image.
        return batch, single

    # Standardise the channels of an image or batch.
    def preprocess(self, data: NDArray[Any]) -> NDArray[np.float32]:
        # Batch view.
        batch, single = self._as_batch(data)
        # Standardise when configured.
        if self.config.normalize:
            # Means broadcast over the channels.
            mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, -1, 1, 1)
            # Standard deviations broadcast over the channels.
            std = np.asarray(self.config.std, dtype=np.float32).reshape(1, -1, 1, 1)
            # (x - mean) / std.
            batch = (batch - mean) / std
        # Same layout as the input.
        return batch[0] if single else batch

    # Run the model on a preprocessed batch of shape (N, C, H, W).
    def _forward(self, batch: NDArray[np.float32]) -> NDArray[Any]:
        # The model.
        model = self.model
        # ONNX Runtime sessions.
        if hasattr(model, "get_inputs") and hasattr(model, "run"):
            # Name of the first input.
            name = model.get_inputs()[0].name
            # First output.
            return np.asarray(model.run(None, {name: batch})[0])
        # scikit-learn style estimators classify pixels.
        if hasattr(model, "predict") and not callable(getattr(model, "forward", None)):
            # Batch size, channels, height and width.
            n, c, h, w = batch.shape
            # Pixels as samples with the channels as features.
            samples = batch.transpose(0, 2, 3, 1).reshape(-1, c)
            # One prediction per pixel.
            return np.asarray(model.predict(samples)).reshape(n, 1, h, w)
        # PyTorch modules, including TorchScript.
        if callable(getattr(model, "forward", None)):
            # Imported lazily.
            import torch

            # Inference without gradients.
            with torch.no_grad():
                # Output tensor.
                output = model(torch.from_numpy(np.ascontiguousarray(batch)))
            # NumPy array.
            return output.detach().cpu().numpy()
        # Plain callables.
        if callable(model):
            # Output array.
            return np.asarray(model(batch))
        # Nothing to run.
        raise RuntimeError("Model not loaded")

    # Decode raw outputs of shape (N, K, ...) or (N, K).
    def postprocess(self, output: NDArray[Any]) -> NDArray[Any]:
        # Output array.
        out = np.asarray(output)
        # Segmentation maps.
        if self.config.task is ModelTask.SEGMENTATION and out.ndim >= 3:
            # Channel axis of (N, K, H, W) or (K, H, W).
            axis = 1 if out.ndim == 4 else 0
            # Binary segmentation.
            if out.shape[axis] == 1:
                # Probabilities.
                prob = _sigmoid(out) if self.config.logits else out
                # Threshold and drop the channel axis.
                return np.squeeze(prob >= self.config.threshold, axis=axis).astype(np.uint8)
            # Arg max over the classes (softmax does not change it).
            return np.argmax(out, axis=axis).astype(np.uint8)
        # Classification scores.
        if self.config.task is ModelTask.CLASSIFICATION:
            # Softmax of logits; probabilities are returned unchanged.
            return _softmax(out, axis=-1) if self.config.logits else out.astype(np.float32)
        # Other tasks return the raw output.
        return out

    # Preprocess, run and postprocess an image or batch.
    def predict(self, data: NDArray[Any]) -> NDArray[Any]:
        # Models must be loaded first.
        if self.model is None:
            # Explain the problem.
            raise RuntimeError("Model not loaded")
        # Batch view of the standardised input.
        batch, single = self._as_batch(self.preprocess(data))
        # Decoded output.
        result = self.postprocess(self._forward(batch))
        # Drop the batch axis of single images.
        return result[0] if single else result

    # Plain dictionary for listings.
    def to_dict(self) -> dict[str, Any]:
        # One entry per descriptive field.
        return {
            "model_id": self.config.model_id,  # Identifier.
            "name": self.config.name,  # Name.
            "task": self.config.task.value,  # Task.
            "framework": self.config.framework.value,  # Framework.
            "input_channels": self.config.input_channels,  # Inputs.
            "num_classes": self.config.num_classes,  # Outputs.
            "version": self.config.version,  # Version.
            "weights": str(self.weights_path) if self.weights_path else None,  # File.
        }  # End of the dictionary.


# =============================================================================
# End of module src/unbihexium/core/model.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
