"""Model wrapper abstraction for ML models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ModelFramework(str, Enum):
    """Supported ML frameworks."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


class ModelTask(str, Enum):
    """Model task types."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SUPER_RESOLUTION = "super_resolution"
    CHANGE_DETECTION = "change_detection"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    name: str
    task: ModelTask
    framework: ModelFramework = ModelFramework.PYTORCH
    input_channels: int = 3
    num_classes: int = 1
    input_size: tuple[int, int] | None = None
    normalize: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    threshold: float = 0.5
    version: str = "1.0.0"
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelWrapper:
    """Wrapper for ML models with preprocessing and postprocessing."""

    config: ModelConfig
    model: Any = None
    weights_path: Path | None = None

    def __post_init__(self) -> None:
        if self.weights_path and self.model is None:
            self.load_weights(self.weights_path)

    def load_weights(self, path: Path) -> None:
        """Load model weights from file."""
        self.weights_path = path
        # Framework-specific loading handled by subclasses

    def preprocess(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Preprocess input data."""
        if not self.config.normalize:
            return data
        mean = np.array(self.config.mean).reshape(-1, 1, 1)
        std = np.array(self.config.std).reshape(-1, 1, 1)
        return (data - mean) / std

    def postprocess(self, output: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Postprocess model output."""
        if self.config.task == ModelTask.SEGMENTATION:
            return (output > self.config.threshold).astype(np.float32)
        if self.config.task == ModelTask.DETECTION:
            return output
        return output

    def predict(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Run inference on input data."""
        preprocessed = self.preprocess(data)
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
        output = self._forward(preprocessed)
        return self.postprocess(output)

    def _forward(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Forward pass through the model."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "name": self.config.name,
            "task": self.config.task.value,
            "framework": self.config.framework.value,
            "input_channels": self.config.input_channels,
            "num_classes": self.config.num_classes,
            "version": self.config.version,
        }
