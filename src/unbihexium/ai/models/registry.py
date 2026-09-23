# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model registry for architecture management.

This module provides a registry system for mapping model IDs to their
architectures, configurations, and weight loaders.

Features:
    - Model registration with metadata
    - Weight loading with architecture validation
    - Model instantiation by ID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class ModelEntry:
    """Entry in the model registry.

    Attributes:
        model_id: Unique model identifier.
        architecture: Architecture class or factory function.
        config: Default configuration for the model.
        task: Task type (detection, segmentation, classification, super_resolution).
        description: Human-readable description.
    """

    model_id: str
    architecture: type | Callable[..., Any]
    config: Any = None
    task: str = "unknown"
    description: str = ""


class ModelRegistry:
    """Registry for model architectures and weights.

    Provides a centralized way to manage model definitions and
    instantiate models by ID.
    """

    _registry: dict[str, ModelEntry] = {}

    @classmethod
    def register(
        cls,
        model_id: str,
        architecture: type | Callable[..., Any],
        config: Any = None,
        task: str = "unknown",
        description: str = "",
    ) -> None:
        """Register a model architecture.

        Args:
            model_id: Unique model identifier.
            architecture: Architecture class or factory.
            config: Default configuration.
            task: Task type.
            description: Model description.
        """
        cls._registry[model_id] = ModelEntry(
            model_id=model_id,
            architecture=architecture,
            config=config,
            task=task,
            description=description,
        )

    @classmethod
    def get(cls, model_id: str) -> ModelEntry:
        """Get a registered model entry.

        Args:
            model_id: Model identifier.

        Returns:
            ModelEntry for the requested model.

        Raises:
            KeyError: If model is not registered.
        """
        if model_id not in cls._registry:
            raise KeyError(
                f"Model '{model_id}' not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[model_id]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model IDs."""
        return list(cls._registry.keys())

    @classmethod
    def list_by_task(cls, task: str) -> list[str]:
        """List models for a specific task.

        Args:
            task: Task type (detection, segmentation, etc.).

        Returns:
            List of model IDs for the task.
        """
        return [entry.model_id for entry in cls._registry.values() if entry.task == task]

    @classmethod
    def instantiate(
        cls,
        model_id: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a model by ID.

        Args:
            model_id: Model identifier.
            config: Override configuration (uses default if None).
            **kwargs: Additional arguments for model constructor.

        Returns:
            Instantiated model.
        """
        entry = cls.get(model_id)
        cfg = config if config is not None else entry.config
        return entry.architecture(cfg, **kwargs) if cfg else entry.architecture(**kwargs)

    @classmethod
    def load_weights(
        cls,
        model_id: str,
        weights_path: str | Path,
        strict: bool = True,
    ) -> Any:
        """Load model with weights.

        Args:
            model_id: Model identifier.
            weights_path: Path to weights file.
            strict: Whether to strictly enforce state_dict matching.

        Returns:
            Model with loaded weights.

        Raises:
            ValueError: If weights are incompatible with architecture.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for weight loading")

        model = cls.instantiate(model_id)
        weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Validate keys if strict
        if strict:
            model_keys = set(model.state_dict().keys())
            weight_keys = set(state_dict.keys())

            missing = model_keys - weight_keys
            unexpected = weight_keys - model_keys

            if missing:
                raise ValueError(f"Missing keys in weights: {missing}")
            if unexpected:
                raise ValueError(f"Unexpected keys in weights: {unexpected}")

        model.load_state_dict(state_dict, strict=strict)
        return model


def _register_default_models() -> None:
    """Register default model architectures."""
    from unbihexium.ai.models.resnet import ResNet, ResNetConfig
    from unbihexium.ai.models.unet import UNet, UNetConfig

    # UNet models
    ModelRegistry.register(
        model_id="unet_segmentation",
        architecture=UNet,
        config=UNetConfig(num_classes=2),
        task="segmentation",
        description="UNet for binary segmentation",
    )

    ModelRegistry.register(
        model_id="unet_landcover",
        architecture=UNet,
        config=UNetConfig(num_classes=10),
        task="segmentation",
        description="UNet for land cover classification",
    )

    # ResNet models
    ModelRegistry.register(
        model_id="resnet18_scene",
        architecture=ResNet,
        config=ResNetConfig(variant=18, num_classes=10),
        task="classification",
        description="ResNet18 for scene classification",
    )

    ModelRegistry.register(
        model_id="resnet50_feature",
        architecture=ResNet,
        config=ResNetConfig(variant=50, num_classes=1000),
        task="feature_extraction",
        description="ResNet50 feature extractor",
    )


# Register defaults on import
try:
    _register_default_models()
except ImportError:
    pass  # Dependencies not available
