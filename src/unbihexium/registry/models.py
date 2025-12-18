"""Model registry for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unbihexium.core.model import ModelConfig, ModelWrapper


@dataclass
class ModelEntry:
    """Entry in the model registry."""

    model_id: str
    config: ModelConfig
    sha256: str = ""
    download_url: str | None = None
    size_bytes: int = 0
    license: str = "Apache-2.0"
    source: str = "release"  # repo, release, lfs, external

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.config.name,
            "task": self.config.task.value,
            "sha256": self.sha256,
            "download_url": self.download_url,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "source": self.source,
        }


class ModelRegistry:
    """Registry for ML models."""

    _models: dict[str, ModelEntry] = {}
    _loaded: dict[str, ModelWrapper] = {}

    @classmethod
    def register(cls, entry: ModelEntry) -> None:
        """Register a model."""
        cls._models[entry.model_id] = entry

    @classmethod
    def get(cls, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID."""
        return cls._models.get(model_id)

    @classmethod
    def list_all(cls) -> list[ModelEntry]:
        """List all registered models."""
        return list(cls._models.values())

    @classmethod
    def ids(cls) -> list[str]:
        """List all model IDs."""
        return list(cls._models.keys())

    @classmethod
    def load(cls, model_id: str) -> ModelWrapper | None:
        """Load a model from the registry."""
        if model_id in cls._loaded:
            return cls._loaded[model_id]
        entry = cls.get(model_id)
        if entry is None:
            return None
        # Loading logic would be implemented here
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear loaded model cache."""
        cls._loaded.clear()
