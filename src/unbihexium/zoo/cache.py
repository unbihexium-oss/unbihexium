"""Model cache management.

Provides caching functionality for downloaded models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_cache_dir() -> Path:
    """Get the default model cache directory.

    Uses UNBIHEXIUM_CACHE environment variable if set,
    otherwise uses ~/.cache/unbihexium/models.

    Returns:
        Path to cache directory.
    """
    cache_env = os.environ.get("UNBIHEXIUM_CACHE")
    if cache_env:
        return Path(cache_env) / "models"

    home = Path.home()
    return home / ".cache" / "unbihexium" / "models"


def ensure_cache_dir() -> Path:
    """Ensure cache directory exists and return path.

    Returns:
        Path to cache directory.
    """
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_cache_path(model_id: str) -> Path:
    """Get cache path for a specific model.

    Args:
        model_id: Model identifier.

    Returns:
        Path to model cache directory.
    """
    return get_cache_dir() / model_id


def is_model_cached(model_id: str) -> bool:
    """Check if a model is cached.

    Args:
        model_id: Model identifier.

    Returns:
        True if model is cached with required files.
    """
    model_path = get_model_cache_path(model_id)
    if not model_path.exists():
        return False

    required_file = model_path / "model.onnx"
    return required_file.exists()


def get_cached_model_path(model_id: str) -> Path | None:
    """Get path to cached model if available.

    Args:
        model_id: Model identifier.

    Returns:
        Path to model directory if cached, None otherwise.
    """
    if is_model_cached(model_id):
        return get_model_cache_path(model_id)
    return None


def clear_model_cache(model_id: str) -> bool:
    """Clear cache for a specific model.

    Args:
        model_id: Model identifier.

    Returns:
        True if cache was cleared.
    """
    import shutil

    model_path = get_model_cache_path(model_id)
    if model_path.exists():
        shutil.rmtree(model_path)
        return True
    return False


def clear_all_cache() -> int:
    """Clear entire model cache.

    Returns:
        Number of models cleared.
    """
    import shutil

    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    for item in cache_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            count += 1

    return count


def get_cache_info() -> dict[str, Any]:
    """Get information about cached models.

    Returns:
        Dictionary with cache statistics.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return {"path": str(cache_dir), "models": [], "total_size_mb": 0}

    models = []
    total_size = 0

    for item in cache_dir.iterdir():
        if item.is_dir():
            model_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            total_size += model_size
            models.append({
                "model_id": item.name,
                "size_mb": round(model_size / (1024 * 1024), 2),
                "files": [f.name for f in item.iterdir() if f.is_file()],
            })

    return {
        "path": str(cache_dir),
        "models": models,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


class ModelCache:
    """Model cache manager with persistence."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Custom cache directory.
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        """Load cache index from disk."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"models": {}}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register(self, model_id: str, metadata: dict[str, Any]) -> None:
        """Register a cached model.

        Args:
            model_id: Model identifier.
            metadata: Model metadata.
        """
        self._index["models"][model_id] = {
            "path": str(self.cache_dir / model_id),
            **metadata,
        }
        self._save_index()

    def get(self, model_id: str) -> dict[str, Any] | None:
        """Get cached model info.

        Args:
            model_id: Model identifier.

        Returns:
            Model info if cached, None otherwise.
        """
        return self._index["models"].get(model_id)

    def list(self) -> list[str]:
        """List all cached model IDs.

        Returns:
            List of cached model IDs.
        """
        return list(self._index["models"].keys())
