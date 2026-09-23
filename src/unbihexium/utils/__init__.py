# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Common utilities module.

This module provides common utility functions used throughout
the library.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import NDArray


def compute_sha256(path: str | Path) -> str:
    """Compute SHA256 hash of file.

    Args:
        path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def tile_image(
    image: NDArray,
    tile_size: int = 512,
    overlap: int = 64,
) -> Generator[tuple[NDArray, int, int], None, None]:
    """Tile image into overlapping patches.

    Args:
        image: Input image array (CHW or HW)
        tile_size: Size of each tile
        overlap: Overlap between tiles

    Yields:
        Tuple of (tile, row, col)
    """
    if image.ndim == 3:
        _, h, w = image.shape
    else:
        h, w = image.shape

    stride = tile_size - overlap

    for i in range(0, h - overlap, stride):
        for j in range(0, w - overlap, stride):
            # Handle edge cases
            end_i = min(i + tile_size, h)
            end_j = min(j + tile_size, w)
            start_i = max(0, end_i - tile_size)
            start_j = max(0, end_j - tile_size)

            if image.ndim == 3:
                tile = image[:, start_i:end_i, start_j:end_j]
            else:
                tile = image[start_i:end_i, start_j:end_j]

            yield tile, start_i, start_j


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Resolved path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bytes_to_human(size: int) -> str:
    """Convert bytes to human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Human-readable string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def count_parameters(model) -> int:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "compute_sha256",
    "tile_image",
    "ensure_dir",
    "bytes_to_human",
    "count_parameters",
]
