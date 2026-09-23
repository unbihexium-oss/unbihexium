# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/tiling.py
# Title       : Overlapping tiles of large images and their mosaic
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Scenes are usually too large for one pass of an algorithm, so they are cut
# into overlapping tiles that are processed separately and merged again:
#
#   tile_starts     start offsets along one axis
#   tile_windows    (row, col, height, width) windows covering an image
#   tile_image      generator of (tile, row, col) as in earlier releases
#   merge_tiles     mosaic of processed tiles; overlapping pixels are
#                   averaged, optionally with weights
#
# Method
# ------
# Tiles are placed with a step of tile_size - overlap. The last tile of an
# axis is shifted back so that it ends exactly at the image border, which
# keeps every tile at full size (no padding) and covers every pixel. Images
# smaller than a tile give one tile of the image size. Averaging the
# overlaps suppresses the seams that appear at tile borders in
# convolutional predictions (Huang et al., 2018).
#
# References
# ----------
# Huang, B., Reichman, D., Collins, L. M., Bradbury, K. and Malof, J. M.
# (2018). Tiling and stitching segmentation output for remote sensing: basic
# challenges and recommendations. arXiv:1805.12219.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Iterables of tiles.
from collections.abc import Iterable, Iterator

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Window: top row, left column, height and width.
Window = tuple[int, int, int, int]


# Start offsets of tiles of length `tile` along an axis of length `size`.
def tile_starts(size: int, tile: int, overlap: int = 0) -> list[int]:
    # Validate the sizes.
    if size < 1 or tile < 1:
        # Explain the problem.
        raise ValueError(f"size and tile must be positive, got {size} and {tile}")
    # The overlap must leave a positive step.
    if not 0 <= overlap < tile:
        # Explain the problem.
        raise ValueError(f"overlap must be in [0, tile), got {overlap} for tile {tile}")
    # Small axes hold one tile.
    if size <= tile:
        # Single tile at the origin.
        return [0]
    # Step between tiles.
    step = tile - overlap
    # Regular starts that keep the tile inside the image.
    starts = list(range(0, size - tile + 1, step))
    # Shift a final tile back to the border if the last one stops short.
    if starts[-1] + tile < size:
        # Tile that ends at the border.
        starts.append(size - tile)
    # Return the starts.
    return starts


# Windows (row, col, height, width) of the tiles of an image.
def tile_windows(height: int, width: int, tile_size: int, overlap: int = 0) -> list[Window]:
    # Row starts.
    rows = tile_starts(height, tile_size, overlap)
    # Column starts.
    cols = tile_starts(width, tile_size, overlap)
    # Tile height, clipped for small images.
    th = min(tile_size, height)
    # Tile width, clipped for small images.
    tw = min(tile_size, width)
    # Row-major list of windows.
    return [(r, c, th, tw) for r in rows for c in cols]


# Generator of (tile, row, col) for (bands, H, W) or (H, W) images.
def tile_image(
    image: NDArray[Any],  # Image array.
    tile_size: int = 512,  # Tile side in pixels.
    overlap: int = 64,  # Overlap between neighbouring tiles in pixels.
) -> Iterator[tuple[NDArray[Any], int, int]]:  # Tiles with their upper-left corner.
    # Only 2-D and 3-D arrays are images.
    if image.ndim not in (2, 3):
        # Explain the expected layout.
        raise ValueError(f"image must be (H, W) or (bands, H, W), got shape {image.shape}")
    # Spatial size.
    height, width = image.shape[-2:]
    # Visit every window.
    for row, col, th, tw in tile_windows(height, width, tile_size, overlap):
        # Cut the window from the last two axes.
        yield image[..., row : row + th, col : col + tw], row, col


# Mosaic of tiles; overlapping pixels are the (weighted) mean of the tiles.
def merge_tiles(
    tiles: Iterable[NDArray[Any]],  # Processed tiles, (H, W) or (bands, H, W).
    windows: Iterable[tuple[int, int] | Window],  # Upper-left corner of each tile.
    shape: tuple[int, ...],  # Shape of the mosaic, (H, W) or (bands, H, W).
    weights: NDArray[Any] | None = None,  # Optional per-pixel weights of a tile.
) -> NDArray[np.float32]:  # Mosaic; pixels covered by no tile are NaN.
    # Weighted sum of the tiles.
    total = np.zeros(shape, dtype=np.float64)
    # Sum of the weights per pixel.
    norm = np.zeros(shape[-2:], dtype=np.float64)
    # Add every tile.
    for tile, window in zip(tiles, windows):
        # Tile values.
        values = np.asarray(tile, dtype=np.float64)
        # Upper-left corner.
        row, col = int(window[0]), int(window[1])
        # Spatial size of the tile.
        th, tw = values.shape[-2:]
        # Tiles must fit the mosaic.
        if row < 0 or col < 0 or row + th > shape[-2] or col + tw > shape[-1]:
            # Explain the problem.
            raise ValueError(f"tile at ({row}, {col}) of size {th}x{tw} exceeds shape {shape}")
        # Weights of this tile, uniform by default.
        w = np.ones((th, tw)) if weights is None else np.asarray(weights, dtype=np.float64)
        # Weights must match the tile.
        if w.shape != (th, tw):
            # Explain the problem.
            raise ValueError(f"weights of shape {w.shape} do not match tile {th}x{tw}")
        # Missing tile values do not contribute.
        valid = np.isfinite(values)
        # Weight per value, zero where the value is missing.
        wv = np.where(valid, w, 0.0)
        # Accumulate the weighted values.
        total[..., row : row + th, col : col + tw] += np.where(valid, values, 0.0) * wv
        # Accumulate the weights (per pixel, from the first band of the tile).
        norm[row : row + th, col : col + tw] += wv if wv.ndim == 2 else wv[0]
    # Mean where at least one tile contributed.
    return np.divide(total, norm, out=np.full(shape, np.nan), where=norm > 0).astype(np.float32)


# =============================================================================
# End of module src/unbihexium/utils/tiling.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
