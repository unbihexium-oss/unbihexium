# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/postprocessing/tiles.py
# Title       : Tile windows and blended mosaicking of tiled predictions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Large scenes are predicted tile by tile with overlap; these functions
# create the windows and put the predictions back together:
#
#   tile_positions   (row, col) origins of overlapping tiles covering an
#                    image, the last row and column aligned with its edge
#   blend_weights    per-pixel weights of a tile: uniform or a linear ramp
#                    towards the tile edges
#   stitch_tiles     weighted average of overlapping tiles
#
# Method
# ------
# Predictions near tile borders see less context and are less reliable.
# With the "linear" blend each tile is weighted by a pyramid that falls
# linearly towards its edges over the overlap width, so that seams vanish
# (the feathering of image mosaicking, Burt and Adelson, 1983).
#
# References
# ----------
#   Burt, P. J., Adelson, E. H. (1983). A multiresolution spline with
#     application to image mosaics. ACM Transactions on Graphics 2(4),
#     217-236.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray


# Tile origins of one axis.
def _axis_positions(length: int, tile: int, step: int) -> list[int]:
    # A short axis needs a single tile.
    if length <= tile:
        # One tile at the origin.
        return [0]
    # Regular positions.
    positions = list(range(0, length - tile + 1, step))
    # Align the last tile with the edge.
    if positions[-1] != length - tile:
        # Add the edge-aligned tile.
        positions.append(length - tile)
    # Return the origins.
    return positions


# Origins of overlapping tiles that cover an image.
def tile_positions(
    shape: tuple[int, int],  # (H, W) of the image.
    tile_size: int | tuple[int, int],  # Tile height and width.
    overlap: int = 0,  # Overlap of neighbouring tiles in pixels.
) -> list[tuple[int, int]]:  # (row, col) origins in row-major order.
    # Tile height and width.
    th, tw = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    # The overlap must be smaller than the tile.
    if overlap < 0 or overlap >= min(th, tw):
        # Explain the requirement.
        raise ValueError("overlap must be between 0 and the tile size minus one")
    # Row origins.
    rows = _axis_positions(shape[0], th, th - overlap)
    # Column origins.
    cols = _axis_positions(shape[1], tw, tw - overlap)
    # Every combination.
    return [(r, c) for r in rows for c in cols]


# Per-pixel weights of a tile.
def blend_weights(
    tile_shape: tuple[int, int],  # (h, w) of the tile.
    overlap: int = 0,  # Width of the ramp in pixels.
    blend: str = "mean",  # "mean" (uniform) or "linear" (ramp).
) -> NDArray[np.float64]:  # (h, w) weights in (0, 1].
    # Uniform weights.
    if blend == "mean" or overlap == 0:
        # One everywhere.
        return np.ones(tile_shape)
    # Only the two blends exist.
    if blend != "linear":
        # Explain the accepted names.
        raise ValueError("blend must be 'mean' or 'linear'")
    # Distance of every row to the nearest tile edge, starting at 1.
    ry = np.minimum(np.arange(tile_shape[0]), np.arange(tile_shape[0])[::-1]) + 1.0
    # Same for the columns.
    rx = np.minimum(np.arange(tile_shape[1]), np.arange(tile_shape[1])[::-1]) + 1.0
    # Ramp that saturates at the overlap width.
    wy = np.minimum(ry / (overlap + 1.0), 1.0)
    # Same for the columns.
    wx = np.minimum(rx / (overlap + 1.0), 1.0)
    # Separable pyramid.
    return np.outer(wy, wx)


# Weighted average of overlapping tiles.
def stitch_tiles(
    tiles: Sequence[NDArray[Any]],  # Tiles of shape (h, w) or (C, h, w).
    positions: Sequence[tuple[int, int]],  # (row, col) origin of every tile.
    output_shape: tuple[int, ...],  # (H, W) or (C, H, W) of the mosaic.
    overlap: int = 0,  # Overlap used for the linear blend.
    blend: str = "mean",  # "mean" or "linear".
) -> NDArray[np.float64]:  # Mosaic; NaN where no tile contributes.
    # One position per tile.
    if len(tiles) != len(positions):
        # Explain the requirement.
        raise ValueError("tiles and positions must have the same length")
    # Sum of weighted values.
    total = np.zeros(output_shape, dtype=np.float64)
    # Sum of weights over the spatial axes.
    weights = np.zeros(output_shape[-2:], dtype=np.float64)
    # Add every tile.
    for tile, (r, c) in zip(tiles, positions):
        # Tile as float.
        t = np.asarray(tile, dtype=np.float64)
        # Tile height and width.
        h, w = t.shape[-2:]
        # The tile must lie inside the mosaic.
        if r < 0 or c < 0 or r + h > output_shape[-2] or c + w > output_shape[-1]:
            # Explain the problem.
            raise ValueError(f"tile at ({r}, {c}) with size ({h}, {w}) exceeds the output")
        # Weights of the tile.
        wt = blend_weights((h, w), overlap, blend)
        # Missing values of the tile do not contribute.
        valid = np.isfinite(t) if t.ndim == 2 else np.isfinite(t).all(axis=0)
        # Zero weight where missing.
        wt = np.where(valid, wt, 0.0)
        # Accumulate the weighted values.
        total[..., r : r + h, c : c + w] += np.where(valid, t, 0.0) * wt
        # Accumulate the weights.
        weights[r : r + h, c : c + w] += wt
    # Divide, NaN where nothing contributed.
    return np.divide(total, weights, out=np.full_like(total, np.nan), where=weights > 0)


# =============================================================================
# End of module src/unbihexium/postprocessing/tiles.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
