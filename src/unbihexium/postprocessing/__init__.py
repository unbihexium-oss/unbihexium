# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Postprocessing module for result refinement.

This module provides functions and classes for postprocessing
model outputs into usable geospatial products.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def threshold(predictions: NDArray, threshold: float = 0.5, above: bool = True) -> NDArray:
    """Apply threshold to predictions.

    Args:
        predictions: Prediction array
        threshold: Threshold value
        above: If True, values above threshold are 1

    Returns:
        Binary array
    """
    if above:
        return (predictions > threshold).astype(np.uint8)
    return (predictions < threshold).astype(np.uint8)


def argmax(predictions: NDArray, axis: int = 0) -> NDArray:
    """Apply argmax for multi-class predictions.

    Args:
        predictions: Prediction array with class probabilities
        axis: Axis along which to compute argmax

    Returns:
        Class label array
    """
    return np.argmax(predictions, axis=axis).astype(np.uint8)


def softmax(predictions: NDArray, axis: int = 0) -> NDArray:
    """Apply softmax to logits.

    Args:
        predictions: Logit array
        axis: Axis along which to compute softmax

    Returns:
        Probability array
    """
    exp_p = np.exp(predictions - np.max(predictions, axis=axis, keepdims=True))
    return exp_p / np.sum(exp_p, axis=axis, keepdims=True)


def sigmoid(predictions: NDArray) -> NDArray:
    """Apply sigmoid to logits.

    Args:
        predictions: Logit array

    Returns:
        Probability array
    """
    return 1 / (1 + np.exp(-predictions))


def morphology_clean(mask: NDArray, operation: str = "open", kernel_size: int = 3) -> NDArray:
    """Apply morphological operations to clean mask.

    Args:
        mask: Binary mask array
        operation: 'open', 'close', 'erode', 'dilate'
        kernel_size: Size of structuring element

    Returns:
        Cleaned mask array
    """
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

    ops = {
        "open": binary_opening,
        "close": binary_closing,
        "erode": binary_erosion,
        "dilate": binary_dilation,
    }

    structure = np.ones((kernel_size, kernel_size))
    return ops[operation](mask, structure=structure).astype(mask.dtype)


def remove_small_objects(mask: NDArray, min_size: int = 100) -> NDArray:
    """Remove small connected components from mask.

    Args:
        mask: Binary mask array
        min_size: Minimum object size in pixels

    Returns:
        Cleaned mask array
    """
    from scipy.ndimage import label

    labeled, num_features = label(mask)
    sizes = np.bincount(labeled.ravel())

    mask_sizes = sizes > min_size
    mask_sizes[0] = False  # Background

    return mask_sizes[labeled].astype(mask.dtype)


def stitch_tiles(
    tiles: list[NDArray],
    positions: list[tuple[int, int]],
    output_shape: tuple[int, ...],
    overlap: int = 0,
) -> NDArray:
    """Stitch tiles back into full image.

    Args:
        tiles: List of tile arrays
        positions: List of (row, col) positions
        output_shape: Shape of output array
        overlap: Overlap between tiles

    Returns:
        Stitched array
    """
    output = np.zeros(output_shape, dtype=tiles[0].dtype)
    weights = np.zeros(output_shape[:2], dtype=np.float32)

    for tile, (r, c) in zip(tiles, positions):
        h, w = tile.shape[-2:]
        if tile.ndim == 3:
            output[:, r : r + h, c : c + w] += tile
        else:
            output[r : r + h, c : c + w] += tile
        weights[r : r + h, c : c + w] += 1

    weights = np.maximum(weights, 1)
    if output.ndim == 3:
        return output / weights[np.newaxis, ...]
    return output / weights


__all__ = [
    "threshold",
    "argmax",
    "softmax",
    "sigmoid",
    "morphology_clean",
    "remove_small_objects",
    "stitch_tiles",
]
