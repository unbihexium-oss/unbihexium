# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Visualization module for result display.

This module provides functions for visualizing geospatial
analysis results and model outputs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def colorize_mask(
    mask: NDArray,
    colormap: dict[int, tuple[int, int, int]] | None = None,
) -> NDArray:
    """Convert class mask to RGB image.

    Args:
        mask: Class label array
        colormap: Dictionary mapping class IDs to RGB tuples

    Returns:
        RGB image array
    """
    if colormap is None:
        colormap = {
            0: (0, 0, 0),  # Background
            1: (255, 0, 0),  # Class 1
            2: (0, 255, 0),  # Class 2
            3: (0, 0, 255),  # Class 3
            4: (255, 255, 0),  # Class 4
            5: (255, 0, 255),  # Class 5
        }

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        rgb[mask == class_id] = color

    return rgb


def overlay_mask(
    image: NDArray,
    mask: NDArray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> NDArray:
    """Overlay binary mask on image.

    Args:
        image: RGB image array
        mask: Binary mask array
        alpha: Overlay transparency
        color: Mask color

    Returns:
        Blended RGB image
    """
    result = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = color[0] * mask
    mask_rgb[..., 1] = color[1] * mask
    mask_rgb[..., 2] = color[2] * mask

    mask_bool = mask > 0
    result[mask_bool] = ((1 - alpha) * result[mask_bool] + alpha * mask_rgb[mask_bool]).astype(
        np.uint8
    )

    return result


def create_legend(
    labels: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
    size: tuple[int, int] = (200, 20),
) -> NDArray:
    """Create color legend image.

    Args:
        labels: Class labels
        colors: Class colors
        size: Size of each legend entry

    Returns:
        Legend image array
    """
    n = len(labels)
    legend = np.ones((n * size[1], size[0] + 100, 3), dtype=np.uint8) * 255

    for i, (label, color) in enumerate(zip(labels, colors)):
        y = i * size[1]
        legend[y : y + size[1], : size[0]] = color

    return legend


def normalize_for_display(
    image: NDArray,
    percentile: tuple[float, float] = (2, 98),
) -> NDArray:
    """Normalize image for display using percentile stretch.

    Args:
        image: Input image array
        percentile: Low and high percentiles

    Returns:
        Normalized uint8 image
    """
    low = np.percentile(image, percentile[0])
    high = np.percentile(image, percentile[1])

    stretched = (image - low) / (high - low + 1e-8)
    stretched = np.clip(stretched, 0, 1)

    return (stretched * 255).astype(np.uint8)


__all__ = [
    "colorize_mask",
    "overlay_mask",
    "create_legend",
    "normalize_for_display",
]
