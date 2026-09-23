# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Preprocessing module for image preparation.

This module provides functions and classes for preprocessing
satellite imagery before model inference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class Normalize:
    """Normalize array values to specified range."""

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
        self.mean = np.array(mean) if mean else None
        self.std = np.array(std) if std else None
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image: NDArray) -> NDArray:
        if self.mean is not None and self.std is not None:
            return (image - self.mean) / self.std
        return (image - self.min_val) / (self.max_val - self.min_val + 1e-8)


class Resize:
    """Resize image to target size."""

    def __init__(self, size: tuple[int, int], interpolation: str = "bilinear"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image: NDArray) -> NDArray:
        from skimage.transform import resize

        return resize(image, self.size, mode="reflect", anti_aliasing=True)


class Pad:
    """Pad image to specified size."""

    def __init__(self, target_size: tuple[int, int], mode: str = "constant"):
        self.target_size = target_size
        self.mode = mode

    def __call__(self, image: NDArray) -> NDArray:
        h, w = image.shape[-2:]
        th, tw = self.target_size

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if image.ndim == 3:
            padding = ((0, 0), (0, pad_h), (0, pad_w))
        else:
            padding = ((0, pad_h), (0, pad_w))

        return np.pad(image, padding, mode=self.mode)


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: Sequence):
        self.transforms = transforms

    def __call__(self, image: NDArray) -> NDArray:
        for t in self.transforms:
            image = t(image)
        return image


def to_tensor(image: NDArray) -> NDArray:
    """Convert image to tensor format (CHW)."""
    if image.ndim == 2:
        return image[np.newaxis, ...]
    elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        return np.transpose(image, (2, 0, 1))
    return image


def from_tensor(tensor: NDArray) -> NDArray:
    """Convert tensor to image format (HWC)."""
    if tensor.ndim == 3:
        return np.transpose(tensor, (1, 2, 0))
    return tensor


__all__ = [
    "Normalize",
    "Resize",
    "Pad",
    "Compose",
    "to_tensor",
    "from_tensor",
]
