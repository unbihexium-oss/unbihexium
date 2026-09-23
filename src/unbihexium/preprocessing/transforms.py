# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/transforms.py
# Title       : Composable array transforms and band normalisation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and scikit-image
# =============================================================================
#
# Abstract
# --------
# Small callable transforms that prepare image arrays for model inference,
# and NaN-aware band normalisation:
#
#   Normalize            per-band standardisation or min-max scaling
#   Resize, Pad          change the spatial size of (H, W) or (C, H, W) arrays
#   Compose              apply several transforms in order
#   to_tensor            (H, W) or (H, W, C) to (C, H, W)
#   from_tensor          (C, H, W) to (H, W, C)
#   minmax_normalize     per-band scaling to [0, 1], ignoring NaN and nodata
#   standardize          per-band z-scores, ignoring NaN and nodata
#
# Multi-band arrays use the band-first layout (C, H, W) throughout the
# package; statistics are always computed per band over finite values.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and callables.
from typing import Any, Callable, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray


# Float copy of an array with nodata values replaced by NaN.
def as_float(array: NDArray[Any], nodata: float | None = None) -> NDArray[np.float64]:
    # Convert to float64 so that NaN can mark invalid values.
    out = np.array(array, dtype=np.float64, copy=True)
    # Replace the nodata value when one is given.
    if nodata is not None and not np.isnan(nodata):
        # Pixels equal to nodata become NaN.
        out[out == nodata] = np.nan
    # Return the float copy.
    return out


# View an array as (C, H, W), adding a band axis to single-band images.
def band_first(array: NDArray[Any]) -> NDArray[Any]:
    # Arrays of rank two are single-band images.
    if array.ndim == 2:
        # Add a leading band axis.
        return array[np.newaxis]
    # Only rank-three arrays remain valid.
    if array.ndim != 3:
        # Explain the accepted layouts.
        raise ValueError(f"expected an (H, W) or (C, H, W) array, got shape {array.shape}")
    # Return the array unchanged.
    return array


# Per-band standardisation with fixed statistics, or min-max scaling.
class Normalize:
    # Store the statistics.
    def __init__(
        self,  # The transform.
        mean: Sequence[float] | None = None,  # Per-band means for standardisation.
        std: Sequence[float] | None = None,  # Per-band standard deviations.
        min_val: float = 0.0,  # Lower bound of min-max scaling.
        max_val: float = 1.0,  # Upper bound of min-max scaling.
    ) -> None:  # The constructor returns nothing.
        # Means as an array, or None for min-max scaling.
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        # Standard deviations as an array.
        self.std = None if std is None else np.asarray(std, dtype=np.float64)
        # Both statistics are needed together.
        if (self.mean is None) != (self.std is None):
            # Explain the requirement.
            raise ValueError("mean and std must be given together")
        # Standard deviations must be positive.
        if self.std is not None and np.any(self.std <= 0):
            # Explain the requirement.
            raise ValueError("std must be positive")
        # A min-max range must not be empty.
        if max_val <= min_val:
            # Explain the requirement.
            raise ValueError("max_val must be greater than min_val")
        # Lower bound of the scaling.
        self.min_val = min_val
        # Upper bound of the scaling.
        self.max_val = max_val

    # Apply the transform to an (H, W) or (C, H, W) array.
    def __call__(self, image: NDArray[Any]) -> NDArray[np.float64]:
        # Work in float64.
        x = np.asarray(image, dtype=np.float64)
        # Min-max scaling when no statistics are given.
        if self.mean is None or self.std is None:
            # Map [min_val, max_val] onto [0, 1].
            return (x - self.min_val) / (self.max_val - self.min_val)
        # Statistics are broadcast along the band axis of (C, H, W) arrays.
        shape = (-1, 1, 1) if x.ndim == 3 else self.mean.shape
        # Standardise.
        return (x - self.mean.reshape(shape)) / self.std.reshape(shape)


# Resize an (H, W) or (C, H, W) array to a target (height, width).
class Resize:
    # Interpolation names and their spline orders.
    ORDERS = {"nearest": 0, "bilinear": 1, "bicubic": 3}

    # Store the target size.
    def __init__(self, size: tuple[int, int], interpolation: str = "bilinear") -> None:
        # The interpolation must be known.
        if interpolation not in self.ORDERS:
            # Explain the accepted names.
            raise ValueError(f"interpolation must be one of {sorted(self.ORDERS)}")
        # Target height and width.
        self.size = (int(size[0]), int(size[1]))
        # Interpolation name.
        self.interpolation = interpolation

    # Resize the array, keeping the band axis.
    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        # Resampling routine of scikit-image.
        from skimage.transform import resize

        # Spline order of the interpolation.
        order = self.ORDERS[self.interpolation]
        # Output shape: bands are kept.
        shape = self.size if image.ndim == 2 else (image.shape[0], *self.size)
        # Nearest-neighbour keeps labels, so it disables anti-aliasing.
        return resize(
            image,  # Input array.
            shape,  # Output shape.
            order=order,  # Spline order.
            mode="edge",  # Extend border values.
            anti_aliasing=order > 0,  # Smooth before downsampling.
            preserve_range=True,  # Keep the physical values.
        ).astype(image.dtype if order == 0 else np.float64)  # Labels keep their type.


# Pad an (H, W) or (C, H, W) array at the bottom and right to a target size.
class Pad:
    # Store the target size.
    def __init__(self, target_size: tuple[int, int], mode: str = "constant") -> None:
        # Target height and width.
        self.target_size = target_size
        # NumPy padding mode.
        self.mode = mode

    # Pad the array.
    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        # Current height and width.
        h, w = image.shape[-2:]
        # Target height and width.
        th, tw = self.target_size
        # Rows to add, never negative.
        pad_h = max(0, th - h)
        # Columns to add.
        pad_w = max(0, tw - w)
        # Leading axes are not padded.
        padding = [(0, 0)] * (image.ndim - 2) + [(0, pad_h), (0, pad_w)]
        # Pad the array.
        return np.pad(image, padding, mode=self.mode)


# Apply several transforms in order.
class Compose:
    # Store the transforms.
    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        # Transforms to apply.
        self.transforms = list(transforms)

    # Apply every transform.
    def __call__(self, image: Any) -> Any:
        # Pass the image through the chain.
        for transform in self.transforms:
            # Output of one transform is the input of the next.
            image = transform(image)
        # Return the result.
        return image


# Convert an (H, W) or (H, W, C) image to the band-first (C, H, W) layout.
def to_tensor(image: NDArray[Any]) -> NDArray[Any]:
    # Single-band images get a band axis.
    if image.ndim == 2:
        # Add the axis in front.
        return image[np.newaxis, ...]
    # Channel-last images with up to four channels are transposed.
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        # Move the channels to the front.
        return np.transpose(image, (2, 0, 1))
    # Other arrays are returned unchanged.
    return image


# Convert a band-first (C, H, W) array to the channel-last (H, W, C) layout.
def from_tensor(tensor: NDArray[Any]) -> NDArray[Any]:
    # Rank-three arrays are transposed.
    if tensor.ndim == 3:
        # Move the channels to the back.
        return np.transpose(tensor, (1, 2, 0))
    # Other arrays are returned unchanged.
    return tensor


# Per-band min-max scaling to [0, 1] over finite values.
def minmax_normalize(image: NDArray[Any], nodata: float | None = None) -> NDArray[np.float64]:
    # Float copy with nodata as NaN.
    x = band_first(as_float(image, nodata))
    # Band minima over finite values.
    lo = np.nanmin(x, axis=(1, 2), keepdims=True)
    # Band maxima over finite values.
    hi = np.nanmax(x, axis=(1, 2), keepdims=True)
    # Width of the range; constant bands map to zero.
    span = np.where(hi > lo, hi - lo, 1.0)
    # Scale every band.
    out = (x - lo) / span
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# Per-band z-scores over finite values.
def standardize(image: NDArray[Any], nodata: float | None = None) -> NDArray[np.float64]:
    # Float copy with nodata as NaN.
    x = band_first(as_float(image, nodata))
    # Band means over finite values.
    mean = np.nanmean(x, axis=(1, 2), keepdims=True)
    # Band standard deviations (population).
    std = np.nanstd(x, axis=(1, 2), keepdims=True)
    # Constant bands keep zero after centring.
    out = (x - mean) / np.where(std > 0, std, 1.0)
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# =============================================================================
# End of module src/unbihexium/preprocessing/transforms.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
