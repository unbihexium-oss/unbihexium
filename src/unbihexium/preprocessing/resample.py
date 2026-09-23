# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/resample.py
# Title       : Resampling and block aggregation of raster arrays
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Change of pixel size for arrays on a regular grid, keeping the footprint
# of the image:
#
#   resample    interpolate (H, W) or (C, H, W) arrays to a new shape with
#               nearest, bilinear or cubic splines; missing values are
#               handled by normalised convolution
#   aggregate   reduce by an integer factor with mean, sum, min, max,
#               median or mode (majority, for class maps), ignoring NaN
#   scaled_transform
#               affine geotransform of the resampled grid
#
# Method
# ------
# Interpolation uses pixel-area alignment (grid_mode in SciPy): the outer
# edges of the first and last pixels stay fixed, as in GDAL. With missing
# values, the interpolated data and the interpolated validity weights are
# divided, so that NaN does not spread across the image (Knutsson and
# Westin, 1993); output pixels whose weight is below one half are missing.
#
# References
# ----------
#   Knutsson, H., Westin, C.-F. (1993). Normalized and differential
#     convolution. Proceedings of IEEE CVPR, 515-523.
#   Keys, R. (1981). Cubic convolution interpolation for digital image
#     processing. IEEE Transactions on Acoustics, Speech, and Signal
#     Processing 29(6), 1153-1160.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Silence expected warnings of empty blocks.
import warnings

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Spline interpolation.
from scipy.ndimage import zoom

# Float conversion and band layout helpers.
from unbihexium.preprocessing.transforms import as_float, band_first

# Spline order of each interpolation method.
RESAMPLING_ORDERS = {"nearest": 0, "bilinear": 1, "cubic": 3}

# Reductions of the aggregate function.
AGGREGATIONS = ("mean", "sum", "min", "max", "median", "mode")


# Interpolate an (H, W) or (C, H, W) array to a new (height, width).
def resample(
    image: NDArray[Any],  # Input array.
    shape: tuple[int, int],  # Output (height, width).
    method: str = "bilinear",  # nearest, bilinear or cubic.
    nodata: float | None = None,  # Value treated as missing.
) -> NDArray[Any]:  # Resampled array; float unless nearest.
    # The method must be known.
    if method not in RESAMPLING_ORDERS:
        # Explain the accepted names.
        raise ValueError(f"method must be one of {sorted(RESAMPLING_ORDERS)}")
    # The output must have pixels.
    if shape[0] < 1 or shape[1] < 1:
        # Explain the requirement.
        raise ValueError("shape must be positive")
    # Input in band-first layout.
    x = band_first(np.asarray(image))
    # Zoom factors; bands are kept.
    factors = (1.0, shape[0] / x.shape[1], shape[1] / x.shape[2])
    # Spline order of the method.
    order = RESAMPLING_ORDERS[method]
    # Nearest neighbour keeps the values and the data type.
    if order == 0:
        # Pixel-area aligned nearest neighbour.
        out = zoom(x, factors, order=0, mode="nearest", grid_mode=True)
    # Interpolating methods work in float with missing values as NaN.
    else:
        # Float copy with nodata as NaN.
        xf = as_float(x, nodata)
        # Validity weights.
        weight = np.isfinite(xf).astype(np.float64)
        # Data with missing values set to zero.
        data = np.where(weight > 0, xf, 0.0)
        # Interpolated data.
        num = zoom(data, factors, order=order, mode="grid-mirror", grid_mode=True)
        # Interpolated weights.
        den = zoom(weight, factors, order=order, mode="grid-mirror", grid_mode=True)
        # Normalised values where enough valid input contributes.
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= 0.5)
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# Mode of the last axis of an array, ignoring NaN; ties go to the smallest value.
def _mode_last_axis(blocks: NDArray[np.float64]) -> NDArray[np.float64]:
    # Sort every block so that equal values are adjacent; NaN goes last.
    s = np.sort(blocks, axis=-1)
    # Number of values per block.
    n = s.shape[-1]
    # Output with NaN for blocks without data.
    out = np.full(s.shape[:-1], np.nan)
    # Largest run length seen so far per block.
    best = np.zeros(s.shape[:-1], dtype=np.int64)
    # Length of the current run of equal values.
    run = np.zeros(s.shape[:-1], dtype=np.int64)
    # Walk along the sorted values.
    for i in range(n):
        # Current value.
        v = s[..., i]
        # A run continues when the value equals its predecessor.
        same = (v == s[..., i - 1]) if i > 0 else np.zeros_like(v, dtype=bool)
        # Extend or restart the run; NaN never counts.
        run = np.where(np.isnan(v), 0, np.where(same, run + 1, 1))
        # Strictly longer runs replace the mode, so ties keep the smaller value.
        better = run > best
        # Update the longest run.
        best = np.where(better, run, best)
        # Update the mode.
        out = np.where(better, v, out)
    # Return the modes.
    return out


# Reduce an (H, W) or (C, H, W) array by an integer factor.
def aggregate(
    image: NDArray[Any],  # Input array.
    factor: int,  # Pixels per output pixel along each axis.
    method: str = "mean",  # mean, sum, min, max, median or mode.
    nodata: float | None = None,  # Value treated as missing.
    trim: bool = True,  # Drop rows and columns that do not fill a block.
) -> NDArray[np.float64]:  # Aggregated array with NaN for empty blocks.
    # The method must be known.
    if method not in AGGREGATIONS:
        # Explain the accepted names.
        raise ValueError(f"method must be one of {AGGREGATIONS}")
    # The factor must be a positive integer.
    if int(factor) != factor or factor < 1:
        # Explain the requirement.
        raise ValueError("factor must be a positive integer")
    # Float copy in band-first layout with nodata as NaN.
    x = band_first(as_float(image, nodata))
    # Bands, rows and columns.
    c, h, w = x.shape
    # Rows and columns of the output.
    oh, ow = (h // factor, w // factor) if trim else (-(-h // factor), -(-w // factor))
    # Without trimming, pad partial blocks with NaN.
    if not trim:
        # Padding at the bottom and right.
        x = np.pad(x, ((0, 0), (0, oh * factor - h), (0, ow * factor - w)), constant_values=np.nan)
    # The output must not be empty.
    if oh == 0 or ow == 0:
        # Explain the requirement.
        raise ValueError("factor is larger than the image")
    # Blocks as (C, oh, ow, factor * factor).
    blocks = (
        x[:, : oh * factor, : ow * factor]  # Whole blocks only.
        .reshape(c, oh, factor, ow, factor)  # Split rows and columns.
        .transpose(0, 1, 3, 2, 4)  # Move the block axes last.
        .reshape(c, oh, ow, factor * factor)  # Flatten each block.
    )  # End of the block view.
    # Blocks with at least one valid value.
    has_data = np.isfinite(blocks).any(axis=-1)
    # Silence the warnings of all-NaN blocks; they are set to NaN below.
    with warnings.catch_warnings():
        # Ignore "Mean of empty slice" and similar warnings.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Majority for class maps.
        if method == "mode":
            # Most frequent value.
            out = _mode_last_axis(blocks)
        # Sum of valid values.
        elif method == "sum":
            # nansum returns zero for empty blocks.
            out = np.nansum(blocks, axis=-1)
        # Other reductions.
        else:
            # NaN-aware NumPy reduction of the given name.
            out = getattr(np, f"nan{method}")(blocks, axis=-1)
    # Empty blocks are missing.
    out = np.where(has_data, out, np.nan)
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# Geotransform of a grid resampled from (H, W) to a new shape.
def scaled_transform(
    transform: tuple[float, float, float, float, float, float],  # (a, b, c, d, e, f).
    old_shape: tuple[int, int],  # (H, W) of the input grid.
    new_shape: tuple[int, int],  # (height, width) of the output grid.
) -> tuple[float, float, float, float, float, float]:  # Transform of the output grid.
    # Affine coefficients x = a col + b row + c, y = d col + e row + f.
    a, b, c, d, e, f = (float(v) for v in transform[:6])
    # Scale of the columns.
    sx = old_shape[1] / new_shape[1]
    # Scale of the rows.
    sy = old_shape[0] / new_shape[0]
    # The origin is fixed; pixel vectors are scaled.
    return (a * sx, b * sy, c, d * sx, e * sy, f)


# =============================================================================
# End of module src/unbihexium/preprocessing/resample.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
