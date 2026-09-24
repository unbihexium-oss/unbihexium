# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/enhancement.py
# Title       : Contrast stretches and histogram matching
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Radiometric enhancement of (H, W) and (C, H, W) arrays, band by band and
# ignoring NaN:
#
#   linear_stretch       map [low, high] linearly onto [0, 1]
#   percentile_stretch   linear stretch between two band percentiles
#   gamma_correction     power-law adjustment of values in [0, 1]
#   histogram_equalize   map values through the empirical CDF
#   histogram_match      match the distribution of a reference image,
#                        e.g. for relative radiometric normalisation of
#                        multi-date images or before pansharpening
#
# Method
# ------
# Histogram matching maps every source value to the reference quantile at
# the same empirical probability, F_ref^-1(F_src(x)), with linear
# interpolation between quantiles (Gonzalez and Woods, 2018, sec. 3.3).
#
# References
# ----------
#   Gonzalez, R. C., Woods, R. E. (2018). Digital Image Processing, 4th ed.
#     Pearson. Chapter 3, intensity transformations.
#   Richards, J. A. (2022). Remote Sensing Digital Image Analysis, 6th ed.
#     Springer. Chapter 4, radiometric enhancement.
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

# Float conversion and band layout helpers.
from unbihexium.preprocessing.transforms import as_float, band_first


# Broadcast a scalar or per-band sequence to shape (C, 1, 1).
def _per_band(value: float | Sequence[float] | NDArray[Any], bands: int) -> NDArray[np.float64]:
    # Values as a flat float array.
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    # A scalar applies to every band.
    if v.size == 1:
        # Repeat for every band.
        v = np.repeat(v, bands)
    # Otherwise one value per band is required.
    if v.size != bands:
        # Explain the requirement.
        raise ValueError(f"expected 1 or {bands} values, got {v.size}")
    # Shape for broadcasting over rows and columns.
    return v.reshape(-1, 1, 1)


# Linear stretch of [low, high] onto [0, 1], per band.
def linear_stretch(
    image: NDArray[Any],  # (H, W) or (C, H, W) array.
    low: float | Sequence[float] | NDArray[Any],  # Value mapped to 0, scalar or per band.
    high: float | Sequence[float] | NDArray[Any],  # Value mapped to 1, scalar or per band.
    clip: bool = True,  # Clip the result to [0, 1].
    nodata: float | None = None,  # Value treated as missing.
) -> NDArray[np.float64]:  # Stretched array with NaN for missing values.
    # Float copy in band-first layout.
    x = band_first(as_float(image, nodata))
    # Lower bounds per band.
    lo = _per_band(low, x.shape[0])
    # Upper bounds per band.
    hi = _per_band(high, x.shape[0])
    # The ranges must not be empty.
    if np.any(hi <= lo):
        # Explain the requirement.
        raise ValueError("high must be greater than low")
    # Scale every band.
    out = (x - lo) / (hi - lo)
    # Clip to the unit interval, keeping NaN.
    if clip:
        # np.clip propagates NaN.
        out = np.clip(out, 0.0, 1.0)
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# Percentile bounds of every band over finite values.
def percentile_bounds(
    image: NDArray[Any],  # (H, W) or (C, H, W) array.
    low: float = 2.0,  # Lower percentile.
    high: float = 98.0,  # Upper percentile.
    nodata: float | None = None,  # Value treated as missing.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Lower and upper values per band.
    # Percentiles must be ordered within [0, 100].
    if not 0.0 <= low < high <= 100.0:
        # Explain the requirement.
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    # Float copy in band-first layout.
    x = band_first(as_float(image, nodata))
    # Every band needs finite values.
    if not np.isfinite(x).any(axis=(1, 2)).all():
        # Explain the requirement.
        raise ValueError("every band needs at least one finite value")
    # Percentiles of every band.
    bounds = np.nanpercentile(x, [low, high], axis=(1, 2))
    # Split into lower and upper values.
    return bounds[0], bounds[1]


# Linear stretch between two percentiles of every band.
def percentile_stretch(
    image: NDArray[Any],  # (H, W) or (C, H, W) array.
    low: float = 2.0,  # Lower percentile.
    high: float = 98.0,  # Upper percentile.
    nodata: float | None = None,  # Value treated as missing.
) -> NDArray[np.float64]:  # Values in [0, 1], NaN for missing values.
    # Percentile values of every band.
    lo, hi = percentile_bounds(image, low, high, nodata)
    # Constant bands get a unit range so that they map to zero.
    hi = np.where(hi > lo, hi, lo + 1.0)
    # Apply the linear stretch.
    return linear_stretch(image, lo, hi, clip=True, nodata=nodata)


# Power-law adjustment out = in ** (1 / gamma) of values in [0, 1].
def gamma_correction(image: NDArray[Any], gamma: float = 1.0) -> NDArray[np.float64]:
    # Gamma must be positive.
    if gamma <= 0:
        # Explain the requirement.
        raise ValueError("gamma must be positive")
    # Values in the unit interval.
    x = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    # Values above 1 brighten mid-tones, below 1 darken them.
    return x ** (1.0 / gamma)


# Histogram equalisation of every band to [0, 1] through the empirical CDF.
def histogram_equalize(image: NDArray[Any], nodata: float | None = None) -> NDArray[np.float64]:
    # Float copy in band-first layout.
    x = band_first(as_float(image, nodata))
    # Output with NaN where the input is missing.
    out = np.full_like(x, np.nan)
    # Equalise every band.
    for b in range(x.shape[0]):
        # Finite pixels of the band.
        valid = np.isfinite(x[b])
        # Bands without data stay NaN.
        if not valid.any():
            # Next band.
            continue
        # Distinct values and their counts.
        values, counts = np.unique(x[b][valid], return_counts=True)
        # Empirical CDF at each distinct value.
        cdf = np.cumsum(counts) / counts.sum()
        # Look up the CDF of every pixel.
        out[b][valid] = np.interp(x[b][valid], values, cdf)
    # Restore the input rank.
    return out[0] if np.ndim(image) == 2 else out


# Match the histogram of every source band to the matching reference band.
def histogram_match(
    source: NDArray[Any],  # Image to adjust, (H, W) or (C, H, W).
    reference: NDArray[Any],  # Image with the target distribution, same bands.
    nodata: float | None = None,  # Value treated as missing in both images.
) -> NDArray[np.float64]:  # Source values mapped onto the reference distribution.
    # Source in band-first layout.
    src = band_first(as_float(source, nodata))
    # Reference in band-first layout.
    ref = band_first(as_float(reference, nodata))
    # The number of bands must agree; spatial sizes may differ.
    if src.shape[0] != ref.shape[0]:
        # Explain the requirement.
        raise ValueError("source and reference must have the same number of bands")
    # Output with NaN where the source is missing.
    out = np.full_like(src, np.nan)
    # Match every band.
    for b in range(src.shape[0]):
        # Finite source pixels.
        valid = np.isfinite(src[b])
        # Finite reference values.
        ref_values = ref[b][np.isfinite(ref[b])]
        # Both bands need data.
        if not valid.any() or ref_values.size == 0:
            # Explain the requirement.
            raise ValueError(f"band {b} has no finite values in source or reference")
        # Distinct source values, their positions and counts.
        values, inverse, counts = np.unique(
            src[b][valid],  # Finite source values.
            return_inverse=True,  # Position of every pixel in the distinct values.
            return_counts=True,  # Frequency of every distinct value.
        )  # End of the source histogram.
        # Empirical source quantiles at the centre of each value's mass.
        src_q = (np.cumsum(counts) - 0.5 * counts) / counts.sum()
        # Sorted reference values.
        ref_sorted = np.sort(ref_values)
        # Empirical reference quantiles with the same plotting positions.
        ref_q = (np.arange(ref_sorted.size) + 0.5) / ref_sorted.size
        # Reference value at each source quantile.
        mapped = np.interp(src_q, ref_q, ref_sorted)
        # Write the mapped value of every pixel.
        out[b][valid] = mapped[inverse.reshape(-1)]
    # Restore the input rank.
    return out[0] if np.ndim(source) == 2 else out


# =============================================================================
# End of module src/unbihexium/preprocessing/enhancement.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
