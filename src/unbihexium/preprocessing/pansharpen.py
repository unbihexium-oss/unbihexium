# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/pansharpen.py
# Title       : Component substitution pansharpening
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Fusion of a panchromatic band with multispectral bands that have been
# resampled to the panchromatic grid (use upsample_to_pan first):
#
#   upsample_to_pan   interpolate (C, h, w) bands onto the (H, W) pan grid
#   brovey            ratio (Brovey) transform
#   ihs               fast generalised intensity-hue-saturation fusion
#   gram_schmidt      Gram-Schmidt adaptive (GSA) component substitution
#
# Method
# ------
# All three methods are component substitution: an intensity
# I = sum_k w_k MS_k (+ b) is computed and the spatial detail P - I is
# injected into every band, F_k = MS_k + g_k (P - I).
#
#   Brovey  multiplicative form F_k = MS_k P / I (g_k = MS_k / I), with I
#           the weighted mean of the bands.
#   IHS     g_k = 1 with I the weighted mean of the bands (Tu et al., 2001).
#   GSA     weights w_k and b from a least-squares regression of the pan on
#           the multispectral bands, and g_k = cov(MS_k, I) / var(I)
#           (Aiazzi et al., 2007), which reproduces Gram-Schmidt spectral
#           sharpening (Laben and Brower, 2000) with an adaptive intensity.
#
# Before injection the pan is matched to the mean and standard deviation of
# the intensity (histogram matching in the first two moments). Pixels with
# NaN in any input are NaN in the output.
#
# References
# ----------
#   Gillespie, A. R., Kahle, A. B., Walker, R. E. (1987). Color enhancement
#     of highly correlated images. II. Channel ratio and "chromaticity"
#     transformation techniques. Remote Sensing of Environment 22(3), 343-365.
#   Tu, T.-M., Su, S.-C., Shyu, H.-C., Huang, P. S. (2001). A new look at
#     IHS-like image fusion methods. Information Fusion 2(3), 177-186.
#   Laben, C. A., Brower, B. V. (2000). Process for enhancing the spatial
#     resolution of multispectral imagery using pan-sharpening. US Patent
#     6,011,875.
#   Aiazzi, B., Baronti, S., Selva, M. (2007). Improving component
#     substitution pansharpening through multivariate regression of MS+Pan
#     data. IEEE Transactions on Geoscience and Remote Sensing 45(10),
#     3230-3239.
#   Vivone, G., et al. (2015). A critical comparison among pansharpening
#     algorithms. IEEE Transactions on Geoscience and Remote Sensing 53(5),
#     2565-2586.
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

# Spline interpolation for upsampling.
from scipy.ndimage import zoom


# Validate the inputs and return them as float arrays.
def _prepare(
    pan: NDArray[Any],  # (H, W) panchromatic band.
    ms: NDArray[Any],  # (C, H, W) multispectral bands on the pan grid.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Float pan and bands.
    # Pan as float.
    p = np.asarray(pan, dtype=np.float64)
    # Bands as float.
    m = np.asarray(ms, dtype=np.float64)
    # The pan must be a single band.
    if p.ndim != 2:
        # Explain the requirement.
        raise ValueError(f"pan must be (H, W), got shape {p.shape}")
    # The bands must share the pan grid.
    if m.ndim != 3 or m.shape[1:] != p.shape:
        # Explain the requirement.
        raise ValueError(f"ms must be (C, {p.shape[0]}, {p.shape[1]}), got shape {m.shape}")
    # Return the validated arrays.
    return p, m


# Normalised band weights, equal by default.
def _weights(weights: Sequence[float] | None, bands: int) -> NDArray[np.float64]:
    # Equal weights by default.
    if weights is None:
        # Arithmetic mean of the bands.
        return np.full(bands, 1.0 / bands)
    # Weights as an array.
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    # One weight per band, with a positive sum.
    if w.size != bands or w.sum() <= 0:
        # Explain the requirement.
        raise ValueError(f"expected {bands} weights with a positive sum")
    # Normalise to unit sum.
    return w / w.sum()


# Match the pan to the mean and standard deviation of the intensity.
def _match_moments(pan: NDArray[Any], intensity: NDArray[Any]) -> NDArray[np.float64]:
    # Pixels valid in both images.
    valid = np.isfinite(pan) & np.isfinite(intensity)
    # Statistics need data.
    if not valid.any():
        # Explain the requirement.
        raise ValueError("pan and multispectral bands have no common valid pixels")
    # Moments of the pan.
    mp, sp = pan[valid].mean(), pan[valid].std()
    # Moments of the intensity.
    mi, si = intensity[valid].mean(), intensity[valid].std()
    # A constant pan has no detail to inject.
    if sp == 0:
        # Replace it by the intensity mean.
        return np.where(np.isfinite(pan), mi, np.nan)
    # Linear moment matching.
    return (pan - mp) * (si / sp) + mi


# Interpolate multispectral bands onto the grid of the panchromatic band.
def upsample_to_pan(
    ms: NDArray[Any],  # (C, h, w) multispectral bands.
    shape: tuple[int, int],  # (H, W) of the pan grid.
    order: int = 3,  # Spline order: 0 nearest, 1 bilinear, 3 cubic.
) -> NDArray[np.float64]:  # (C, H, W) bands on the pan grid.
    # Bands as float.
    m = np.asarray(ms, dtype=np.float64)
    # A band axis is required.
    if m.ndim != 3:
        # Explain the requirement.
        raise ValueError(f"ms must be (C, h, w), got shape {m.shape}")
    # Zoom factors of rows and columns; bands are kept.
    factors = (1.0, shape[0] / m.shape[1], shape[1] / m.shape[2])
    # Pixel-area aligned interpolation keeps the footprint of the image.
    out = zoom(m, factors, order=order, mode="grid-mirror", grid_mode=True)
    # Return the upsampled bands.
    return out


# Brovey (ratio) transform.
def brovey(
    pan: NDArray[Any],  # (H, W) panchromatic band.
    ms: NDArray[Any],  # (C, H, W) bands on the pan grid.
    weights: Sequence[float] | None = None,  # Band weights of the intensity.
    match: bool = True,  # Match the pan to the intensity first.
) -> NDArray[np.float64]:  # (C, H, W) sharpened bands.
    # Validated float inputs.
    p, m = _prepare(pan, ms)
    # Intensity as the weighted mean of the bands.
    intensity = np.tensordot(_weights(weights, m.shape[0]), m, axes=1)
    # Detail image, matched to the intensity when requested.
    detail = _match_moments(p, intensity) if match else p
    # Ratio of pan to intensity, NaN where the intensity is zero.
    ratio = np.divide(detail, intensity, out=np.full_like(p, np.nan), where=intensity != 0)
    # Scale every band.
    return m * ratio[None]


# Fast generalised IHS fusion.
def ihs(
    pan: NDArray[Any],  # (H, W) panchromatic band.
    ms: NDArray[Any],  # (C, H, W) bands on the pan grid.
    weights: Sequence[float] | None = None,  # Band weights of the intensity.
    match: bool = True,  # Match the pan to the intensity first.
) -> NDArray[np.float64]:  # (C, H, W) sharpened bands.
    # Validated float inputs.
    p, m = _prepare(pan, ms)
    # Intensity as the weighted mean of the bands.
    intensity = np.tensordot(_weights(weights, m.shape[0]), m, axes=1)
    # Detail image, matched to the intensity when requested.
    detail = _match_moments(p, intensity) if match else p
    # Add the same detail to every band.
    return m + (detail - intensity)[None]


# Gram-Schmidt adaptive (GSA) component substitution.
def gram_schmidt(
    pan: NDArray[Any],  # (H, W) panchromatic band.
    ms: NDArray[Any],  # (C, H, W) bands on the pan grid.
    weights: Sequence[float] | None = None,  # Fixed intensity weights; None regresses them.
) -> NDArray[np.float64]:  # (C, H, W) sharpened bands.
    # Validated float inputs.
    p, m = _prepare(pan, ms)
    # Number of bands.
    c = m.shape[0]
    # Pixels valid in the pan and every band.
    valid = np.isfinite(p) & np.isfinite(m).all(axis=0)
    # At least one pixel more than unknowns is needed.
    if valid.sum() <= c + 1:
        # Explain the requirement.
        raise ValueError("too few valid pixels to estimate the intensity")
    # Valid band values as (N, C).
    x = m[:, valid].T
    # Estimate the intensity weights by least squares when not given.
    if weights is None:
        # Design matrix with an intercept column.
        design = np.column_stack([x, np.ones(x.shape[0])])
        # Regression of the pan on the bands.
        coef = np.linalg.lstsq(design, p[valid], rcond=None)[0]
        # Band weights and intercept.
        w, b = coef[:c], coef[c]
    # Otherwise use the given weights without intercept.
    else:
        # Normalised weights.
        w, b = _weights(weights, c), 0.0
    # Intensity image.
    intensity = np.tensordot(w, m, axes=1) + b
    # Detail image matched to the intensity.
    detail = _match_moments(p, intensity) - intensity
    # Valid intensity values.
    iv = intensity[valid]
    # Variance of the intensity.
    var_i = iv.var()
    # A constant intensity carries no information for the gains.
    if var_i == 0:
        # Explain the problem.
        raise ValueError("the intensity is constant; gains are undefined")
    # Injection gains g_k = cov(MS_k, I) / var(I).
    gains = ((x - x.mean(axis=0)) * (iv - iv.mean())[:, None]).mean(axis=0) / var_i
    # Inject the detail into every band.
    return m + gains[:, None, None] * detail[None]


# =============================================================================
# End of module src/unbihexium/preprocessing/pansharpen.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
