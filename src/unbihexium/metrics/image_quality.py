# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/metrics/image_quality.py
# Title       : Image quality and spectral fidelity of fused and enhanced images
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Full-reference quality measures for super-resolution, enhancement and
# pansharpening products:
#
#   psnr                    peak signal-to-noise ratio in dB
#   ssim                    structural similarity (Wang et al., 2004)
#   spectral_angle          per-pixel spectral angle in degrees
#   sam                     mean spectral angle mapper in degrees
#   ergas                   relative dimensionless global error in synthesis
#   q_index                 universal image quality index Q, mean over bands
#   calculate_sam, calculate_ergas, calculate_qindex
#                           aliases of sam, ergas and q_index
#
# PSNR and SSIM are those of unbihexium.ai.evaluation, so that training and
# product validation report identical numbers. Multi-band images are
# band-first (C, H, W); pixels with NaN are ignored where stated.
#
# Method
# ------
# SAM: arccos(<x, y> / (|x| |y|)) of the spectra x, y of every pixel,
# averaged over pixels (Yuhas et al., 1992).
# ERGAS = 100 (h / l) sqrt(mean_k (RMSE_k / mu_k)^2), with h / l the ratio
# of the pixel sizes of the high and low resolution images (for example 1/4)
# and mu_k the mean of reference band k (Wald, 2002). The ratio argument is
# l / h, so ERGAS = 100 / ratio sqrt(...).
# Q = 4 s_xy m_x m_y / ((s_x^2 + s_y^2)(m_x^2 + m_y^2)) over sliding
# windows, averaged over windows (Wang and Bovik, 2002); a window size of
# None computes Q over the whole band.
#
# References
# ----------
#   Wang, Z., Bovik, A. C. (2002). A universal image quality index. IEEE
#     Signal Processing Letters 9(3), 81-84.
#   Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. (2004). Image
#     quality assessment: from error visibility to structural similarity.
#     IEEE Transactions on Image Processing 13(4), 600-612.
#   Yuhas, R. H., Goetz, A. F. H., Boardman, J. W. (1992). Discrimination
#     among semi-arid landscape endmembers using the spectral angle mapper
#     (SAM) algorithm. Summaries of the Third Annual JPL Airborne Geoscience
#     Workshop, JPL Publication 92-14, 147-149.
#   Wald, L. (2002). Data Fusion: Definitions and Architectures. Fusion of
#     Images of Different Spatial Resolutions. Presses de l'Ecole, Ecole des
#     Mines de Paris.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Implementations shared with model evaluation.
from unbihexium.ai import evaluation as _evaluation


# Peak signal-to-noise ratio in decibels, ignoring NaN.
def psnr(pred: NDArray[Any], target: NDArray[Any], max_val: float = 1.0) -> float:
    # The value range must be positive.
    if max_val <= 0:
        # Explain the requirement.
        raise ValueError("max_val must be positive")
    # 10 log10(max^2 / MSE); infinite for identical images.
    return _evaluation.psnr(pred, target, data_range=max_val)


# Mean structural similarity of (H, W) or (C, H, W) images.
def ssim(
    pred: NDArray[Any],  # Estimated image.
    target: NDArray[Any],  # Reference image.
    data_range: float = 1.0,  # Value range of the images.
    sigma: float = 1.5,  # Standard deviation of the Gaussian window.
) -> float:  # Mean SSIM over pixels and bands.
    # Shapes must agree.
    if np.shape(pred) != np.shape(target):
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {np.shape(pred)} and {np.shape(target)}")
    # Gaussian-window SSIM of Wang et al. (2004).
    return _evaluation.ssim(pred, target, data_range=data_range, sigma=sigma)


# Validate a reference and an estimate as float (C, H, W) arrays.
def _bands(reference: NDArray[Any], estimate: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    # Reference as float.
    r = np.asarray(reference, dtype=np.float64)
    # Estimate as float.
    e = np.asarray(estimate, dtype=np.float64)
    # Shapes must agree.
    if r.shape != e.shape:
        # Explain the requirement.
        raise ValueError(f"shape mismatch: {r.shape} and {e.shape}")
    # Single bands get a band axis.
    if r.ndim == 2:
        # Add the axis.
        return r[None], e[None]
    # Only (C, H, W) remains valid.
    if r.ndim != 3:
        # Explain the requirement.
        raise ValueError(f"expected (H, W) or (C, H, W) arrays, got shape {r.shape}")
    # Return both.
    return r, e


# Spectral angle of every pixel in degrees, NaN where undefined.
def spectral_angle(reference: NDArray[Any], estimate: NDArray[Any]) -> NDArray[np.float64]:
    # Validated arrays.
    r, e = _bands(reference, estimate)
    # Dot product of the spectra.
    dot = np.sum(r * e, axis=0)
    # Product of the norms.
    norms = np.sqrt(np.sum(r * r, axis=0) * np.sum(e * e, axis=0))
    # Cosine of the angle, NaN for zero spectra.
    cos = np.divide(dot, norms, out=np.full_like(dot, np.nan), where=norms > 0)
    # Angle in degrees; clip against rounding outside [-1, 1].
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# Mean spectral angle in degrees over pixels where it is defined.
def sam(reference: NDArray[Any], estimate: NDArray[Any]) -> float:
    # Angles of every pixel.
    angles = spectral_angle(reference, estimate)
    # Defined angles.
    valid = np.isfinite(angles)
    # At least one pixel is needed.
    if not valid.any():
        # Explain the requirement.
        raise ValueError("no pixel with a defined spectral angle")
    # Mean angle.
    return float(angles[valid].mean())


# Relative dimensionless global error in synthesis (ERGAS).
def ergas(reference: NDArray[Any], estimate: NDArray[Any], ratio: float = 4.0) -> float:
    # The resolution ratio must be positive.
    if ratio <= 0:
        # Explain the requirement.
        raise ValueError("ratio must be positive")
    # Validated arrays.
    r, e = _bands(reference, estimate)
    # Pixels finite in both images and every band.
    valid = np.isfinite(r).all(axis=0) & np.isfinite(e).all(axis=0)
    # At least one pixel is needed.
    if not valid.any():
        # Explain the requirement.
        raise ValueError("no finite pixels to evaluate")
    # RMSE of every band.
    band_rmse = np.sqrt(np.mean((r[:, valid] - e[:, valid]) ** 2, axis=1))
    # Mean of every reference band.
    band_mean = np.mean(r[:, valid], axis=1)
    # Band means must not be zero.
    if np.any(band_mean == 0):
        # Explain the requirement.
        raise ValueError("ERGAS is undefined for reference bands with zero mean")
    # 100 / ratio times the root mean of the squared relative errors.
    return float(100.0 / ratio * np.sqrt(np.mean((band_rmse / band_mean) ** 2)))


# Window sums of an (H, W) array for every fully contained window.
def _window_sums(x: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    # Integral image with a leading zero row and column.
    s = np.pad(np.cumsum(np.cumsum(x, axis=0), axis=1), ((1, 0), (1, 0)))
    # Inclusion-exclusion of the four corners.
    return s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]


# Universal image quality index of one band.
def _q_band(x: NDArray[np.float64], y: NDArray[np.float64], size: int | None) -> float:
    # Whole band as a single window.
    if size is None:
        # Means.
        mx, my = x.mean(), y.mean()
        # Variances.
        vx, vy = x.var(), y.var()
        # Covariance.
        cxy = float(np.mean((x - mx) * (y - my)))
        # Arrays of one element so that the common code below applies.
        mx, my, vx, vy, cxy = (np.array([v], dtype=np.float64) for v in (mx, my, vx, vy, cxy))
    # Sliding windows.
    else:
        # Pixels per window.
        n = size * size
        # Window means.
        mx, my = _window_sums(x, size) / n, _window_sums(y, size) / n
        # Window variances.
        vx = _window_sums(x * x, size) / n - mx**2
        # Variance of the estimate.
        vy = _window_sums(y * y, size) / n - my**2
        # Window covariance.
        cxy = _window_sums(x * y, size) / n - mx * my
    # Contrast denominator.
    d1 = vx + vy
    # Luminance denominator.
    d2 = mx**2 + my**2
    # Q = 4 s_xy m_x m_y / (d1 d2); both constant and equal windows give 1.
    q = np.ones_like(d1)
    # Both denominators positive: the full formula.
    both = (d1 > 0) & (d2 > 0)
    # Apply it.
    q[both] = 4 * cxy[both] * mx[both] * my[both] / (d1[both] * d2[both])
    # Constant windows with different means: luminance term only.
    lum = (d1 <= 0) & (d2 > 0)
    # Apply it.
    q[lum] = 2 * mx[lum] * my[lum] / d2[lum]
    # Zero means with variation: correlation and contrast term only.
    con = (d1 > 0) & (d2 <= 0)
    # Apply it.
    q[con] = 2 * cxy[con] / d1[con]
    # Mean over windows.
    return float(q.mean())


# Universal image quality index Q, averaged over bands.
def q_index(
    reference: NDArray[Any],  # Reference image.
    estimate: NDArray[Any],  # Estimated image.
    block_size: int | None = 8,  # Sliding window size, None for global.
) -> float:  # Q in [-1, 1].
    # Validated arrays.
    r, e = _bands(reference, estimate)
    # NaN would spread through the window sums.
    if not (np.isfinite(r).all() and np.isfinite(e).all()):
        # Explain the requirement.
        raise ValueError("q_index needs finite images")
    # The window must fit the image.
    if block_size is not None and not 1 <= block_size <= min(r.shape[1:]):
        # Explain the requirement.
        raise ValueError("block_size must lie between 1 and the smaller image side")
    # Mean over bands.
    return float(np.mean([_q_band(rb, eb, block_size) for rb, eb in zip(r, e)]))


# Alias used in the documentation.
calculate_sam = sam

# Alias used in the documentation.
calculate_ergas = ergas

# Alias used in the documentation.
calculate_qindex = q_index


# =============================================================================
# End of module src/unbihexium/metrics/image_quality.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
