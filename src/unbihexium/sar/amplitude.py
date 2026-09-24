# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/sar/amplitude.py
# Title       : SAR radiometric calibration, multilooking and speckle filters
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Radiometric processing of detected (amplitude or intensity) synthetic
# aperture radar images:
#
#   radiometric_calibration   |DN|^2 minus thermal noise, divided by the
#                             squared calibration look-up table value
#   compute_beta0             radar brightness from amplitude
#   compute_sigma0            backscatter per unit ground area
#   compute_gamma0            backscatter per unit area perpendicular to the
#                             look direction
#   power_to_db, db_to_power  decibel conversion of intensities
#   amplitude_to_db           decibel conversion of amplitudes
#   multilook                 block averaging of intensities
#   equivalent_number_of_looks  ENL of a homogeneous area
#   speckle_filter            dispatcher for the filters below
#   lee_filter, kuan_filter, enhanced_lee_filter, frost_filter,
#   gamma_map_filter, refined_lee_filter, boxcar and median filters
#
# Radiometry
# ----------
# With calibrated amplitude A (so that A^2 = beta0 when K = 1) and the
# ellipsoid incidence angle theta:
#
#   beta0  = A^2 / K
#   sigma0 = beta0 * sin(theta)
#   gamma0 = sigma0 / cos(theta)
#
# Angles are given in degrees unless `degrees=False`.
#
# Speckle model
# -------------
# Fully developed speckle is multiplicative: I = R * v, where the noise v
# has unit mean and coefficient of variation Cu = 1 / sqrt(L) for an L-look
# intensity image, and Cu = sqrt(4 / pi - 1) / sqrt(L) (about 0.5227 for one
# look) for an amplitude image. Every adaptive filter compares the local
# coefficient of variation Ci = std / mean of a moving window with Cu:
# homogeneous areas (Ci close to Cu) are averaged, heterogeneous areas and
# edges (Ci much larger than Cu) keep the observed value. Local statistics
# ignore NaN pixels, and NaN pixels stay NaN.
#
# References
# ----------
# Miranda, N., Meadows, P. J. (2015). Radiometric calibration of S-1 Level-1
#   products from the S-1 IPF. ESA-EOPG-CSCOP-TN-0002, issue 1.0.
# Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of
#   local statistics. IEEE Transactions on Pattern Analysis and Machine
#   Intelligence, PAMI-2(2), 165-168.
# Lee, J.-S. (1981). Refined filtering of image noise using local
#   statistics. Computer Graphics and Image Processing, 15(4), 380-389.
# Frost, V. S., Stiles, J. A., Shanmugan, K. S., Holtzman, J. C. (1982). A
#   model for radar images and its application to adaptive digital
#   filtering of multiplicative noise. IEEE Transactions on Pattern Analysis
#   and Machine Intelligence, PAMI-4(2), 157-166.
# Kuan, D. T., Sawchuk, A. A., Strand, T. C., Chavel, P. (1985). Adaptive
#   noise smoothing filter for images with signal-dependent noise. IEEE
#   Transactions on Pattern Analysis and Machine Intelligence, PAMI-7(2),
#   165-177.
# Lopes, A., Touzi, R., Nezry, E. (1990). Adaptive speckle filters and scene
#   heterogeneity. IEEE Transactions on Geoscience and Remote Sensing,
#   28(6), 992-1000.
# Lopes, A., Nezry, E., Touzi, R., Laur, H. (1990). Maximum a posteriori
#   speckle filtering and first order texture models in SAR images. Proc.
#   IGARSS 1990, 2409-2412.
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

# Moving window filters.
from scipy import ndimage

# Coefficient of variation of one-look amplitude speckle, sqrt(4 / pi - 1).
AMPLITUDE_SPECKLE_CV = float(np.sqrt(4.0 / np.pi - 1.0))

# Names accepted by speckle_filter.
SPECKLE_FILTERS = (
    "boxcar",  # Moving average.
    "median",  # Moving median.
    "lee",  # Lee (1980).
    "kuan",  # Kuan et al. (1985).
    "enhanced_lee",  # Lopes, Touzi and Nezry (1990).
    "frost",  # Frost et al. (1982).
    "gamma_map",  # Lopes et al. (1990).
    "refined_lee",  # Lee (1981).
)  # End of the filter names.


# Convert an angle array or scalar to radians and check its range.
def _incidence_radians(angle: NDArray[Any] | float, degrees: bool) -> NDArray[np.float64]:
    # Work in float64.
    theta = np.asarray(angle, dtype=np.float64)
    # Convert from degrees when requested.
    if degrees:
        # Degrees to radians.
        theta = np.deg2rad(theta)
    # Finite angles must lie strictly between 0 and 90 degrees.
    finite = theta[np.isfinite(theta)]
    # Reject angles outside the physical range.
    if finite.size and (finite.min() <= 0.0 or finite.max() >= np.pi / 2):
        # Explain the expected range and unit.
        unit = "degrees" if degrees else "radians"
        # Report the problem.
        raise ValueError(f"incidence angle must lie in (0, 90) degrees; got values in {unit}")
    # Return the angle in radians.
    return theta


# Calibrate a detected or complex image with look-up tables (Sentinel-1 style).
def radiometric_calibration(
    dn: NDArray[Any],  # Digital numbers, complex (SLC) or real (GRD amplitude).
    lut: NDArray[Any] | float,  # Calibration look-up table A (sigmaNought, betaNought, ...).
    noise: NDArray[Any] | float | None = None,  # Thermal noise power to subtract.
    clip_negative: bool = True,  # Set negative results of noise removal to zero.
) -> NDArray[np.float64]:  # Calibrated intensity in linear units.
    # Detected power |DN|^2 of complex or real input.
    power = np.abs(np.asarray(dn)).astype(np.float64) ** 2
    # Calibration table as float64.
    a = np.asarray(lut, dtype=np.float64)
    # The table divides the power, so it must be positive.
    if np.any(a[np.isfinite(a)] <= 0):
        # Report the invalid table.
        raise ValueError("calibration look-up table values must be positive")
    # Subtract the thermal noise power when given.
    if noise is not None:
        # Noise power in the same units as |DN|^2.
        power = power - np.asarray(noise, dtype=np.float64)
    # Apply the calibration constant.
    value = power / a**2
    # Noise removal may produce small negative powers.
    if clip_negative:
        # Keep NaN, clip negative values.
        value = np.where(value < 0, 0.0, value)
    # Return the calibrated intensity.
    return value


# Scale amplitudes (or the modulus of complex samples) by a constant.
def calibrate_amplitude(
    data: NDArray[Any],  # Complex samples or amplitudes.
    calibration_factor: float = 1.0,  # Multiplicative calibration constant.
) -> NDArray[np.float64]:  # Calibrated amplitude.
    # Amplitude as float64; the modulus of complex samples.
    amplitude = np.abs(np.asarray(data)).astype(np.float64)
    # Apply the constant.
    return amplitude * float(calibration_factor)


# Radar brightness beta0 from calibrated amplitude.
def compute_beta0(
    amplitude: NDArray[Any],  # Calibrated amplitude.
    calibration_lut: NDArray[Any] | float | None = None,  # Constant K, per pixel or scalar.
) -> NDArray[np.float64]:  # beta0 in linear units.
    # Intensity from amplitude.
    beta0 = np.abs(np.asarray(amplitude)).astype(np.float64) ** 2
    # Divide by the calibration constant when given.
    if calibration_lut is not None:
        # Constant K.
        k = np.asarray(calibration_lut, dtype=np.float64)
        # K must be positive.
        if np.any(k[np.isfinite(k)] <= 0):
            # Report the invalid table.
            raise ValueError("calibration_lut values must be positive")
        # Apply K.
        beta0 = beta0 / k
    # Return beta0.
    return beta0


# Backscatter coefficient sigma0 = A^2 / K * sin(theta).
def compute_sigma0(
    amplitude: NDArray[Any],  # Calibrated amplitude.
    incidence_angle: NDArray[Any] | float,  # Ellipsoid incidence angle.
    calibration_lut: NDArray[Any] | float | None = None,  # Constant K, per pixel or scalar.
    degrees: bool = True,  # Whether the angle is given in degrees.
) -> NDArray[np.float64]:  # sigma0 in linear units, never negative.
    # Incidence angle in radians.
    theta = _incidence_radians(incidence_angle, degrees)
    # beta0 times the sine of the incidence angle.
    return compute_beta0(amplitude, calibration_lut) * np.sin(theta)


# Backscatter coefficient gamma0 = sigma0 / cos(theta).
def compute_gamma0(
    sigma0: NDArray[Any],  # sigma0 in linear units.
    incidence_angle: NDArray[Any] | float,  # Ellipsoid incidence angle.
    degrees: bool = True,  # Whether the angle is given in degrees.
) -> NDArray[np.float64]:  # gamma0 in linear units.
    # Incidence angle in radians.
    theta = _incidence_radians(incidence_angle, degrees)
    # Normalise by the cosine of the incidence angle.
    return np.asarray(sigma0, dtype=np.float64) / np.cos(theta)


# Intensity to decibels, 10 log10(p); non-positive values become NaN or the floor.
def power_to_db(
    power: NDArray[Any],  # Linear intensity.
    floor: float | None = None,  # Lowest value returned; None keeps NaN for p <= 0.
) -> NDArray[np.float64]:  # Values in dB.
    # Intensity as float64.
    p = np.asarray(power, dtype=np.float64)
    # Take logarithms of positive values only.
    with np.errstate(divide="ignore", invalid="ignore"):
        # NaN for zero or negative powers.
        db = np.where(p > 0, 10.0 * np.log10(np.where(p > 0, p, 1.0)), np.nan)
    # Apply the floor when requested.
    if floor is not None:
        # Non-positive powers and very small values take the floor; NaN input stays NaN.
        db = np.where(np.isnan(p), np.nan, np.where(np.isnan(db), floor, np.maximum(db, floor)))
    # Return decibels.
    return db


# Decibels to linear intensity, 10^(dB / 10).
def db_to_power(db: NDArray[Any]) -> NDArray[np.float64]:
    # Inverse of power_to_db.
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


# Amplitude to decibels, 20 log10(A), with a floor for zero amplitudes.
def amplitude_to_db(
    amplitude: NDArray[Any],  # Linear amplitude.
    floor: float = -40.0,  # Lowest value returned.
) -> NDArray[np.float64]:  # Values in dB.
    # Amplitude squared is the intensity.
    power = np.abs(np.asarray(amplitude, dtype=np.float64)) ** 2
    # Reuse the intensity conversion with the floor.
    return power_to_db(power, floor=floor)


# Average intensities over non-overlapping blocks of (rows, cols) pixels.
def multilook(
    intensity: NDArray[Any],  # 2-D intensity image, or complex samples.
    looks: tuple[int, int] = (1, 1),  # Looks in azimuth (rows) and range (cols).
) -> NDArray[np.float64]:  # Image of shape (H // rows, W // cols).
    # Number of looks per axis.
    la, lr = (int(v) for v in looks)
    # Both factors must be positive.
    if la < 1 or lr < 1:
        # Report the invalid factors.
        raise ValueError(f"looks must be positive integers, got {looks}")
    # Complex samples are detected first.
    data = np.asarray(intensity)
    # Detection of complex samples.
    if np.iscomplexobj(data):
        # Intensity |s|^2.
        data = np.abs(data) ** 2
    # Work in float64.
    data = data.astype(np.float64)
    # Only 2-D images are supported.
    if data.ndim != 2:
        # Report the wrong shape.
        raise ValueError(f"multilook expects a 2-D image, got shape {data.shape}")
    # Output size; incomplete blocks at the far edges are dropped.
    rows, cols = data.shape[0] // la, data.shape[1] // lr
    # The image must hold at least one block.
    if rows == 0 or cols == 0:
        # Report the too small image.
        raise ValueError(f"image of shape {data.shape} is smaller than one {looks} block")
    # Group the pixels of each block on separate axes.
    blocks = data[: rows * la, : cols * lr].reshape(rows, la, cols, lr)
    # Mean of the valid pixels of each block, NaN for empty blocks.
    valid = np.isfinite(blocks)
    # Number of valid pixels per block.
    count = valid.sum(axis=(1, 3))
    # Sum of the valid pixels per block.
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    # Divide where the block has data.
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


# Equivalent number of looks, mean^2 / variance of a homogeneous intensity area.
def equivalent_number_of_looks(intensity: NDArray[Any]) -> float:
    # Valid samples of the area.
    values = np.asarray(intensity, dtype=np.float64)
    # Ignore NaN pixels.
    values = values[np.isfinite(values)]
    # At least two samples are needed for a variance.
    if values.size < 2:
        # Report the too small area.
        raise ValueError("at least two valid pixels are needed to estimate the ENL")
    # Sample variance.
    var = float(values.var(ddof=1))
    # A constant area has infinitely many looks.
    if var == 0.0:
        # Infinite ENL.
        return float("inf")
    # mean^2 / variance.
    return float(values.mean() ** 2 / var)


# Coefficient of variation of the speckle for the given looks and data type.
def speckle_cv(looks: float = 1.0, amplitude: bool = False) -> float:
    # The number of looks must be positive.
    if looks <= 0:
        # Report the invalid value.
        raise ValueError(f"looks must be positive, got {looks}")
    # Amplitude speckle is less variable than intensity speckle.
    base = AMPLITUDE_SPECKLE_CV if amplitude else 1.0
    # Speckle variability falls with the square root of the looks.
    return base / float(np.sqrt(looks))


# Check the window size of a moving-window filter.
def _check_window(window_size: int) -> int:
    # Integer size.
    size = int(window_size)
    # The window must be odd and at least 3 pixels wide.
    if size < 3 or size % 2 == 0:
        # Report the invalid size.
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    # Return the size.
    return size


# NaN-aware correlation of an image with a non-negative kernel, normalised by the kernel mass.
def _masked_mean(
    data: NDArray[np.float64],  # Image with NaN for invalid pixels.
    kernel: NDArray[np.float64],  # Window weights.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Local mean and local mean of squares.
    # Valid pixel mask.
    valid = np.isfinite(data)
    # Zero-filled data.
    filled = np.where(valid, data, 0.0)
    # Weighted count of valid pixels in the window.
    count = ndimage.correlate(valid.astype(np.float64), kernel, mode="reflect")
    # Weighted sum of values.
    s1 = ndimage.correlate(filled, kernel, mode="reflect")
    # Weighted sum of squares.
    s2 = ndimage.correlate(filled * filled, kernel, mode="reflect")
    # Windows without valid pixels give NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Local mean.
        mean = np.where(count > 1e-12, s1 / np.where(count > 1e-12, count, 1.0), np.nan)
        # Local mean of squares.
        mean_sq = np.where(count > 1e-12, s2 / np.where(count > 1e-12, count, 1.0), np.nan)
    # Return the moments.
    return mean, mean_sq


# Local mean and variance in a square window, ignoring NaN pixels.
def local_statistics(
    data: NDArray[Any],  # 2-D image.
    window_size: int,  # Odd window size.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Local mean and variance.
    # Square window of equal weights.
    kernel = np.ones((window_size, window_size), dtype=np.float64)
    # Moments of the window.
    mean, mean_sq = _masked_mean(np.asarray(data, dtype=np.float64), kernel)
    # Variance, clipped at zero against round-off.
    return mean, np.maximum(mean_sq - mean * mean, 0.0)


# Squared local coefficient of variation var / mean^2, NaN where the mean is zero.
def _ci2(mean: NDArray[np.float64], var: NDArray[np.float64]) -> NDArray[np.float64]:
    # Avoid dividing by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Ratio for non-zero means, zero variation for an all-zero window.
        return np.where(mean != 0, var / np.where(mean != 0, mean * mean, 1.0), 0.0)


# Prepare an image for filtering: float64 2-D array.
def _as_image(data: NDArray[Any]) -> NDArray[np.float64]:
    # Float64 copy of the input.
    image = np.asarray(data, dtype=np.float64)
    # Filters work on single images.
    if image.ndim != 2:
        # Report the wrong shape.
        raise ValueError(f"speckle filters expect a 2-D image, got shape {image.shape}")
    # Return the image.
    return image


# Lee (1980) minimum mean square error filter for multiplicative noise.
def lee_filter(
    data: NDArray[Any],  # Intensity (or amplitude with amplitude=True).
    window_size: int = 5,  # Odd window size.
    looks: float = 1.0,  # Number of looks of the image.
    amplitude: bool = False,  # Whether the image holds amplitudes.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image and window.
    image, size = _as_image(data), _check_window(window_size)
    # Local statistics.
    mean, var = local_statistics(image, size)
    # Speckle variation.
    cu2 = speckle_cv(looks, amplitude) ** 2
    # Local variation.
    ci2 = _ci2(mean, var)
    # Weight 1 - Cu^2 / Ci^2, limited to [0, 1].
    with np.errstate(divide="ignore", invalid="ignore"):
        # Zero variation windows are fully averaged.
        w = np.where(ci2 > 0, 1.0 - cu2 / np.where(ci2 > 0, ci2, 1.0), 0.0)
    # Clip the weight.
    w = np.clip(w, 0.0, 1.0)
    # Blend the local mean and the observation.
    return mean + w * (image - mean)


# Kuan et al. (1985) linear MMSE filter.
def kuan_filter(
    data: NDArray[Any],  # Intensity (or amplitude with amplitude=True).
    window_size: int = 5,  # Odd window size.
    looks: float = 1.0,  # Number of looks of the image.
    amplitude: bool = False,  # Whether the image holds amplitudes.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image and window.
    image, size = _as_image(data), _check_window(window_size)
    # Local statistics.
    mean, var = local_statistics(image, size)
    # Speckle variation.
    cu2 = speckle_cv(looks, amplitude) ** 2
    # Local variation.
    ci2 = _ci2(mean, var)
    # Weight (1 - Cu^2 / Ci^2) / (1 + Cu^2), limited to [0, 1].
    with np.errstate(divide="ignore", invalid="ignore"):
        # Zero variation windows are fully averaged.
        w = np.where(ci2 > 0, (1.0 - cu2 / np.where(ci2 > 0, ci2, 1.0)) / (1.0 + cu2), 0.0)
    # Clip the weight.
    w = np.clip(w, 0.0, 1.0)
    # Blend the local mean and the observation.
    return mean + w * (image - mean)


# Enhanced Lee filter of Lopes, Touzi and Nezry (1990).
def enhanced_lee_filter(
    data: NDArray[Any],  # Intensity (or amplitude with amplitude=True).
    window_size: int = 5,  # Odd window size.
    looks: float = 1.0,  # Number of looks of the image.
    damping: float = 1.0,  # Damping factor of the exponential weight.
    amplitude: bool = False,  # Whether the image holds amplitudes.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image and window.
    image, size = _as_image(data), _check_window(window_size)
    # Local statistics.
    mean, var = local_statistics(image, size)
    # Speckle variation.
    cu = speckle_cv(looks, amplitude)
    # Upper limit of the variation, sqrt(1 + 2 / L).
    cmax = float(np.sqrt(1.0 + 2.0 / looks))
    # Local variation.
    ci = np.sqrt(_ci2(mean, var))
    # Intermediate class Cu < Ci < Cmax, the only one that uses the weight.
    middle = (ci > cu) & (ci < cmax)
    # Distance to Cmax, replaced by 1 outside the class so that no division by
    # zero or overflow occurs there (those pixels are overwritten below).
    gap = np.where(middle, cmax - ci, 1.0)
    # Weight outside the class: 0, or NaN where the variation is undefined.
    rest = np.where(np.isnan(ci), np.nan, 0.0)
    # exp(-damping (Ci - Cu) / (Cmax - Ci)) in the class; the exponent is not
    # positive there, so the weight lies in [0, 1].
    w = np.where(middle, np.exp(-damping * np.where(middle, ci - cu, 0.0) / gap), rest)
    # Homogeneous windows take the mean.
    out = np.where(ci <= cu, mean, mean * w + image * (1.0 - w))
    # Point targets and strong edges keep the observation.
    out = np.where(ci >= cmax, image, out)
    # Keep NaN at invalid pixels.
    return np.where(np.isfinite(image), out, np.nan)


# Frost et al. (1982) exponentially weighted filter.
def frost_filter(
    data: NDArray[Any],  # Intensity or amplitude image.
    window_size: int = 5,  # Odd window size.
    damping: float = 2.0,  # Damping factor K.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image and window.
    image, size = _as_image(data), _check_window(window_size)
    # Local statistics.
    mean, var = local_statistics(image, size)
    # Decay rate K * Ci^2 per unit distance.
    rate = damping * _ci2(mean, var)
    # Half the window size.
    half = size // 2
    # Image padded by reflection so that every offset is defined.
    padded = np.pad(image, half, mode="reflect")
    # Weighted sum and sum of weights.
    num = np.zeros_like(image)
    # Sum of weights.
    den = np.zeros_like(image)
    # Visit every offset of the window.
    for dy in range(-half, half + 1):
        # Visit every column offset.
        for dx in range(-half, half + 1):
            # Shifted image.
            shifted = _shift(image, dy, dx, half)
            # Distance of the offset from the centre.
            dist = float(np.hypot(dy, dx))
            # Exponential weight; invalid neighbours get no weight.
            m = np.where(np.isfinite(shifted), np.exp(-rate * dist), 0.0)
            # Accumulate the weighted values.
            num += m * np.where(np.isfinite(shifted), shifted, 0.0)
            # Accumulate the weights.
            den += m
    # Normalise; invalid centres stay NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Weighted mean.
        out = num / den
    # Keep NaN at invalid pixels.
    return np.where(np.isfinite(image), out, np.nan)


# Gamma maximum a posteriori filter of Lopes et al. (1990) for intensities.
def gamma_map_filter(
    data: NDArray[Any],  # Intensity image.
    window_size: int = 5,  # Odd window size.
    looks: float = 1.0,  # Number of looks of the image.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image and window.
    image, size = _as_image(data), _check_window(window_size)
    # Local statistics.
    mean, var = local_statistics(image, size)
    # Speckle variation for intensities.
    cu2 = speckle_cv(looks) ** 2
    # Upper limit of the variation, sqrt(1 + 2 / L), squared.
    cmax2 = 1.0 + 2.0 / looks
    # Local variation.
    ci2 = _ci2(mean, var)
    # Shape parameter of the scene texture, (1 + Cu^2) / (Ci^2 - Cu^2).
    with np.errstate(divide="ignore", invalid="ignore"):
        # Defined in the intermediate class only.
        alpha = (1.0 + cu2) / (ci2 - cu2)
    # Linear coefficient of the MAP equation.
    b = alpha - looks - 1.0
    # Discriminant of the quadratic MAP equation.
    d = mean * mean * b * b + 4.0 * alpha * looks * image * mean
    # Positive root of the MAP equation.
    with np.errstate(divide="ignore", invalid="ignore"):
        # (b * mean + sqrt(d)) / (2 alpha).
        estimate = (b * mean + np.sqrt(np.maximum(d, 0.0))) / (2.0 * alpha)
    # Homogeneous windows take the mean.
    out = np.where(ci2 <= cu2, mean, estimate)
    # Point targets and strong edges keep the observation.
    out = np.where(ci2 >= cmax2, image, out)
    # Keep NaN at invalid pixels.
    return np.where(np.isfinite(image), out, np.nan)


# Eight edge-aligned 7 x 7 windows of the refined Lee filter.
def _refined_lee_masks() -> list[NDArray[np.float64]]:
    # Row and column indices of a 7 x 7 window.
    i, j = np.mgrid[0:7, 0:7]
    # Half windows on each side of the four edge directions.
    masks = [
        j <= 3,  # Left of a vertical edge.
        j >= 3,  # Right of a vertical edge.
        i <= 3,  # Above a horizontal edge.
        i >= 3,  # Below a horizontal edge.
        j - i >= 0,  # Upper right of a diagonal edge.
        j - i <= 0,  # Lower left of a diagonal edge.
        i + j <= 6,  # Upper left of an anti-diagonal edge.
        i + j >= 6,  # Lower right of an anti-diagonal edge.
    ]  # End of the masks.
    # Masks as float kernels.
    return [m.astype(np.float64) for m in masks]


# Shift an image by (dy, dx) with reflection at the borders.
def _shift(image: NDArray[np.float64], dy: int, dx: int, pad: int) -> NDArray[np.float64]:
    # Pad once by the largest shift.
    padded = np.pad(image, pad, mode="reflect")
    # Slice the shifted view.
    return padded[pad + dy : pad + dy + image.shape[0], pad + dx : pad + dx + image.shape[1]]


# Refined Lee (1981) filter with edge-aligned windows (7 x 7).
def refined_lee_filter(
    data: NDArray[Any],  # Intensity image.
    looks: float = 1.0,  # Number of looks of the image.
) -> NDArray[np.float64]:  # Filtered image.
    # Input image.
    image = _as_image(data)
    # The 7 x 7 window needs at least 4 pixels per side for reflection.
    if min(image.shape) < 4:
        # Report the too small image.
        raise ValueError("refined_lee_filter needs an image of at least 4 x 4 pixels")
    # Means of 3 x 3 sub-windows centred on each pixel.
    m3, _ = local_statistics(image, 3)
    # Sub-window means at offsets -2, 0 and 2 in each direction (3 x 3 grid).
    grid = [[_shift(m3, 2 * (a - 1), 2 * (b - 1), 2) for b in range(3)] for a in range(3)]
    # Gradient magnitudes of the four directions.
    grads = np.stack(
        [  # One gradient image per direction.
            np.abs(grid[1][2] - grid[1][0]),  # Across a vertical edge.
            np.abs(grid[2][1] - grid[0][1]),  # Across a horizontal edge.
            np.abs(grid[0][2] - grid[2][0]),  # Across a diagonal edge.
            np.abs(grid[0][0] - grid[2][2]),  # Across an anti-diagonal edge.
        ]
    )  # End of the gradients.
    # Direction with the strongest gradient; NaN gradients count as zero.
    direction = np.argmax(np.nan_to_num(grads, nan=-1.0), axis=0)
    # The two sides compared for each direction.
    sides = [
        (grid[1][0], grid[1][2]),  # Left and right.
        (grid[0][1], grid[2][1]),  # Top and bottom.
        (grid[0][2], grid[2][0]),  # Upper right and lower left.
        (grid[0][0], grid[2][2]),  # Upper left and lower right.
    ]  # End of the sides.
    # Centre sub-window mean.
    centre = grid[1][1]
    # Index of the chosen mask, 2 * direction + side.
    choice = np.zeros(image.shape, dtype=np.int64)
    # Choose the side whose mean is closer to the centre.
    for d, (first, second) in enumerate(sides):
        # Second side is closer.
        second_closer = np.abs(second - centre) < np.abs(first - centre)
        # Mask index for this direction.
        choice = np.where(direction == d, 2 * d + second_closer.astype(np.int64), choice)
    # Speckle variation of the intensity.
    cu2 = speckle_cv(looks) ** 2
    # Output image.
    out = np.full(image.shape, np.nan)
    # Filter with each edge-aligned window where it was chosen.
    for index, mask in enumerate(_refined_lee_masks()):
        # Pixels that use this window.
        where = choice == index
        # Skip unused windows.
        if not where.any():
            # Next window.
            continue
        # Local moments in the window.
        mean, mean_sq = _masked_mean(image, mask)
        # Variance of the observations.
        var_z = np.maximum(mean_sq - mean * mean, 0.0)
        # Variance of the underlying signal.
        var_x = np.maximum((var_z - mean * mean * cu2) / (1.0 + cu2), 0.0)
        # MMSE weight.
        with np.errstate(divide="ignore", invalid="ignore"):
            # var_x / var_z, zero for constant windows.
            b = np.where(var_z > 0, var_x / np.where(var_z > 0, var_z, 1.0), 0.0)
        # Filtered value.
        out[where] = (mean + b * (image - mean))[where]
    # Keep NaN at invalid pixels.
    return np.where(np.isfinite(image), out, np.nan)


# Median of the finite values of a window, NaN when there are none.
def _valid_median(values: NDArray[np.float64]) -> float:
    # Finite values of the window.
    valid = values[np.isfinite(values)]
    # Median, or NaN without a warning for empty windows.
    return float(np.median(valid)) if valid.size else float("nan")


# Moving-window speckle filter selected by name.
def speckle_filter(
    data: NDArray[Any],  # Intensity (or amplitude) image.
    filter_type: str = "lee",  # One of SPECKLE_FILTERS.
    window_size: int = 5,  # Odd window size (refined_lee always uses 7).
    looks: float = 1.0,  # Number of looks of the image.
) -> NDArray[np.float64]:  # Filtered image.
    # Normalise the name.
    name = filter_type.lower().replace("-", "_")
    # Moving average.
    if name == "boxcar":
        # Mean of the valid pixels in the window.
        mean, _ = local_statistics(_as_image(data), _check_window(window_size))
        # Keep NaN at invalid pixels.
        return np.where(np.isfinite(_as_image(data)), mean, np.nan)
    # Moving median.
    if name == "median":
        # Median of the valid pixels in the window.
        size = _check_window(window_size)
        # Input image.
        image = _as_image(data)
        # NaN-aware median filter; windows without valid pixels give NaN.
        out = ndimage.generic_filter(image, _valid_median, size=size, mode="reflect")
        # Keep NaN at invalid pixels.
        return np.where(np.isfinite(image), out, np.nan)
    # Lee filter.
    if name == "lee":
        # Lee (1980).
        return lee_filter(data, window_size, looks)
    # Kuan filter.
    if name == "kuan":
        # Kuan et al. (1985).
        return kuan_filter(data, window_size, looks)
    # Enhanced Lee filter.
    if name == "enhanced_lee":
        # Lopes, Touzi and Nezry (1990).
        return enhanced_lee_filter(data, window_size, looks)
    # Frost filter.
    if name == "frost":
        # Frost et al. (1982).
        return frost_filter(data, window_size)
    # Gamma MAP filter.
    if name == "gamma_map":
        # Lopes et al. (1990).
        return gamma_map_filter(data, window_size, looks)
    # Refined Lee filter.
    if name == "refined_lee":
        # Lee (1981).
        return refined_lee_filter(data, looks)
    # Unknown name.
    raise ValueError(f"unknown filter type {filter_type!r}; expected one of {SPECKLE_FILTERS}")


# =============================================================================
# End of module src/unbihexium/sar/amplitude.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
