"""SAR amplitude processing functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def calibrate_amplitude(
    data: NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]],
    calibration_factor: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """Calibrate SAR amplitude data.

    Calibration formula:
    $$A_{cal} = |s| \\cdot K$$

    Where $K$ is the calibration factor.

    Args:
        data: Complex SAR data or amplitude data.
        calibration_factor: Calibration constant.

    Returns:
        Calibrated amplitude.
    """
    if np.iscomplexobj(data):
        amplitude = np.abs(data)
    else:
        amplitude = data.astype(np.float32)

    return amplitude * calibration_factor


def compute_sigma0(
    amplitude: NDArray[np.floating[Any]],
    incidence_angle: NDArray[np.floating[Any]] | float,
    calibration_lut: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute radar backscatter coefficient sigma0.

    Formula:
    $$\\sigma^0 = \\frac{A^2 \\sin(\\theta)}{K}$$

    Args:
        amplitude: Calibrated amplitude.
        incidence_angle: Incidence angle in radians.
        calibration_lut: Optional calibration lookup table.

    Returns:
        Sigma0 in linear scale.
    """
    a_squared = amplitude**2

    if isinstance(incidence_angle, (int, float)):
        sin_theta = np.sin(incidence_angle)
    else:
        sin_theta = np.sin(incidence_angle)

    sigma0 = a_squared * sin_theta

    if calibration_lut is not None:
        sigma0 = sigma0 / calibration_lut

    return sigma0.astype(np.float32)


def compute_gamma0(
    sigma0: NDArray[np.floating[Any]],
    incidence_angle: NDArray[np.floating[Any]] | float,
) -> NDArray[np.floating[Any]]:
    """Compute gamma0 from sigma0.

    Formula:
    $$\\gamma^0 = \\frac{\\sigma^0}{\\cos(\\theta)}$$

    Args:
        sigma0: Sigma0 values.
        incidence_angle: Incidence angle in radians.

    Returns:
        Gamma0 values.
    """
    if isinstance(incidence_angle, (int, float)):
        cos_theta = np.cos(incidence_angle)
    else:
        cos_theta = np.cos(incidence_angle)

    # Avoid division by zero
    cos_theta = np.maximum(cos_theta, 1e-10)

    return (sigma0 / cos_theta).astype(np.float32)


def speckle_filter(
    data: NDArray[np.floating[Any]],
    filter_type: str = "lee",
    window_size: int = 5,
) -> NDArray[np.floating[Any]]:
    """Apply speckle filtering to SAR data.

    Supported filters:
    - lee: Lee filter (adaptive)
    - frost: Frost filter
    - boxcar: Simple moving average
    - median: Median filter

    Args:
        data: SAR intensity data.
        filter_type: Filter type.
        window_size: Filter window size (must be odd).

    Returns:
        Filtered data.
    """
    from scipy import ndimage

    if window_size % 2 == 0:
        window_size += 1

    if filter_type == "boxcar":
        return ndimage.uniform_filter(data, size=window_size).astype(np.float32)

    elif filter_type == "median":
        return ndimage.median_filter(data, size=window_size).astype(np.float32)

    elif filter_type == "lee":
        # Lee filter implementation
        mean = ndimage.uniform_filter(data, size=window_size)
        mean_sq = ndimage.uniform_filter(data**2, size=window_size)
        var = mean_sq - mean**2

        # Estimate noise variance (assuming Rayleigh distribution)
        noise_var = var / (mean**2 + 1e-10)
        k = var / (var + noise_var * mean**2 + 1e-10)

        filtered = mean + k * (data - mean)
        return filtered.astype(np.float32)

    elif filter_type == "frost":
        # Simplified Frost filter
        mean = ndimage.uniform_filter(data, size=window_size)
        var = ndimage.uniform_filter(data**2, size=window_size) - mean**2

        cv = np.sqrt(var) / (mean + 1e-10)
        alpha = cv**2

        # Weighted average based on distance
        filtered = ndimage.uniform_filter(data * np.exp(-alpha), size=window_size)
        weights = ndimage.uniform_filter(np.exp(-alpha), size=window_size)

        return (filtered / (weights + 1e-10)).astype(np.float32)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def amplitude_to_db(
    amplitude: NDArray[np.floating[Any]],
    floor: float = -40.0,
) -> NDArray[np.floating[Any]]:
    """Convert amplitude to decibels.

    Formula:
    $$dB = 10 \\log_{10}(A^2) = 20 \\log_{10}(A)$$

    Args:
        amplitude: Amplitude values.
        floor: Minimum dB value (for zero/negative inputs).

    Returns:
        Values in dB.
    """
    amplitude = np.maximum(amplitude, 1e-10)
    db = 20.0 * np.log10(amplitude)
    return np.maximum(db, floor).astype(np.float32)
