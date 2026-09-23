# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Terrain analysis module.

This module provides functions for computing terrain derivatives
from Digital Elevation Models (DEMs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def slope(dem: NDArray, resolution: float) -> NDArray:
    """Compute slope from DEM.

    Args:
        dem: Digital Elevation Model array
        resolution: Cell size in meters

    Returns:
        Slope array in degrees
    """
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def aspect(dem: NDArray, resolution: float) -> NDArray:
    """Compute aspect from DEM.

    Args:
        dem: Digital Elevation Model array
        resolution: Cell size in meters

    Returns:
        Aspect array in degrees (0-360, north=0)
    """
    dy, dx = np.gradient(dem, resolution)
    aspect_rad = np.arctan2(-dy, dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360 + aspect_deg, aspect_deg)
    return aspect_deg


def hillshade(
    dem: NDArray,
    resolution: float,
    azimuth: float = 315,
    altitude: float = 45,
) -> NDArray:
    """Compute hillshade from DEM.

    Args:
        dem: Digital Elevation Model array
        resolution: Cell size in meters
        azimuth: Sun azimuth in degrees
        altitude: Sun altitude in degrees

    Returns:
        Hillshade array (0-255)
    """
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    hillshade = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(
        az_rad - aspect_rad
    )

    return np.clip(hillshade * 255, 0, 255).astype(np.uint8)


def curvature(dem: NDArray, resolution: float) -> tuple[NDArray, NDArray]:
    """Compute profile and plan curvature from DEM.

    Args:
        dem: Digital Elevation Model array
        resolution: Cell size in meters

    Returns:
        Tuple of (profile_curvature, plan_curvature)
    """
    dy, dx = np.gradient(dem, resolution)
    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)

    p = dx**2 + dy**2
    q = p + 1

    profile = -(dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / (p * np.sqrt(q**3) + 1e-8)
    plan = -(dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (p**1.5 + 1e-8)

    return profile, plan


def twi(dem: NDArray, resolution: float) -> NDArray:
    """Compute Topographic Wetness Index.

    TWI = ln(A / tan(slope))

    Args:
        dem: Digital Elevation Model array
        resolution: Cell size in meters

    Returns:
        TWI array
    """
    slope_arr = slope(dem, resolution)
    slope_rad = np.radians(slope_arr)

    # Simplified flow accumulation (D8)
    flow_acc = np.ones_like(dem)

    return np.log(flow_acc / (np.tan(slope_rad) + 0.001))


def roughness(dem: NDArray) -> NDArray:
    """Compute Terrain Ruggedness Index.

    Args:
        dem: Digital Elevation Model array

    Returns:
        TRI array
    """
    from scipy.ndimage import generic_filter

    def tri_kernel(values):
        center = values[4]
        return np.sqrt(np.mean((values - center) ** 2))

    return generic_filter(dem, tri_kernel, size=3)


__all__ = [
    "slope",
    "aspect",
    "hillshade",
    "curvature",
    "twi",
    "roughness",
]
