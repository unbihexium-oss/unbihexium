# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/visualization/relief.py
# Title       : Hillshading and shaded relief overlays
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Relief shading of digital elevation models:
#
#   slope_aspect     slope and aspect in radians from Horn's gradients
#   hillshade        illumination in [0, 1] (or bytes) for a sun position
#   multidirectional_hillshade
#                    weighted blend of several light directions
#   shade_image      multiply an RGB image by a hillshade
#
# Method
# ------
# Horn (1981) estimates the gradients from the 3 x 3 neighbourhood
#   a b c
#   d e f
#   g h i
# as dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 dx) and
# dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 dy), rows running south.
# With the zenith Z = 90 deg - altitude, slope S = atan(z sqrt(dzdx^2 +
# dzdy^2)), aspect A = atan2(dzdy, -dzdx) and the sun azimuth converted to
# the mathematical convention M = 90 deg - azimuth, the illumination is
# cos Z cos S + sin Z sin S cos(M - A), clipped at zero (Burrough and
# McDonnell, 1998). Edges replicate the border pixels; pixels next to NaN
# are NaN.
#
# References
# ----------
#   Horn, B. K. P. (1981). Hill shading and the reflectance map. Proceedings
#     of the IEEE 69(1), 14-47.
#   Burrough, P. A., McDonnell, R. A. (1998). Principles of Geographical
#     Information Systems. Oxford University Press.
#   Mark, R. K. (1992). Multidirectional, oblique-weighted, shaded-relief
#     image of the Island of Hawaii. U.S. Geological Survey Open-File Report
#     92-422.
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


# Pixel sizes (dx, dy) from a scalar or a pair.
def _cellsize(cellsize: float | tuple[float, float]) -> tuple[float, float]:
    # A pair gives x and y separately.
    dx, dy = (cellsize, cellsize) if isinstance(cellsize, (int, float, np.number)) else cellsize
    # Absolute values: north-up transforms have negative y sizes.
    dx, dy = abs(float(dx)), abs(float(dy))
    # Sizes must be positive.
    if dx == 0 or dy == 0:
        # Explain the requirement.
        raise ValueError("cellsize must be non-zero")
    # Return both.
    return dx, dy


# Slope and aspect in radians from Horn's gradients.
def slope_aspect(
    dem: NDArray[Any],  # (H, W) elevations.
    cellsize: float | tuple[float, float] = 1.0,  # Pixel size in elevation units.
    z_factor: float = 1.0,  # Vertical exaggeration or unit conversion.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Slope and aspect.
    # Elevations as float.
    z = np.asarray(dem, dtype=np.float64)
    # Only single bands are supported.
    if z.ndim != 2:
        # Explain the requirement.
        raise ValueError(f"expected an (H, W) DEM, got shape {z.shape}")
    # Pixel sizes.
    dx, dy = _cellsize(cellsize)
    # Replicate the border so that the output has the input size.
    p = np.pad(z, 1, mode="edge")
    # Neighbours a, b, c of the row above.
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    # Neighbours d and f of the same row.
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    # Neighbours g, h, i of the row below.
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    # Gradient towards the east.
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx) * z_factor
    # Gradient towards the south.
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy) * z_factor
    # Slope angle.
    slope = np.arctan(np.hypot(dzdx, dzdy))
    # Aspect in the mathematical convention, in [0, 2 pi).
    aspect = np.mod(np.arctan2(dzdy, -dzdx), 2.0 * np.pi)
    # Return both.
    return slope, aspect


# Illumination of a DEM for one sun position.
def hillshade(
    dem: NDArray[Any],  # (H, W) elevations.
    cellsize: float | tuple[float, float] = 1.0,  # Pixel size in elevation units.
    azimuth: float = 315.0,  # Sun azimuth in degrees clockwise from north.
    altitude: float = 45.0,  # Sun altitude above the horizon in degrees.
    z_factor: float = 1.0,  # Vertical exaggeration or unit conversion.
    as_uint8: bool = False,  # Return bytes 0 to 255 instead of [0, 1].
) -> NDArray[Any]:  # Illumination.
    # The altitude must lie in [0, 90].
    if not 0.0 <= altitude <= 90.0:
        # Explain the requirement.
        raise ValueError("altitude must lie between 0 and 90 degrees")
    # Slope and aspect.
    slope, aspect = slope_aspect(dem, cellsize, z_factor)
    # Zenith angle of the sun.
    zenith = np.deg2rad(90.0 - altitude)
    # Azimuth in the mathematical convention.
    az = np.deg2rad(np.mod(360.0 - azimuth + 90.0, 360.0))
    # Lambertian illumination.
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)
    # Surfaces facing away are dark.
    shade = np.clip(shade, 0.0, 1.0)
    # Bytes on request; NaN becomes 0.
    if as_uint8:
        # Scale and round.
        return np.round(np.nan_to_num(shade) * 255.0).astype(np.uint8)
    # Return [0, 1].
    return shade


# Weighted blend of hillshades from several directions.
def multidirectional_hillshade(
    dem: NDArray[Any],  # (H, W) elevations.
    cellsize: float | tuple[float, float] = 1.0,  # Pixel size in elevation units.
    azimuths: Sequence[float] = (225.0, 270.0, 315.0, 360.0),  # Light directions.
    weights: Sequence[float] | None = None,  # Weight of every direction.
    altitude: float = 45.0,  # Sun altitude in degrees.
    z_factor: float = 1.0,  # Vertical exaggeration.
) -> NDArray[np.float64]:  # Illumination in [0, 1].
    # Equal weights by default.
    w = np.ones(len(azimuths)) if weights is None else np.asarray(weights, dtype=np.float64)
    # One positive weight sum per direction list.
    if w.size != len(azimuths) or w.sum() <= 0:
        # Explain the requirement.
        raise ValueError("expected one weight per azimuth with a positive sum")
    # Hillshade of every direction.
    shades = [hillshade(dem, cellsize, az, altitude, z_factor) for az in azimuths]
    # Weighted mean.
    return np.tensordot(w / w.sum(), np.stack(shades), axes=1)


# Multiply an RGB image by a hillshade.
def shade_image(
    rgb: NDArray[Any],  # (H, W, 3) uint8 image.
    shade: NDArray[Any],  # (H, W) illumination in [0, 1].
    strength: float = 0.6,  # 0 keeps the image, 1 applies the full shade.
) -> NDArray[np.uint8]:  # Shaded image.
    # Strength must lie in [0, 1].
    if not 0.0 <= strength <= 1.0:
        # Explain the requirement.
        raise ValueError("strength must lie between 0 and 1")
    # Illumination in [0, 1], NaN as fully lit.
    s = np.clip(np.nan_to_num(np.asarray(shade, dtype=np.float64), nan=1.0), 0.0, 1.0)
    # Blend factor between no shading and full shading.
    factor = 1.0 - strength + strength * s
    # Multiply every channel.
    out = np.asarray(rgb, dtype=np.float64)[..., :3] * factor[..., None]
    # Round to bytes.
    return np.round(out).clip(0, 255).astype(np.uint8)


# =============================================================================
# End of module src/unbihexium/visualization/relief.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
