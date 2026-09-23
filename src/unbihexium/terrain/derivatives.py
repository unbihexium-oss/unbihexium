# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/terrain/derivatives.py
# Title       : Local surface derivatives of digital elevation models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# 3 x 3 window derivatives of a gridded DEM whose first row is the northern
# edge (the usual north-up raster layout):
#
#   gradient          dz/dx (east) and dz/dy (north), Horn (1981)
#   slope             in degrees, percent or radians
#   aspect            downslope direction, degrees clockwise from north
#   hillshade         illumination by a distant light source, 0 to 255
#   curvature         profile and plan curvature, Zevenbergen and Thorne
#                     (1987)
#   total_curvature   -2 (D + E) of the same polynomial
#   tpi               topographic position index (Weiss, 2001)
#   tri               terrain ruggedness index (Riley et al., 1999, or the
#                     mean absolute difference of Wilson et al., 2007)
#   roughness         range of elevations in the window (Wilson et al., 2007)
#   vrm               vector ruggedness measure (Sappington et al., 2007)
#
# Window layout and borders
# -------------------------
# The window is numbered
#
#   z1 z2 z3        north
#   z4 z5 z6   west       east
#   z7 z8 z9        south
#
# The DEM is extended by one cell with linear extrapolation (2 z_edge -
# z_inner), so a plane keeps its exact slope up to the border. NaN cells
# (nodata) propagate to every result that uses them. `resolution` is the
# cell size in the units of the elevations, either one number or
# (x size, y size); `z_factor` converts elevation units when they differ.
#
# Curvature signs
# ---------------
# With the polynomial z = D x^2 + E y^2 + F x y + G x + H y + z5, the three
# curvatures are the negated second derivatives of the surface along the
# gradient (profile), along the contour (plan) and their sum (total), each
# in 1 / length units without the (1 + |grad z|^2) normalisation. All three
# are positive on convex forms: a slope that steepens downhill, diverging
# contours of a ridge, a hilltop.
#
# References
# ----------
# Horn, B. K. P. (1981). Hill shading and the reflectance map. Proceedings
#   of the IEEE, 69(1), 14-47.
# Zevenbergen, L. W., Thorne, C. R. (1987). Quantitative analysis of land
#   surface topography. Earth Surface Processes and Landforms, 12(1), 47-56.
# Burrough, P. A., McDonnell, R. A. (1998). Principles of Geographical
#   Information Systems. Oxford University Press.
# Riley, S. J., DeGloria, S. D., Elliot, R. (1999). A terrain ruggedness
#   index that quantifies topographic heterogeneity. Intermountain Journal
#   of Sciences, 5(1-4), 23-27.
# Weiss, A. (2001). Topographic position and landforms analysis. Poster,
#   ESRI User Conference, San Diego.
# Wilson, M. F. J., O'Connell, B., Brown, C., Guinan, J. C., Grehan, A. J.
#   (2007). Multiscale terrain analysis of multibeam bathymetry data for
#   habitat mapping on the continental slope. Marine Geodesy, 30(1-2), 3-35.
# Sappington, J. M., Longshore, K. M., Thompson, D. B. (2007). Quantifying
#   landscape ruggedness for animal habitat analysis: a case study using
#   bighorn sheep in the Mojave Desert. Journal of Wildlife Management,
#   71(5), 1419-1426.
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

# Units accepted by slope.
SLOPE_UNITS = ("degrees", "percent", "radians")


# Cell size as (x size, y size), both positive.
def cell_size(resolution: float | tuple[float, float]) -> tuple[float, float]:
    # One number means square cells.
    if np.ndim(resolution) == 0:
        # Same size in both directions.
        xres = yres = float(resolution)  # type: ignore[arg-type]
    # Two numbers give x and y sizes.
    else:
        # Unpack the sizes.
        xres, yres = (abs(float(v)) for v in resolution)  # type: ignore[union-attr]
    # Sizes must be positive.
    if not (xres > 0 and yres > 0):
        # Report the invalid resolution.
        raise ValueError(f"resolution must be positive, got {resolution}")
    # Return the sizes.
    return xres, yres


# DEM as a float64 2-D array of at least 2 x 2 cells.
def as_dem(dem: NDArray[Any]) -> NDArray[np.float64]:
    # Float64 copy.
    z = np.asarray(dem, dtype=np.float64)
    # Only single-band grids are supported.
    if z.ndim != 2 or min(z.shape) < 2:
        # Report the wrong shape.
        raise ValueError(f"DEM must be a 2-D array of at least 2 x 2 cells, got {z.shape}")
    # Return the grid.
    return z


# The nine shifted views z1 .. z9 of the DEM extended by linear extrapolation.
def window_views(dem: NDArray[Any]) -> list[NDArray[np.float64]]:
    # Extend by one cell; reflect_type="odd" gives 2 z_edge - z_inner.
    p = np.pad(as_dem(dem), 1, mode="reflect", reflect_type="odd")
    # Size of the original grid.
    rows, cols = p.shape[0] - 2, p.shape[1] - 2
    # Views in reading order z1 .. z9.
    return [p[r : r + rows, c : c + cols] for r in range(3) for c in range(3)]


# Horn (1981) gradient: dz/dx towards east and dz/dy towards north.
def gradient(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
    z_factor: float = 1.0,  # Elevation unit conversion.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # dz/dx and dz/dy.
    # Cell sizes.
    xres, yres = cell_size(resolution)
    # Window views.
    z1, z2, z3, z4, _, z6, z7, z8, z9 = window_views(dem)
    # East minus west, weighted 1-2-1.
    dzdx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * xres)
    # North minus south, weighted 1-2-1.
    dzdy = ((z1 + 2.0 * z2 + z3) - (z7 + 2.0 * z8 + z9)) / (8.0 * yres)
    # Apply the elevation factor.
    return dzdx * z_factor, dzdy * z_factor


# Slope of the surface.
def slope(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
    units: str = "degrees",  # One of SLOPE_UNITS.
    z_factor: float = 1.0,  # Elevation unit conversion.
) -> NDArray[np.float64]:  # Slope per cell.
    # Gradient components.
    dzdx, dzdy = gradient(dem, resolution, z_factor)
    # Gradient magnitude, the tangent of the slope.
    rise = np.hypot(dzdx, dzdy)
    # Percent rise.
    if units == "percent":
        # 100 tan(slope).
        return 100.0 * rise
    # Angle in radians.
    if units == "radians":
        # arctan of the rise.
        return np.arctan(rise)
    # Angle in degrees.
    if units == "degrees":
        # arctan in degrees.
        return np.degrees(np.arctan(rise))
    # Unknown unit.
    raise ValueError(f"unknown slope units {units!r}; expected one of {SLOPE_UNITS}")


# Aspect: compass direction the slope faces (downslope), NaN on flat cells.
def aspect(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
) -> NDArray[np.float64]:  # Degrees in [0, 360), clockwise from north.
    # Gradient components.
    dzdx, dzdy = gradient(dem, resolution)
    # Downslope direction (-dz/dx, -dz/dy) as an azimuth.
    azimuth = np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0
    # Flat cells have no aspect.
    return np.where((dzdx == 0) & (dzdy == 0), np.nan, azimuth)


# Hillshade for a light source at the given azimuth and altitude (Burrough and McDonnell, 1998).
def hillshade(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
    azimuth: float = 315.0,  # Light azimuth, degrees clockwise from north.
    altitude: float = 45.0,  # Light elevation above the horizon in degrees.
    z_factor: float = 1.0,  # Elevation unit conversion (vertical exaggeration).
) -> NDArray[np.float64]:  # Brightness in [0, 255], NaN at nodata.
    # The altitude must lie between the horizon and the zenith.
    if not 0.0 <= altitude <= 90.0:
        # Report the invalid altitude.
        raise ValueError(f"altitude must lie in [0, 90] degrees, got {altitude}")
    # Gradient components.
    dzdx, dzdy = gradient(dem, resolution, z_factor)
    # Slope angle.
    s = np.arctan(np.hypot(dzdx, dzdy))
    # Downslope azimuth; its value on flat cells does not matter because sin(s) = 0.
    a = np.arctan2(-dzdx, -dzdy)
    # Zenith angle of the light.
    zenith = np.radians(90.0 - altitude)
    # Cosine of the incidence angle of the light on the surface.
    relative = np.radians(azimuth) - a
    # Lambertian illumination of the tilted surface.
    cos_i = np.cos(zenith) * np.cos(s) + np.sin(zenith) * np.sin(s) * np.cos(relative)
    # Self-shadowed cells are black.
    return 255.0 * np.clip(cos_i, 0.0, 1.0)


# Coefficients D, E, F, G, H of the Zevenbergen and Thorne (1987) polynomial.
def _zt_coefficients(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float],  # Cell size.
) -> tuple[NDArray[np.float64], ...]:  # D, E, F, G, H.
    # Cell sizes.
    xres, yres = cell_size(resolution)
    # Window views.
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = window_views(dem)
    # Second derivative in x, halved.
    d = ((z4 + z6) / 2.0 - z5) / xres**2
    # Second derivative in y, halved.
    e = ((z2 + z8) / 2.0 - z5) / yres**2
    # Mixed derivative with y towards north.
    f = (-z1 + z3 + z7 - z9) / (4.0 * xres * yres)
    # First derivative towards east.
    g = (z6 - z4) / (2.0 * xres)
    # First derivative towards north.
    h = (z2 - z8) / (2.0 * yres)
    # Return the coefficients.
    return d, e, f, g, h


# Profile and plan curvature (Zevenbergen and Thorne, 1987), in 1 / length units.
def curvature(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Profile and plan curvature.
    # Polynomial coefficients.
    d, e, f, g, h = _zt_coefficients(dem, resolution)
    # Squared gradient; curvatures along and across the slope need a direction.
    p = g * g + h * h
    # Avoid dividing by zero on flat cells.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Profile curvature, positive where the slope steepens downhill (convex).
        profile = np.where(p > 0, -2.0 * (d * g * g + e * h * h + f * g * h) / p, 0.0)
        # Plan curvature, positive for divergent contours (ridges and noses).
        plan = np.where(p > 0, -2.0 * (d * h * h + e * g * g - f * g * h) / p, 0.0)
    # Keep NaN from the input.
    nan = np.isnan(p)
    # Return both curvatures.
    return np.where(nan, np.nan, profile), np.where(nan, np.nan, plan)


# Total curvature -2 (D + E): positive on convex (upward bulging) cells.
def total_curvature(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
) -> NDArray[np.float64]:  # Curvature in 1 / length units.
    # Polynomial coefficients.
    d, e, _, _, _ = _zt_coefficients(dem, resolution)
    # Negative Laplacian halved.
    return -2.0 * (d + e)


# Topographic position index: elevation minus the mean of its neighbourhood (Weiss, 2001).
def tpi(
    dem: NDArray[Any],  # Elevation grid.
    radius: int = 1,  # Neighbourhood half-width in cells (square window).
) -> NDArray[np.float64]:  # Positive on ridges, negative in valleys.
    # The radius must be positive.
    if radius < 1:
        # Report the invalid radius.
        raise ValueError(f"radius must be >= 1, got {radius}")
    # Elevations.
    z = as_dem(dem)
    # Square window without its centre.
    kernel = np.ones((2 * radius + 1, 2 * radius + 1))
    # Exclude the centre cell.
    kernel[radius, radius] = 0.0
    # Valid cells.
    valid = np.isfinite(z)
    # Sum of valid neighbours.
    total = ndimage.correlate(np.where(valid, z, 0.0), kernel, mode="constant")
    # Number of valid neighbours.
    count = ndimage.correlate(valid.astype(np.float64), kernel, mode="constant")
    # Mean of the neighbours, NaN without any.
    mean = np.divide(total, count, out=np.full(z.shape, np.nan), where=count > 0)
    # Difference from the neighbourhood mean.
    return z - mean


# Terrain ruggedness index of the 3 x 3 window.
def tri(
    dem: NDArray[Any],  # Elevation grid.
    method: str = "riley",  # "riley": sqrt of summed squares; "wilson": mean absolute difference.
) -> NDArray[np.float64]:  # Ruggedness in elevation units.
    # Window views.
    views = window_views(dem)
    # Centre cell.
    centre = views[4]
    # Differences to the eight neighbours.
    diffs = np.stack([v - centre for i, v in enumerate(views) if i != 4])
    # Riley et al. (1999).
    if method == "riley":
        # Square root of the summed squared differences.
        return np.sqrt(np.sum(diffs * diffs, axis=0))
    # Wilson et al. (2007).
    if method == "wilson":
        # Mean absolute difference.
        return np.mean(np.abs(diffs), axis=0)
    # Unknown method.
    raise ValueError(f"unknown TRI method {method!r}; expected 'riley' or 'wilson'")


# Roughness: largest minus smallest elevation of the 3 x 3 window (Wilson et al., 2007).
def roughness(dem: NDArray[Any]) -> NDArray[np.float64]:
    # Window views.
    stack = np.stack(window_views(dem))
    # Range of the window; NaN propagates.
    return np.max(stack, axis=0) - np.min(stack, axis=0)


# Vector ruggedness measure (Sappington et al., 2007): 0 flat or planar, 1 maximally rugged.
def vrm(
    dem: NDArray[Any],  # Elevation grid, north up.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
    window_size: int = 3,  # Odd neighbourhood size.
) -> NDArray[np.float64]:  # Dispersion of the surface normals.
    # The window must be odd and at least 3 cells wide.
    if window_size < 3 or window_size % 2 == 0:
        # Report the invalid window.
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    # Gradient components.
    dzdx, dzdy = gradient(dem, resolution)
    # Unit normal of the surface z = f(x, y) is (-dzdx, -dzdy, 1) / norm.
    norm = np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
    # Components of the unit normals.
    nx, ny, nz = -dzdx / norm, -dzdy / norm, 1.0 / norm
    # Sum of each component over the window.
    size = (window_size, window_size)
    # Mean vector components; means keep the ratio |R| / n.
    sx = ndimage.uniform_filter(nx, size=size, mode="nearest")
    # North component.
    sy = ndimage.uniform_filter(ny, size=size, mode="nearest")
    # Vertical component.
    sz = ndimage.uniform_filter(nz, size=size, mode="nearest")
    # One minus the length of the mean normal.
    return 1.0 - np.sqrt(sx * sx + sy * sy + sz * sz)


# =============================================================================
# End of module src/unbihexium/terrain/derivatives.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
