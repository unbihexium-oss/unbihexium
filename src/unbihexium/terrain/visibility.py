# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/terrain/visibility.py
# Title       : Line-of-sight viewshed of a digital elevation model
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# viewshed() marks the cells of a DEM that are visible from an observer
# cell. For every target cell, the line of sight from the observer's eye
# (ground plus observer height) to the target (ground plus target height)
# is sampled once per crossed row or column; the terrain is interpolated
# bilinearly at the samples. The target is visible when the tangent of its
# elevation angle is at least the largest tangent of the intermediate
# samples, which is the exact "R3" test of Franklin and Ray (1994) on an
# interpolated surface. The cost is O(N D) for N cells and a line length of
# D cells; limit it with `max_distance`.
#
# Optionally, the elevations are lowered by the curvature of the Earth,
# reduced by atmospheric refraction: dz = (1 - k) d^2 / (2 R), with the
# refraction coefficient k = 0.13 and R = 6 371 000 m (the correction used
# by common GIS viewshed tools).
#
# References
# ----------
# Franklin, W. R., Ray, C. K. (1994). Higher isn't necessarily better:
#   visibility algorithms and experiments. Proc. 6th International
#   Symposium on Spatial Data Handling, Edinburgh, 751-770.
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

# Bilinear interpolation at fractional positions.
from scipy.ndimage import map_coordinates

# Shared DEM helpers.
from unbihexium.terrain.derivatives import as_dem, cell_size

# Mean Earth radius in metres.
EARTH_RADIUS = 6_371_000.0


# Visibility of every cell from an observer cell.
def viewshed(
    dem: NDArray[Any],  # Elevation grid in metres.
    observer: tuple[int, int],  # (row, col) of the observer.
    resolution: float | tuple[float, float] = 1.0,  # Cell size in metres.
    observer_height: float = 1.7,  # Eye height above ground.
    target_height: float = 0.0,  # Height of the targets above ground.
    max_distance: float | None = None,  # Maximum distance in metres; farther cells are hidden.
    earth_curvature: bool = False,  # Apply the curvature and refraction correction.
    refraction: float = 0.13,  # Refraction coefficient k.
) -> NDArray[np.bool_]:  # True where the target is visible.
    # Elevations.
    z = as_dem(dem)
    # Cell sizes.
    xres, yres = cell_size(resolution)
    # Grid size.
    rows, cols = z.shape
    # Observer position.
    r0, c0 = observer
    # The observer must stand on a valid cell.
    if not (0 <= r0 < rows and 0 <= c0 < cols) or not np.isfinite(z[r0, c0]):
        # Report the invalid observer.
        raise ValueError(f"observer {observer} must be a valid cell of the grid {z.shape}")
    # Row and column of every cell.
    rr, cc = np.indices(z.shape)
    # Horizontal distance of every cell from the observer.
    dist = np.hypot((rr - r0) * yres, (cc - c0) * xres)
    # Drop of the Earth surface below the tangent plane, reduced by refraction.
    bend = (1.0 - refraction) / (2.0 * EARTH_RADIUS) if earth_curvature else 0.0
    # Eye elevation (no drop at distance zero).
    eye = z[r0, c0] + observer_height
    # Samples per line: one per crossed row or column.
    steps = np.maximum(np.abs(rr - r0), np.abs(cc - c0)).ravel()
    # Targets beyond the maximum distance need no line.
    if max_distance is not None:
        # No samples for them.
        steps = np.where(dist.ravel() <= max_distance, steps, 0)
    # Tangent of the elevation angle of each target.
    with np.errstate(divide="ignore", invalid="ignore"):
        # (corrected target elevation - eye) / distance.
        target_tan = ((z - bend * dist * dist + target_height - eye) / dist).ravel()
    # Largest tangent of the intermediate samples.
    horizon = np.full(steps.size, -np.inf)
    # Nodata cells block the view like very high terrain.
    surface = np.where(np.isfinite(z), z, np.inf)
    # Row and column offsets of the targets.
    dr, dc = (rr - r0).ravel().astype(np.float64), (cc - c0).ravel().astype(np.float64)
    # Walk along the lines, one sample index at a time for all targets.
    for k in range(1, int(steps.max(initial=0))):
        # Targets whose line has an intermediate sample k.
        sel = steps > k
        # Fraction of the way to the target.
        t = k / steps[sel]
        # Sample position.
        pr, pc = r0 + t * dr[sel], c0 + t * dc[sel]
        # Terrain at the samples; interpolation with inf gives inf or NaN.
        h = map_coordinates(surface, [pr, pc], order=1, mode="nearest")
        # NaN from inf arithmetic blocks the view.
        h = np.where(np.isnan(h), np.inf, h)
        # Horizontal distance of the samples.
        sd = t * dist.ravel()[sel]
        # Curvature correction at the sample distance.
        h = h - bend * sd * sd
        # Tangent of the sample's elevation angle.
        horizon[sel] = np.maximum(horizon[sel], (h - eye) / sd)
    # Visible targets are not below the horizon.
    visible = target_tan >= horizon - 1e-12
    # The observer sees its own cell.
    visible[r0 * cols + c0] = True
    # Nodata cells are not visible.
    visible &= np.isfinite(z).ravel()
    # Cells beyond the maximum distance are not visible.
    if max_distance is not None:
        # Distance limit.
        visible &= dist.ravel() <= max_distance
    # Back to the grid.
    return visible.reshape(z.shape)


# =============================================================================
# End of module src/unbihexium/terrain/visibility.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
