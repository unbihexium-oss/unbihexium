# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/terrain/hydrology.py
# Title       : Depression filling, D8 flow routing and wetness index
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Surface flow analysis of a gridded DEM (first row north):
#
#   fill_depressions    priority-flood filling of pits, optionally with a
#                       small gradient across flats (Barnes et al., 2014)
#   flow_direction_d8   steepest descent towards one of eight neighbours
#                       (O'Callaghan and Mark, 1984)
#   flow_accumulation   number (or weight) of upstream cells
#   watershed           cells that drain to an outlet
#   extract_streams     cells whose accumulation reaches a threshold
#   twi                 topographic wetness index ln(a / tan(beta))
#                       (Beven and Kirkby, 1979)
#
# D8 codes
# --------
# Directions use the ESRI powers of two: 1 east, 2 south-east, 4 south,
# 8 south-west, 16 west, 32 north-west, 64 north, 128 north-east. Code 0
# marks cells without a lower neighbour (pits, flats, nodata and border
# cells that would drain off the grid). Drops to diagonal neighbours are
# divided by the diagonal distance. NaN cells are nodata: they are never
# receivers, and priority-flood treats them as outlets like the border.
#
# Flow accumulation
# -----------------
# The accumulation of a cell counts its upstream cells, excluding the cell
# itself (the ArcGIS convention). The specific catchment area of the TWI is
# (accumulation + 1) * cell area / cell width.
#
# References
# ----------
# O'Callaghan, J. F., Mark, D. M. (1984). The extraction of drainage
#   networks from digital elevation data. Computer Vision, Graphics, and
#   Image Processing, 28(3), 323-344.
# Jenson, S. K., Domingue, J. O. (1988). Extracting topographic structure
#   from digital elevation data for geographic information system analysis.
#   Photogrammetric Engineering and Remote Sensing, 54(11), 1593-1600.
# Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal
#   depression-filling and watershed-labeling algorithm for digital
#   elevation models. Computers and Geosciences, 62, 117-127.
# Beven, K. J., Kirkby, M. J. (1979). A physically based, variable
#   contributing area model of basin hydrology. Hydrological Sciences
#   Bulletin, 24(1), 43-69.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Priority queue of priority-flood.
import heapq

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Shared DEM helpers.
from unbihexium.terrain.derivatives import as_dem, cell_size, slope

# D8 neighbours as (row offset, column offset, ESRI code).
D8_NEIGHBOURS = (
    (0, 1, 1),  # East.
    (1, 1, 2),  # South-east.
    (1, 0, 4),  # South.
    (1, -1, 8),  # South-west.
    (0, -1, 16),  # West.
    (-1, -1, 32),  # North-west.
    (-1, 0, 64),  # North.
    (-1, 1, 128),  # North-east.
)  # End of the neighbours.


# Fill depressions so that every valid cell drains to the border or to nodata.
def fill_depressions(
    dem: NDArray[Any],  # Elevation grid.
    epsilon: float = 0.0,  # Minimum rise per cell across filled areas; 0 leaves flats.
) -> NDArray[np.float64]:  # Filled elevations.
    # The gradient increment cannot be negative.
    if epsilon < 0:
        # Report the invalid increment.
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    # Copy of the elevations.
    z = as_dem(dem).copy()
    # Grid size.
    rows, cols = z.shape
    # Valid cells.
    valid = np.isfinite(z)
    # Cells already in the queue or processed.
    closed = ~valid
    # Seeds: valid border cells and valid cells next to nodata.
    seed = np.zeros(z.shape, dtype=bool)
    # Border rows and columns.
    seed[[0, -1], :] = True
    # Border columns.
    seed[:, [0, -1]] = True
    # Cells next to nodata drain into it.
    if not valid.all():
        # Nodata mask extended by one cell in each direction.
        padded = np.pad(~valid, 1, constant_values=False)
        # Any neighbour is nodata.
        near = np.zeros(z.shape, dtype=bool)
        # Visit the eight neighbours.
        for dr, dc, _ in D8_NEIGHBOURS:
            # Neighbour is nodata.
            near |= padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        # Add them to the seeds.
        seed |= near
    # Only valid cells are seeds.
    seed &= valid
    # Priority queue of (elevation, insertion order, flat index).
    heap: list[tuple[float, int, int]] = []
    # Insertion counter gives a deterministic order among equal elevations.
    order = 0
    # Push the seeds.
    for index in np.flatnonzero(seed):
        # Seed elevation.
        heap.append((float(z.flat[index]), order, int(index)))
        # Next insertion number.
        order += 1
    # Build the heap.
    heapq.heapify(heap)
    # Mark the seeds.
    closed |= seed
    # Grow inwards from the lowest cell of the queue.
    while heap:
        # Lowest open cell.
        level, _, index = heapq.heappop(heap)
        # Its position.
        r, c = divmod(index, cols)
        # Visit the neighbours.
        for dr, dc, _ in D8_NEIGHBOURS:
            # Neighbour position.
            nr, nc = r + dr, c + dc
            # Skip positions outside the grid or already closed.
            if not (0 <= nr < rows and 0 <= nc < cols) or closed[nr, nc]:
                # Next neighbour.
                continue
            # Close the neighbour.
            closed[nr, nc] = True
            # Raise cells below the spill level (plus the increment).
            if z[nr, nc] <= level:
                # Filled elevation.
                z[nr, nc] = level + epsilon if epsilon > 0 else level
            # Queue the neighbour at its (possibly raised) elevation.
            heapq.heappush(heap, (float(z[nr, nc]), order, nr * cols + nc))
            # Next insertion number.
            order += 1
    # Return the filled DEM.
    return z


# D8 flow direction codes of steepest descent.
def flow_direction_d8(
    dem: NDArray[Any],  # Elevation grid, preferably filled.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
) -> NDArray[np.uint8]:  # ESRI D8 codes, 0 where no neighbour is lower.
    # Elevations.
    z = as_dem(dem)
    # Cell sizes.
    xres, yres = cell_size(resolution)
    # Grid size.
    rows, cols = z.shape
    # Neighbours outside the grid behave as +inf (never lower).
    padded = np.pad(z, 1, constant_values=np.inf)
    # NaN neighbours are never receivers.
    padded = np.where(np.isnan(padded), np.inf, padded)
    # Best drop found so far.
    best = np.zeros(z.shape)
    # Code of the best neighbour.
    codes = np.zeros(z.shape, dtype=np.uint8)
    # Visit the neighbours in code order; ties keep the first.
    for dr, dc, code in D8_NEIGHBOURS:
        # Elevation of the neighbour.
        neighbour = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        # Distance to the neighbour.
        dist = float(np.hypot(dr * yres, dc * xres))
        # Drop per unit distance; NaN centres never flow.
        with np.errstate(invalid="ignore"):
            # Positive for lower neighbours.
            drop = (z - neighbour) / dist
        # Strictly steeper than the best so far.
        better = drop > best
        # Record the drop.
        best = np.where(better, drop, best)
        # Record the code.
        codes = np.where(better, np.uint8(code), codes)
    # Return the codes.
    return codes


# Flat index of the receiving cell of every cell, -1 where there is none.
def _receivers(flow_dir: NDArray[Any]) -> NDArray[np.int64]:
    # Codes as integers.
    codes = np.asarray(flow_dir).astype(np.int64)
    # Only valid D8 codes and 0 are accepted.
    known = np.isin(codes, [0] + [code for _, _, code in D8_NEIGHBOURS])
    # Report unknown codes.
    if not known.all():
        # Name the first unknown code.
        raise ValueError(f"invalid D8 code {codes[~known][0]}; expected 0 or powers of two to 128")
    # Grid size.
    rows, cols = codes.shape
    # Row and column of every cell.
    r, c = np.indices(codes.shape)
    # Receivers default to none.
    receiver = np.full(codes.shape, -1, dtype=np.int64)
    # Resolve every direction.
    for dr, dc, code in D8_NEIGHBOURS:
        # Cells flowing in this direction.
        sel = codes == code
        # Target positions.
        tr, tc = r[sel] + dr, c[sel] + dc
        # Targets inside the grid.
        inside = (tr >= 0) & (tr < rows) & (tc >= 0) & (tc < cols)
        # Flat index of the target, -1 outside.
        receiver[sel] = np.where(inside, tr * cols + tc, -1)
    # Return the flat receivers.
    return receiver.ravel()


# Upstream cell count (or summed weights) of every cell.
def flow_accumulation(
    flow_dir: NDArray[Any],  # D8 codes.
    weights: NDArray[Any] | None = None,  # Per-cell weight, for example rainfall; 1 by default.
) -> NDArray[np.float64]:  # Accumulation, excluding the cell's own weight.
    # Receivers of all cells.
    receiver = _receivers(flow_dir)
    # Number of cells.
    n = receiver.size
    # Cell weights.
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    # Weights must match the grid.
    if w.size != n:
        # Report the mismatch.
        raise ValueError(f"weights size {w.size} differs from the grid size {n}")
    # Accumulated upstream weight.
    acc = np.zeros(n)
    # Donors not yet processed per cell.
    pending = np.bincount(receiver[receiver >= 0], minlength=n)
    # Cells whose donors are all processed.
    frontier = np.flatnonzero(pending == 0)
    # Pass accumulations downstream wave by wave.
    while frontier.size:
        # Receivers of the frontier.
        targets = receiver[frontier]
        # Only cells with a receiver pass their flow.
        has = targets >= 0
        # Donors and receivers.
        donors, targets = frontier[has], targets[has]
        # Pass the donor's accumulation plus its own weight.
        np.add.at(acc, targets, acc[donors] + w[donors])
        # One donor fewer per receiver.
        np.subtract.at(pending, targets, 1)
        # Receivers that are now complete form the next frontier.
        frontier = np.unique(targets[pending[targets] == 0])
    # Back to the grid.
    return acc.reshape(np.shape(flow_dir))


# Cells that drain to an outlet cell.
def watershed(
    flow_dir: NDArray[Any],  # D8 codes.
    outlet: tuple[int, int],  # (row, col) of the outlet.
) -> NDArray[np.bool_]:  # True for the outlet and every cell upstream of it.
    # Receivers of all cells.
    receiver = _receivers(flow_dir)
    # Grid shape.
    shape = np.shape(flow_dir)
    # Outlet position.
    row, col = outlet
    # The outlet must lie inside the grid.
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        # Report the invalid outlet.
        raise ValueError(f"outlet {outlet} lies outside the grid of shape {shape}")
    # Cells of the watershed.
    inside = np.zeros(receiver.size, dtype=bool)
    # Start from the outlet.
    inside[row * shape[1] + col] = True
    # Add donors until no cell joins.
    while True:
        # Cells whose receiver is in the watershed.
        joins = (receiver >= 0) & ~inside
        # Check their receivers.
        joins[joins] = inside[receiver[joins]]
        # Stop when nothing changes.
        if not joins.any():
            # Done.
            break
        # Add the new cells.
        inside |= joins
    # Back to the grid.
    return inside.reshape(shape)


# Stream cells: accumulation at or above a threshold.
def extract_streams(
    accumulation: NDArray[Any],  # Flow accumulation.
    threshold: float,  # Minimum number of upstream cells.
) -> NDArray[np.bool_]:  # Stream mask.
    # Threshold the accumulation; NaN is never a stream.
    return np.asarray(accumulation, dtype=np.float64) >= threshold


# Topographic wetness index ln(a / tan(beta)) (Beven and Kirkby, 1979).
def twi(
    dem: NDArray[Any],  # Elevation grid.
    resolution: float | tuple[float, float] = 1.0,  # Cell size.
    fill: bool = True,  # Fill depressions before routing.
    min_slope: float = 0.1,  # Slope floor in degrees, keeps flat cells finite.
) -> NDArray[np.float64]:  # Wetness index, NaN at nodata.
    # Elevations.
    z = as_dem(dem)
    # Cell sizes.
    xres, yres = cell_size(resolution)
    # Routing surface.
    surface = fill_depressions(z, epsilon=1e-6) if fill else z
    # Upstream cells.
    acc = flow_accumulation(flow_direction_d8(surface, (xres, yres)))
    # Specific catchment area: contributing area per unit contour width.
    area = (acc + 1.0) * xres * yres / np.sqrt(xres * yres)
    # Local slope of the original DEM in radians.
    beta = np.radians(np.maximum(slope(z, (xres, yres)), min_slope))
    # Wetness index; NaN slopes give NaN.
    return np.log(area / np.tan(beta))


# =============================================================================
# End of module src/unbihexium/terrain/hydrology.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
