# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/network/cost_surface.py
# Title       : Accumulated cost distance and least-cost paths on rasters
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# A friction (cost) raster gives the cost of crossing one unit of distance
# in each cell. Every cell is a graph node connected to its 4 or 8
# neighbours; moving from cell a to cell b costs
#
#   (c_a + c_b) / 2 * step length
#
# with step lengths of one cell size, or the diagonal for diagonal moves
# (the convention of common GIS cost-distance tools). Dijkstra's algorithm
# on this graph gives
#
#   cost_distance     the least accumulated cost from the nearest source
#                     to every cell, and the index of that source
#   least_cost_path   the cells of the cheapest path between two cells
#
# Cells with NaN, infinite or negative cost are barriers. With a uniform
# cost of 1 and 8-connectivity, the accumulated cost is the octile
# distance (diagonal steps of sqrt(2) cell sizes).
#
# References
# ----------
# Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
#   Numerische Mathematik, 1(1), 269-271.
# Douglas, D. H. (1994). Least-cost path in GIS using an accumulated cost
#   surface and slopelines. Cartographica, 31(3), 37-51.
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

# Sparse graph.
from scipy.sparse import csr_matrix

# Shortest paths on sparse graphs.
from scipy.sparse.csgraph import dijkstra


# Sparse graph of a cost raster.
def _cost_graph(
    cost: NDArray[Any],  # Friction raster.
    resolution: float,  # Cell size.
    connectivity: int,  # 4 or 8.
) -> tuple[csr_matrix, NDArray[np.bool_]]:  # Graph and passable-cell mask.
    # Friction as float64.
    c = np.asarray(cost, dtype=np.float64)
    # Only 2-D rasters are supported.
    if c.ndim != 2:
        # Report the wrong shape.
        raise ValueError(f"cost must be a 2-D raster, got shape {c.shape}")
    # Connectivity must be 4 or 8.
    if connectivity not in (4, 8):
        # Report the invalid value.
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    # The cell size must be positive.
    if resolution <= 0:
        # Report the invalid size.
        raise ValueError(f"resolution must be positive, got {resolution}")
    # Passable cells.
    ok = np.isfinite(c) & (c >= 0)
    # Grid size.
    rows, cols = c.shape
    # Flat index of every cell.
    index = np.arange(c.size).reshape(c.shape)
    # Neighbour offsets; each undirected pair once, both directions are added below.
    offsets = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if connectivity == 8 else [])
    # Edge lists.
    src, dst, weight = [], [], []
    # Visit every offset.
    for dr, dc in offsets:
        # Source window.
        r0, r1 = 0, rows - dr
        # Column window of the sources.
        c0, c1 = max(0, -dc), cols - max(0, dc)
        # Source cells.
        a = (slice(r0, r1), slice(c0, c1))
        # Neighbour cells.
        b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        # Both ends passable.
        both = ok[a] & ok[b]
        # Step length.
        step = resolution * float(np.hypot(dr, dc))
        # Mean friction times the step; tiny positive for zero-cost moves.
        w = np.maximum((c[a] + c[b]) / 2.0 * step, np.finfo(float).tiny)[both]
        # Forward and backward edges.
        src += [index[a][both], index[b][both]]
        # Destinations.
        dst += [index[b][both], index[a][both]]
        # Weights of both directions.
        weight += [w, w]
    # Sparse adjacency matrix.
    graph = csr_matrix(
        (np.concatenate(weight), (np.concatenate(src), np.concatenate(dst))),  # Weights, ends.
        shape=(c.size, c.size),  # One node per cell.
    )  # End of the matrix.
    # Return the graph and the mask.
    return graph, ok


# Accumulated least cost from the nearest source to every cell.
def cost_distance(
    cost: NDArray[Any],  # Friction raster (cost per unit distance).
    sources: list[tuple[int, int]] | NDArray[Any],  # (row, col) of the sources, or a mask.
    resolution: float = 1.0,  # Cell size.
    connectivity: int = 8,  # 4 or 8 neighbours.
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:  # Cost surface and nearest-source index.
    # Graph of the raster.
    graph, ok = _cost_graph(cost, resolution, connectivity)
    # Grid shape.
    shape = ok.shape
    # Source cells from a boolean mask.
    if isinstance(sources, np.ndarray) and sources.dtype == bool:
        # Positions of the mask.
        cells = [tuple(int(v) for v in rc) for rc in np.argwhere(sources)]
    # Source cells from a list of positions.
    else:
        # Positions as integer tuples.
        cells = [(int(r), int(c)) for r, c in sources]
    # At least one source.
    if not cells:
        # Report the missing sources.
        raise ValueError("at least one source cell is needed")
    # Sources must be passable cells of the grid.
    for r, c in cells:
        # Inside and passable.
        if not (0 <= r < shape[0] and 0 <= c < shape[1]) or not ok[r, c]:
            # Report the invalid source.
            raise ValueError(f"source {(r, c)} is outside the grid or a barrier")
    # Flat indices of the sources.
    flat = [r * shape[1] + c for r, c in cells]
    # Multi-source Dijkstra.
    dist, _, nearest = dijkstra(
        graph,  # Cell graph.
        directed=True,  # Edges are stored in both directions.
        indices=flat,  # Sources.
        min_only=True,  # Keep the nearest source only.
        return_predecessors=True,  # Also return the nearest source per cell.
    )  # End of the search.
    # Index of the nearest source in the input order, -1 when unreachable.
    order = {f: i for i, f in enumerate(flat)}
    # Map flat source ids to input positions.
    source_index = np.array([order.get(int(s), -1) for s in nearest], dtype=np.int64)
    # Back to the grid.
    return dist.reshape(shape), source_index.reshape(shape)


# Cheapest path between two cells.
def least_cost_path(
    cost: NDArray[Any],  # Friction raster.
    start: tuple[int, int],  # (row, col) of the start.
    end: tuple[int, int],  # (row, col) of the end.
    resolution: float = 1.0,  # Cell size.
    connectivity: int = 8,  # 4 or 8 neighbours.
) -> tuple[list[tuple[int, int]], float]:  # Cells from start to end and the total cost.
    # Graph of the raster.
    graph, ok = _cost_graph(cost, resolution, connectivity)
    # Grid width.
    cols = ok.shape[1]
    # Both ends must be passable cells.
    for r, c in (start, end):
        # Inside and passable.
        if not (0 <= r < ok.shape[0] and 0 <= c < cols) or not ok[r, c]:
            # Report the invalid cell.
            raise ValueError(f"cell {(r, c)} is outside the grid or a barrier")
    # Flat indices.
    a, b = start[0] * cols + start[1], end[0] * cols + end[1]
    # Single-source Dijkstra with predecessors.
    dist, pred = dijkstra(graph, directed=True, indices=a, return_predecessors=True)
    # Unreachable end.
    if not np.isfinite(dist[b]):
        # Report the disconnected cells.
        raise ValueError(f"no path from {start} to {end}: barriers separate them")
    # Walk back from the end.
    path = [b]
    # Follow the predecessors to the start.
    while path[-1] != a:
        # Previous cell.
        path.append(int(pred[path[-1]]))
    # Start first, as (row, col).
    cells = [divmod(int(p), cols) for p in reversed(path)]
    # Cells and total cost.
    return cells, float(dist[b])


# =============================================================================
# End of module src/unbihexium/analysis/network/cost_surface.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
