# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/network/a_star.py
# Title       : A* shortest paths with distance heuristics
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# A* search (Hart, Nilsson and Raphael, 1968) expands nodes in the order of
# f(n) = g(n) + h(n), the cost from the start plus a heuristic estimate of
# the remaining cost. With an admissible heuristic (never larger than the
# true remaining cost) the first path found to the goal is optimal; with
# h = 0 the search is Dijkstra's algorithm.
#
#   euclidean_distance   straight-line distance of planar coordinates
#   manhattan_distance   sum of coordinate differences (grid movement)
#   haversine_distance   great-circle distance of (latitude, longitude) in
#                        degrees, in the unit of the Earth radius (km)
#
# A heuristic is admissible only when it is expressed in the unit of the
# edge costs: Euclidean distance for edge lengths in map units, haversine
# distance in km for edge lengths in km. For travel times, divide by the
# largest speed of the network, or use Heuristic.NONE.
#
# References
# ----------
# Hart, P. E., Nilsson, N. J., Raphael, B. (1968). A formal basis for the
#   heuristic determination of minimum cost paths. IEEE Transactions on
#   Systems Science and Cybernetics, 4(2), 100-107.
# Sinnott, R. W. (1984). Virtues of the haversine. Sky and Telescope,
#   68(2), 159.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Priority queue.
import heapq

# Trigonometry.
import math

# Type of callables.
from collections.abc import Callable

# Result record.
from dataclasses import dataclass

# Heuristic names.
from enum import Enum

# Mean Earth radius in kilometres.
EARTH_RADIUS_KM = 6371.0088


# Heuristics available to A*.
class Heuristic(Enum):
    # No heuristic: Dijkstra's algorithm.
    NONE = "none"
    # Straight-line distance.
    EUCLIDEAN = "euclidean"
    # Sum of coordinate differences.
    MANHATTAN = "manhattan"
    # Great-circle distance of geographic coordinates.
    HAVERSINE = "haversine"


# Result of an A* search.
@dataclass
class AStarResult:
    # Node ids from start to goal; empty when no path exists.
    path: list[int]
    # Total path cost; infinity when no path exists.
    cost: float
    # Number of nodes removed from the queue.
    nodes_explored: int
    # Whether a path was found.
    success: bool


# Straight-line distance between (x1, y1) and (x2, y2).
def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    # Pythagoras.
    return math.hypot(x2 - x1, y2 - y1)


# Manhattan distance |x2 - x1| + |y2 - y1|.
def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    # Sum of absolute differences.
    return abs(x2 - x1) + abs(y2 - y1)


# Great-circle distance of two (latitude, longitude) points in degrees.
def haversine_distance(
    lat1: float,  # Latitude of the first point.
    lon1: float,  # Longitude of the first point.
    lat2: float,  # Latitude of the second point.
    lon2: float,  # Longitude of the second point.
    earth_radius: float = EARTH_RADIUS_KM,  # Sphere radius; the result has its unit.
) -> float:  # Distance along the sphere.
    # Latitudes in radians.
    p1, p2 = math.radians(lat1), math.radians(lat2)
    # Latitude difference.
    dlat = p2 - p1
    # Longitude difference.
    dlon = math.radians(lon2 - lon1)
    # Haversine of the central angle.
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    # Central angle, clipped against round-off above one.
    c = 2.0 * math.asin(math.sqrt(min(1.0, a)))
    # Arc length.
    return earth_radius * c


# Heuristic that is always zero.
def _zero(x1: float, y1: float, x2: float, y2: float) -> float:
    # No estimate.
    return 0.0


# Heuristic that reads node coordinates as (longitude, latitude).
def _haversine_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    # x is longitude and y latitude.
    return haversine_distance(y1, x1, y2, x2)


# Heuristic function of a name or enum member, taking (x1, y1, x2, y2).
def get_heuristic_function(
    heuristic: Heuristic | str,  # Heuristic name or member.
) -> Callable[[float, float, float, float], float]:  # Function of (x1, y1, x2, y2).
    # Names become enum members.
    kind = Heuristic(heuristic.lower()) if isinstance(heuristic, str) else heuristic
    # Table of functions.
    table = {
        Heuristic.NONE: _zero,  # Dijkstra.
        Heuristic.EUCLIDEAN: euclidean_distance,  # Planar distance.
        Heuristic.MANHATTAN: manhattan_distance,  # Grid distance.
        Heuristic.HAVERSINE: _haversine_xy,  # Node x = longitude, y = latitude.
    }  # End of the table.
    # Return the function.
    return table[kind]


# Rebuild a path from the predecessor map.
def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    # Start at the goal.
    path = [current]
    # Walk back to the start.
    while current in came_from:
        # Predecessor.
        current = came_from[current]
        # Record it.
        path.append(current)
    # Start first.
    path.reverse()
    # Return the path.
    return path


# A* search on an adjacency list.
def astar(
    nodes: dict[int, tuple[float, float]],  # Node id to (x, y).
    adj: dict[int, list[tuple[int, float]]],  # Node id to (neighbour, non-negative cost).
    start: int,  # Start node.
    goal: int,  # Goal node.
    heuristic: Heuristic | str = Heuristic.EUCLIDEAN,  # Estimate of the remaining cost.
) -> AStarResult:  # Path, cost and search statistics.
    # Unknown nodes have no path.
    if start not in nodes or goal not in nodes:
        # Failed search.
        return AStarResult(path=[], cost=math.inf, nodes_explored=0, success=False)
    # Heuristic function.
    h = get_heuristic_function(heuristic)
    # Goal coordinates.
    gx, gy = nodes[goal]
    # Best known cost from the start.
    g_score: dict[int, float] = {start: 0.0}
    # Predecessors on the best paths.
    came_from: dict[int, int] = {}
    # Queue of (f, insertion counter, node); the counter breaks ties deterministically.
    queue: list[tuple[float, int, int]] = [(h(*nodes[start], gx, gy), 0, start)]
    # Insertion counter.
    counter = 1
    # Nodes already expanded.
    closed: set[int] = set()
    # Expand nodes in order of f.
    while queue:
        # Node with the smallest f.
        _, _, current = heapq.heappop(queue)
        # Nodes may be queued several times.
        if current in closed:
            # Skip stale entries.
            continue
        # Expand the node.
        closed.add(current)
        # Goal reached.
        if current == goal:
            # Return the path.
            path = _reconstruct_path(came_from, current)
            # Path, cost, explored count and success flag.
            return AStarResult(path, g_score[current], len(closed), True)
        # Relax the edges.
        for neighbour, cost in adj.get(current, []):
            # Expanded nodes are final.
            if neighbour in closed:
                # Next edge.
                continue
            # Cost through the current node.
            tentative = g_score[current] + cost
            # Keep only improvements.
            if tentative < g_score.get(neighbour, math.inf):
                # Record the better path.
                g_score[neighbour], came_from[neighbour] = tentative, current
                # Queue with its estimate.
                estimate = tentative + h(*nodes[neighbour], gx, gy)
                # Push the neighbour with its estimate.
                heapq.heappush(queue, (estimate, counter, neighbour))
                # Next insertion number.
                counter += 1
    # The goal is unreachable.
    return AStarResult(path=[], cost=math.inf, nodes_explored=len(closed), success=False)


# A* on a graph built node by node.
class AStarPathfinder:
    # Create an empty graph.
    def __init__(self, default_heuristic: Heuristic | str = Heuristic.EUCLIDEAN) -> None:
        # Node coordinates.
        self._nodes: dict[int, tuple[float, float]] = {}
        # Adjacency lists.
        self._adj: dict[int, list[tuple[int, float]]] = {}
        # Heuristic used when none is given.
        self.default_heuristic = Heuristic(default_heuristic)

    # Add a node with coordinates.
    def add_node(self, node_id: int, x: float, y: float) -> None:
        # Record the coordinates.
        self._nodes[node_id] = (float(x), float(y))
        # Create its adjacency list.
        self._adj.setdefault(node_id, [])

    # Add an edge with a non-negative cost.
    def add_edge(
        self,  # The graph.
        from_node: int,  # Tail node.
        to_node: int,  # Head node.
        cost: float,  # Non-negative cost.
        bidirectional: bool = True,  # Also add the reverse edge.
    ) -> None:  # Nothing is returned.
        # A* needs non-negative costs.
        if cost < 0 or math.isnan(cost):
            # Report the invalid cost.
            raise ValueError(f"edge cost must be non-negative, got {cost}")
        # Forward edge.
        self._adj.setdefault(from_node, []).append((to_node, float(cost)))
        # Reverse edge.
        if bidirectional:
            # Same cost backwards.
            self._adj.setdefault(to_node, []).append((from_node, float(cost)))

    # Shortest path from start to goal.
    def find_path(
        self,  # The graph.
        start: int,  # Start node.
        goal: int,  # Goal node.
        heuristic: Heuristic | str | None = None,  # Heuristic; the default when None.
    ) -> AStarResult:  # Path, cost and search statistics.
        # Heuristic of the call or the default.
        h = heuristic if heuristic is not None else self.default_heuristic
        # Run A*.
        return astar(self._nodes, self._adj, start, goal, h)


# =============================================================================
# End of module src/unbihexium/analysis/network/a_star.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
