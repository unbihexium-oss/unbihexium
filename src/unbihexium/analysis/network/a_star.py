# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A* pathfinding algorithm implementation.

This module provides A* algorithm with multiple heuristic options for
network routing analysis.

A* improves upon Dijkstra by incorporating a heuristic estimate of
the remaining distance to the goal, reducing the search space.

Algorithm:
    f(n) = g(n) + h(n)
    where:
        g(n) = cost from start to node n
        h(n) = heuristic estimate from n to goal
        f(n) = total estimated cost through node n

Heuristics:
    - Euclidean: sqrt((x2-x1)^2 + (y2-y1)^2)
    - Manhattan: |x2-x1| + |y2-y1|
    - Haversine: great-circle distance for geographic coordinates
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np


class Heuristic(Enum):
    """Available heuristic functions for A*."""

    NONE = "none"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HAVERSINE = "haversine"


@dataclass
class AStarResult:
    """Result of A* pathfinding.

    Attributes:
        path: List of node IDs from start to goal.
        cost: Total path cost.
        nodes_explored: Number of nodes explored during search.
        success: Whether a path was found.
    """

    path: list[int]
    cost: float
    nodes_explored: int
    success: bool


def euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute Euclidean distance between two points.

    Formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def manhattan_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute Manhattan distance between two points.

    Formula: d = |x2-x1| + |y2-y1|
    """
    return abs(x2 - x1) + abs(y2 - y1)


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    earth_radius: float = 6371.0,
) -> float:
    """Compute great-circle distance using Haversine formula.

    For geographic coordinates (latitude/longitude in degrees).

    Formula:
        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c

    Args:
        lat1, lon1: First point (latitude, longitude in degrees).
        lat2, lon2: Second point (latitude, longitude in degrees).
        earth_radius: Earth radius in km (default 6371).

    Returns:
        Distance in same units as earth_radius.
    """
    # Convert to radians
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def get_heuristic_function(
    heuristic: Heuristic | str,
) -> Callable[[float, float, float, float], float]:
    """Get heuristic function by name.

    Args:
        heuristic: Heuristic type or string name.

    Returns:
        Heuristic function.
    """
    if isinstance(heuristic, str):
        heuristic = Heuristic(heuristic.lower())

    if heuristic == Heuristic.NONE:
        return lambda x1, y1, x2, y2: 0.0
    elif heuristic == Heuristic.EUCLIDEAN:
        return euclidean_distance
    elif heuristic == Heuristic.MANHATTAN:
        return manhattan_distance
    elif heuristic == Heuristic.HAVERSINE:
        return haversine_distance
    else:
        return euclidean_distance


def astar(
    nodes: dict[int, tuple[float, float]],
    adj: dict[int, list[tuple[int, float]]],
    start: int,
    goal: int,
    heuristic: Heuristic | str = Heuristic.EUCLIDEAN,
) -> AStarResult:
    """A* pathfinding algorithm.

    Args:
        nodes: Dictionary mapping node ID to (x, y) coordinates.
        adj: Adjacency list mapping node ID to list of (neighbor, cost).
        start: Starting node ID.
        goal: Goal node ID.
        heuristic: Heuristic function to use.

    Returns:
        AStarResult with path, cost, and statistics.
    """
    if start not in nodes or goal not in nodes:
        return AStarResult(path=[], cost=float("inf"), nodes_explored=0, success=False)

    if start == goal:
        return AStarResult(path=[start], cost=0.0, nodes_explored=1, success=True)

    h_func = get_heuristic_function(heuristic)
    goal_x, goal_y = nodes[goal]

    # Priority queue: (f_score, counter, node_id)
    # Counter breaks ties for determinism
    counter = 0
    open_set: list[tuple[float, int, int]] = []

    # g_score: cost from start to node
    g_score: dict[int, float] = {start: 0.0}

    # f_score: g_score + heuristic
    start_x, start_y = nodes[start]
    f_score: dict[int, float] = {start: h_func(start_x, start_y, goal_x, goal_y)}

    heapq.heappush(open_set, (f_score[start], counter, start))
    counter += 1

    came_from: dict[int, int] = {}
    closed_set: set[int] = set()
    nodes_explored = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        nodes_explored += 1
        closed_set.add(current)

        # Goal reached
        if current == goal:
            path = _reconstruct_path(came_from, current)
            return AStarResult(
                path=path,
                cost=g_score[current],
                nodes_explored=nodes_explored,
                success=True,
            )

        # Explore neighbors
        for neighbor, edge_cost in adj.get(current, []):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + edge_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                neighbor_x, neighbor_y = nodes[neighbor]
                h = h_func(neighbor_x, neighbor_y, goal_x, goal_y)
                f_score[neighbor] = tentative_g + h

                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                counter += 1

    # No path found
    return AStarResult(path=[], cost=float("inf"), nodes_explored=nodes_explored, success=False)


def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    """Reconstruct path from came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


class AStarPathfinder:
    """A* pathfinder with network graph support.

    This class provides a high-level interface for A* pathfinding
    that is compatible with NetworkAnalyzer.
    """

    def __init__(
        self,
        default_heuristic: Heuristic | str = Heuristic.EUCLIDEAN,
    ) -> None:
        """Initialize pathfinder.

        Args:
            default_heuristic: Default heuristic to use.
        """
        self._nodes: dict[int, tuple[float, float]] = {}
        self._adj: dict[int, list[tuple[int, float]]] = {}
        self.default_heuristic = (
            Heuristic(default_heuristic)
            if isinstance(default_heuristic, str)
            else default_heuristic
        )

    def add_node(self, node_id: int, x: float, y: float) -> None:
        """Add a node to the graph."""
        self._nodes[node_id] = (x, y)
        if node_id not in self._adj:
            self._adj[node_id] = []

    def add_edge(
        self,
        from_node: int,
        to_node: int,
        cost: float,
        bidirectional: bool = True,
    ) -> None:
        """Add an edge to the graph."""
        self._adj.setdefault(from_node, []).append((to_node, cost))
        if bidirectional:
            self._adj.setdefault(to_node, []).append((from_node, cost))

    def find_path(
        self,
        start: int,
        goal: int,
        heuristic: Heuristic | str | None = None,
    ) -> AStarResult:
        """Find shortest path using A*.

        Args:
            start: Starting node ID.
            goal: Goal node ID.
            heuristic: Heuristic to use (defaults to instance default).

        Returns:
            AStarResult with path and cost.
        """
        h = heuristic if heuristic is not None else self.default_heuristic
        return astar(self._nodes, self._adj, start, goal, h)
