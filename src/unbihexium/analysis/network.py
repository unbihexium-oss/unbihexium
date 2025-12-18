"""Network analysis for routing and accessibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Route:
    """A route through a network."""

    nodes: list[int]
    distance: float
    time: float | None = None
    geometry: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class AccessibilityResult:
    """Result of accessibility analysis."""

    travel_times: NDArray[np.floating[Any]]
    origin: tuple[float, float]
    threshold: float


class NetworkAnalyzer:
    """Network analysis for routing and accessibility.

    Uses Dijkstra's algorithm for shortest path calculations.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, tuple[float, float]] = {}
        self._edges: list[tuple[int, int, float]] = []
        self._adj: dict[int, list[tuple[int, float]]] = {}

    def add_node(self, node_id: int, x: float, y: float) -> None:
        """Add a node to the network."""
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
        """Add an edge to the network."""
        self._edges.append((from_node, to_node, cost))
        self._adj.setdefault(from_node, []).append((to_node, cost))
        if bidirectional:
            self._adj.setdefault(to_node, []).append((from_node, cost))

    def shortest_path(self, origin: int, destination: int) -> Route | None:
        """Find shortest path between two nodes using Dijkstra."""
        import heapq

        if origin not in self._nodes or destination not in self._nodes:
            return None

        distances: dict[int, float] = {origin: 0}
        previous: dict[int, int | None] = {origin: None}
        pq: list[tuple[float, int]] = [(0, origin)]
        visited: set[int] = set()

        while pq:
            dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == destination:
                break

            for neighbor, cost in self._adj.get(current, []):
                if neighbor in visited:
                    continue
                new_dist = dist + cost
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        if destination not in distances:
            return None

        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous.get(current)
        path.reverse()

        geometry = [self._nodes[n] for n in path]

        return Route(
            nodes=path,
            distance=distances[destination],
            geometry=geometry,
        )

    def service_area(
        self,
        origin: int,
        max_cost: float,
    ) -> list[int]:
        """Find all nodes reachable within a cost threshold."""
        import heapq

        if origin not in self._nodes:
            return []

        distances: dict[int, float] = {origin: 0}
        pq: list[tuple[float, int]] = [(0, origin)]
        visited: set[int] = set()

        while pq:
            dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            for neighbor, cost in self._adj.get(current, []):
                if neighbor in visited:
                    continue
                new_dist = dist + cost
                if new_dist <= max_cost:
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

        return list(distances.keys())

    def closest_facility(
        self,
        origin: int,
        facilities: list[int],
    ) -> tuple[int, float] | None:
        """Find the closest facility to an origin."""
        best_facility = None
        best_distance = float("inf")

        for facility in facilities:
            route = self.shortest_path(origin, facility)
            if route and route.distance < best_distance:
                best_facility = facility
                best_distance = route.distance

        if best_facility is None:
            return None
        return (best_facility, best_distance)
