# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/network/graph.py
# Title       : Shortest paths, service areas and cost matrices on networks
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# NetworkAnalyzer holds a weighted graph with node coordinates (for example
# a road network with lengths or travel times as edge costs) and answers:
#
#   shortest_path      Dijkstra (1959), or A* with a distance heuristic
#   shortest_costs     costs from one node to every reachable node
#   service_area       nodes reachable within a cost budget
#   accessibility      costs to all nodes as an array, with a threshold
#   closest_facility   the cheapest facility from an origin
#   od_cost_matrix     costs between sets of origins and destinations,
#                      computed with scipy.sparse.csgraph
#   nearest_node       snaps a coordinate to the closest node
#
# Edge costs must be non-negative. Parallel edges are allowed; the cheapest
# one is used.
#
# References
# ----------
# Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
#   Numerische Mathematik, 1(1), 269-271.
# Hart, P. E., Nilsson, N. J., Raphael, B. (1968). A formal basis for the
#   heuristic determination of minimum cost paths. IEEE Transactions on
#   Systems Science and Cybernetics, 4(2), 100-107.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Priority queue.
import heapq

# Infinity and NaN checks.
import math

# Result records.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Sparse matrices for the cost matrix.
from scipy.sparse import csr_matrix

# All-pairs shortest paths on sparse graphs.
from scipy.sparse.csgraph import dijkstra

# A* search.
from unbihexium.analysis.network.a_star import Heuristic, astar


# A path through the network.
@dataclass
class Route:
    # Node ids from origin to destination.
    nodes: list[int]
    # Total cost of the path.
    distance: float
    # Travel time, when known.
    time: float | None = None
    # Node coordinates along the path.
    geometry: list[tuple[float, float]] = field(default_factory=list)


# Costs from one origin to every node.
@dataclass
class AccessibilityResult:
    # Cost to each node in `node_ids` order; infinity when unreachable.
    travel_times: NDArray[np.floating[Any]]
    # Coordinates of the origin node.
    origin: tuple[float, float]
    # Cost budget.
    threshold: float
    # Node ids in the order of `travel_times`.
    node_ids: list[int] = field(default_factory=list)

    # Node ids reachable within the budget.
    @property
    def reachable(self) -> list[int]:
        # Ids whose cost does not exceed the threshold.
        return [n for n, t in zip(self.node_ids, self.travel_times) if t <= self.threshold]


# Weighted graph with coordinates and routing queries.
class NetworkAnalyzer:
    # Create an empty network.
    def __init__(self) -> None:
        # Node coordinates by id.
        self._nodes: dict[int, tuple[float, float]] = {}
        # Edges as (from, to, cost), as added.
        self._edges: list[tuple[int, int, float]] = []
        # Adjacency lists of (neighbour, cost).
        self._adj: dict[int, list[tuple[int, float]]] = {}

    # Add a node with coordinates.
    def add_node(self, node_id: int, x: float, y: float) -> None:
        # Record the coordinates.
        self._nodes[node_id] = (float(x), float(y))
        # Create the adjacency list.
        self._adj.setdefault(node_id, [])

    # Add an edge between two existing nodes.
    def add_edge(
        self,  # The network.
        from_node: int,  # Tail node.
        to_node: int,  # Head node.
        cost: float,  # Non-negative cost.
        bidirectional: bool = True,  # Also add the reverse edge.
    ) -> None:  # Nothing is returned.
        # Both nodes must exist.
        if from_node not in self._nodes or to_node not in self._nodes:
            # Report the unknown node.
            raise ValueError(f"add nodes {from_node} and {to_node} before connecting them")
        # Costs must be non-negative numbers.
        if math.isnan(cost) or cost < 0:
            # Report the invalid cost.
            raise ValueError(f"edge cost must be non-negative, got {cost}")
        # Record the edge.
        self._edges.append((from_node, to_node, float(cost)))
        # Forward direction.
        self._adj[from_node].append((to_node, float(cost)))
        # Reverse direction.
        if bidirectional:
            # Same cost backwards.
            self._adj[to_node].append((from_node, float(cost)))

    # Number of nodes.
    @property
    def node_count(self) -> int:
        # Size of the node table.
        return len(self._nodes)

    # Dijkstra from one node: costs and predecessors, optionally limited by a budget.
    def _dijkstra(
        self,  # The instance.
        origin: int,  # Start node.
        max_cost: float = math.inf,  # Do not expand beyond this cost.
        target: int | None = None,  # Stop when this node is settled.
    ) -> tuple[dict[int, float], dict[int, int]]:  # Costs and predecessors.
        # Settled costs.
        cost: dict[int, float] = {}
        # Tentative costs.
        best: dict[int, float] = {origin: 0.0}
        # Predecessors.
        prev: dict[int, int] = {}
        # Queue of (cost, node).
        queue: list[tuple[float, int]] = [(0.0, origin)]
        # Settle nodes in cost order.
        while queue:
            # Cheapest open node.
            d, node = heapq.heappop(queue)
            # Stale entries.
            if node in cost:
                # Skip.
                continue
            # Settle the node.
            cost[node] = d
            # Stop at the target.
            if node == target:
                # Done.
                break
            # Relax the edges.
            for neighbour, weight in self._adj.get(node, []):
                # Cost through this node.
                nd = d + weight
                # Keep improvements within the budget.
                if nd <= max_cost and nd < best.get(neighbour, math.inf):
                    # Record the improvement.
                    best[neighbour], prev[neighbour] = nd, node
                    # Queue the neighbour.
                    heapq.heappush(queue, (nd, neighbour))
        # Return the settled costs and predecessors.
        return cost, prev

    # Costs from an origin to every node reachable within max_cost.
    def shortest_costs(self, origin: int, max_cost: float = math.inf) -> dict[int, float]:
        # Unknown origins reach nothing.
        if origin not in self._nodes:
            # Empty result.
            return {}
        # Settled costs.
        return self._dijkstra(origin, max_cost)[0]

    # Cheapest route between two nodes, or None when there is none.
    def shortest_path(
        self,  # The instance.
        origin: int,  # Start node.
        destination: int,  # End node.
        heuristic: Heuristic | str | None = None,  # A* heuristic; None runs Dijkstra.
    ) -> Route | None:  # Cheapest route, or None.
        # Unknown nodes have no route.
        if origin not in self._nodes or destination not in self._nodes:
            # No route.
            return None
        # A* when a heuristic is given.
        if heuristic is not None:
            # Search with the estimate.
            found = astar(self._nodes, self._adj, origin, destination, heuristic)
            # Convert to a route.
            if not found.success:
                # No route.
                return None
            # Route with coordinates.
            return Route(found.path, found.cost, geometry=[self._nodes[n] for n in found.path])
        # Dijkstra until the destination is settled.
        cost, prev = self._dijkstra(origin, target=destination)
        # Unreachable destination.
        if destination not in cost:
            # No route.
            return None
        # Walk back from the destination.
        path = [destination]
        # Follow the predecessors.
        while path[-1] != origin:
            # Previous node.
            path.append(prev[path[-1]])
        # Origin first.
        path.reverse()
        # Route with coordinates.
        return Route(path, cost[destination], geometry=[self._nodes[n] for n in path])

    # Nodes reachable within a cost budget, cheapest first.
    def service_area(self, origin: int, max_cost: float) -> list[int]:
        # Costs within the budget.
        cost = self.shortest_costs(origin, max_cost)
        # Sort by cost.
        return sorted(cost, key=cost.__getitem__)

    # Costs from an origin to all nodes as an array.
    def accessibility(self, origin: int, threshold: float = math.inf) -> AccessibilityResult:
        # The origin must exist.
        if origin not in self._nodes:
            # Report the unknown node.
            raise ValueError(f"unknown origin node {origin}")
        # Costs to all reachable nodes.
        cost = self.shortest_costs(origin)
        # Node ids in insertion order.
        ids = list(self._nodes)
        # Costs aligned with the ids.
        times = np.array([cost.get(n, math.inf) for n in ids])
        # Package the result.
        return AccessibilityResult(times, self._nodes[origin], float(threshold), ids)

    # Facility with the cheapest route from the origin, and its cost.
    def closest_facility(self, origin: int, facilities: list[int]) -> tuple[int, float] | None:
        # Costs from the origin.
        cost = self.shortest_costs(origin)
        # Reachable facilities.
        reachable = [(cost[f], f) for f in facilities if f in cost]
        # None reachable.
        if not reachable:
            # No facility.
            return None
        # Cheapest, lowest id first on ties.
        best_cost, best = min(reachable)
        # Facility and cost.
        return best, best_cost

    # Sparse adjacency matrix with the cheapest parallel edge.
    def _matrix(self) -> tuple[csr_matrix, dict[int, int]]:
        # Row index of every node id.
        index = {n: i for i, n in enumerate(self._nodes)}
        # Cheapest cost per directed pair.
        best: dict[tuple[int, int], float] = {}
        # Visit all adjacency entries.
        for a, edges in self._adj.items():
            # Visit the edges of the node.
            for b, w in edges:
                # Directed pair of indices.
                key = (index[a], index[b])
                # Keep the cheapest.
                best[key] = min(w, best.get(key, math.inf))
        # Node count.
        n = len(index)
        # csgraph treats explicit zeros as missing edges; use a tiny positive cost instead.
        data = [w if w > 0 else np.finfo(float).tiny for w in best.values()]
        # Row and column indices.
        rows, cols = zip(*best) if best else ((), ())
        # Sparse matrix.
        return csr_matrix((data, (rows, cols)), shape=(n, n)), index

    # Cost matrix between origins and destinations (infinity when unreachable).
    def od_cost_matrix(self, origins: list[int], destinations: list[int]) -> NDArray[np.float64]:
        # Sparse graph and index map.
        graph, index = self._matrix()
        # Unknown nodes are an error.
        missing = [n for n in list(origins) + list(destinations) if n not in index]
        # Report them.
        if missing:
            # Name the first unknown node.
            raise ValueError(f"unknown node ids: {missing}")
        # Costs from every origin to all nodes.
        full = dijkstra(graph, directed=True, indices=[index[o] for o in origins])
        # Keep the destination columns.
        return np.atleast_2d(full)[:, [index[d] for d in destinations]]

    # Node closest to a coordinate.
    def nearest_node(self, x: float, y: float) -> int:
        # The network must have nodes.
        if not self._nodes:
            # Report the empty network.
            raise ValueError("the network has no nodes")
        # Coordinates of all nodes.
        ids = list(self._nodes)
        # Squared distances.
        xy = np.array([self._nodes[n] for n in ids])
        # Closest node.
        return ids[int(np.argmin((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2))]


# =============================================================================
# End of module src/unbihexium/analysis/network/graph.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
