# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_analysis.py
# Title       : Tests of routing, suitability analysis and zonal statistics
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, NumPy and SciPy;
#               the polygon zone test needs rasterio
# =============================================================================
#
# Abstract
# --------
# Checks unbihexium.analysis on small hand-solved cases: Dijkstra and A*
# paths of a four-node network, service areas, the closest facility, the
# origin-destination matrix, distance heuristics, raster cost distance
# (octile and Manhattan distances, barriers, nearest source), AHP weights of
# a consistent matrix and the consistency ratio, factor standardisation,
# weighted overlay with constraints, and zonal statistics including
# majority, minority, percentiles, nodata and polygon zones.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Square roots.
import math

# Arrays.
import numpy as np

# Test framework.
import pytest

# Routing under test.
from unbihexium.analysis.network import (
    AccessibilityResult,  # Costs from one origin.
    AStarPathfinder,  # A* graph.
    NetworkAnalyzer,  # Graph routing.
    Route,  # Path record.
    cost_distance,  # Raster cost distance.
    euclidean_distance,  # Planar distance.
    haversine_distance,  # Great-circle distance.
    least_cost_path,  # Raster path.
    manhattan_distance,  # Grid distance.
)  # End of the routing imports.

# Suitability analysis under test.
from unbihexium.analysis.suitability import (
    AHP,  # Analytic Hierarchy Process.
    WeightedOverlay,  # Weighted linear combination.
    fuzzy_membership,  # Fuzzy standardisation.
    reclassify,  # Class scores.
    rescale_linear,  # Linear standardisation.
    weighted_overlay,  # Overlay of rasters.
)  # End of the suitability imports.

# Zonal statistics under test.
from unbihexium.analysis.zonal import ZonalStatistics, zonal_statistics


# Four-node network: 1 -1- 2 -1- 3, 2 -1.5- 4 -0.5- 3.
@pytest.fixture
def network() -> NetworkAnalyzer:
    # Empty network.
    net = NetworkAnalyzer()
    # Nodes with coordinates.
    for node, (x, y) in {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (1, 1)}.items():
        # Add the node.
        net.add_node(node, x, y)
    # Edges with costs.
    for a, b, cost in [(1, 2, 1.0), (2, 3, 1.0), (2, 4, 1.5), (4, 3, 0.5)]:
        # Add the undirected edge.
        net.add_edge(a, b, cost)
    # Return the network.
    return net


# Route records keep their fields.
def test_route_record() -> None:
    # Route with time and geometry.
    route = Route(nodes=[1, 2, 3], distance=10.5, time=5.0, geometry=[(0.0, 0.0), (1.0, 1.0)])
    # Fields.
    assert route.nodes == [1, 2, 3] and route.distance == 10.5 and route.time == 5.0
    # Geometry.
    assert len(route.geometry) == 2


# Dijkstra finds the optimal path 1-2-3 of cost 2 (the detour 1-2-4-3 costs 3).
def test_shortest_path(network: NetworkAnalyzer) -> None:
    # Direct neighbour.
    direct = network.shortest_path(1, 2)
    # Path and cost.
    assert direct is not None and direct.nodes == [1, 2] and direct.distance == pytest.approx(1.0)
    # Two hops.
    route = network.shortest_path(1, 3)
    # Optimal path and its coordinates.
    assert route is not None and route.nodes == [1, 2, 3] and route.distance == pytest.approx(2.0)
    # Geometry follows the nodes.
    assert route.geometry == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    # A* with the Euclidean heuristic finds the same path.
    astar_route = network.shortest_path(1, 3, heuristic="euclidean")
    # Same result.
    assert astar_route is not None and astar_route.nodes == [1, 2, 3]
    # From 4 to 1 the path goes through 2 (cost 2.5).
    back = network.shortest_path(4, 1)
    # Path and cost.
    assert back is not None and back.nodes == [4, 2, 1] and back.distance == pytest.approx(2.5)
    # Same node.
    same = network.shortest_path(1, 1)
    # Zero cost.
    assert same is not None and same.distance == 0.0 and same.nodes == [1]
    # Unknown node.
    assert network.shortest_path(1, 999) is None


# Disconnected nodes have no route; invalid edges are rejected.
def test_disconnected_and_invalid(network: NetworkAnalyzer) -> None:
    # An isolated node.
    network.add_node(5, 10.0, 10.0)
    # No route to it with either algorithm.
    assert network.shortest_path(1, 5) is None and network.shortest_path(1, 5, "euclidean") is None
    # Negative costs break Dijkstra.
    with pytest.raises(ValueError):
        # Negative edge.
        network.add_edge(1, 5, -1.0)
    # Edges need existing nodes.
    with pytest.raises(ValueError):
        # Unknown node 6.
        network.add_edge(1, 6, 1.0)
    # Only valid edges were recorded.
    assert len(network._edges) == 4


# Service areas, accessibility and the closest facility.
def test_service_area_and_facilities(network: NetworkAnalyzer) -> None:
    # Nodes within cost 1.5: 1 (0) and 2 (1).
    assert network.service_area(1, max_cost=1.5) == [1, 2]
    # All four within 10, in cost order 1, 2, 3, 4.
    assert network.service_area(1, max_cost=10.0) == [1, 2, 3, 4]
    # Costs to all nodes.
    access = network.accessibility(1, threshold=2.0)
    # Record type and values.
    assert isinstance(access, AccessibilityResult)
    # Costs 0, 1, 2 and 2.5.
    assert access.travel_times.tolist() == [0.0, 1.0, 2.0, 2.5]
    # Nodes within the threshold.
    assert access.reachable == [1, 2, 3] and access.origin == (0.0, 0.0)
    # Node 3 (cost 2) is closer than node 4 (cost 2.5).
    assert network.closest_facility(1, facilities=[3, 4]) == (3, 2.0)
    # No facility, no result.
    assert network.closest_facility(1, facilities=[]) is None


# Origin-destination matrix and snapping.
def test_od_matrix_and_nearest_node(network: NetworkAnalyzer) -> None:
    # Costs from 1 and 4 to all nodes.
    matrix = network.od_cost_matrix([1, 4], [1, 2, 3, 4])
    # Hand-computed costs.
    assert np.allclose(matrix, [[0.0, 1.0, 2.0, 2.5], [2.5, 1.5, 0.5, 0.0]])
    # Point (0.9, 0.8) is closest to node 4.
    assert network.nearest_node(0.9, 0.8) == 4


# Distance heuristics and the A* pathfinder.
def test_heuristics_and_astar() -> None:
    # 3-4-5 triangle.
    assert euclidean_distance(0, 0, 3, 4) == pytest.approx(5.0)
    # Grid distance.
    assert manhattan_distance(0, 0, 3, 4) == pytest.approx(7.0)
    # One degree of latitude is R pi / 180 km.
    assert haversine_distance(60.0, 25.0, 61.0, 25.0) == pytest.approx(6371.0088 * math.pi / 180)
    # Grid graph where A* must go around a detour.
    finder = AStarPathfinder()
    # Nodes of a 3 x 3 grid.
    for i in range(9):
        # Row-major positions.
        finder.add_node(i, i % 3, i // 3)
    # Horizontal and vertical unit edges, except the ones into the centre.
    for i in range(9):
        # Right and down neighbours.
        for j in (i + 1 if i % 3 < 2 else None, i + 3 if i < 6 else None):
            # Skip missing neighbours and the centre node 4.
            if j is not None and 4 not in (i, j):
                # Unit cost.
                finder.add_edge(i, j, 1.0)
    # Corner to corner around the blocked centre: four unit steps.
    result = finder.find_path(0, 8)
    # Success, cost and path length.
    assert result.success and result.cost == pytest.approx(4.0) and len(result.path) == 5
    # The centre is unreachable.
    assert not finder.find_path(0, 4).success


# Raster cost distance equals octile and Manhattan distances on uniform cost.
def test_cost_distance() -> None:
    # Uniform friction.
    cost = np.ones((5, 5))
    # 8-connected distances from the corner.
    dist, nearest = cost_distance(cost, [(0, 0)])
    # Octile distance to (3, 4): 3 diagonal steps and 1 straight step.
    assert dist[3, 4] == pytest.approx(3 * math.sqrt(2) + 1)
    # 4-connected distances are Manhattan distances, scaled by the cell size.
    dist4, _ = cost_distance(cost, [(0, 0)], resolution=10.0, connectivity=4)
    # Manhattan distance 7 cells of 10 units.
    assert dist4[3, 4] == pytest.approx(70.0)
    # Two sources: each cell knows its nearest source.
    _, nearest = cost_distance(cost, [(0, 0), (4, 4)])
    # Corners belong to their own source.
    assert nearest[0, 1] == 0 and nearest[4, 3] == 1
    # Friction doubles the cost: mean friction (1 + 3) / 2 over one step.
    cost[0, 1] = 3.0
    # Distance to the neighbour.
    assert cost_distance(cost, [(0, 0)])[0][0, 1] == pytest.approx(2.0)


# Least-cost path around a barrier.
def test_least_cost_path() -> None:
    # Uniform friction.
    cost = np.ones((5, 5))
    # Wall across row 2 with a gap at the right end.
    cost[2, :4] = np.nan
    # Path from the top-left to the bottom-left corner.
    cells, total = least_cost_path(cost, (0, 0), (4, 0))
    # It passes through the gap.
    assert (2, 4) in cells and cells[0] == (0, 0) and cells[-1] == (4, 0)
    # 4 straight and 4 diagonal steps.
    assert total == pytest.approx(4 + 4 * math.sqrt(2))
    # A closed wall disconnects the halves.
    cost[2, 4] = np.nan
    # No path.
    with pytest.raises(ValueError):
        # Separated cells.
        least_cost_path(cost, (0, 0), (4, 0))


# Zonal statistics as a table per statistic.
def test_zonal_table() -> None:
    # Values 1 .. 9.
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    # One zone per row.
    zones = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int32)
    # Several statistics and a percentile.
    names = ["mean", "sum", "min", "max", "range"]
    # Compute the table.
    result = ZonalStatistics().calculate(values, zones, stats=names, percentiles=[25])
    # Means of the rows.
    assert result["mean"] == {1: 2.0, 2: 5.0, 3: 8.0}
    # Sums 1 + 2 + 3, 4 + 5 + 6, 7 + 8 + 9.
    assert result["sum"] == {1: 6.0, 2: 15.0, 3: 24.0}
    # Extremes and range.
    assert result["min"][3] == 7.0 and result["max"][3] == 9.0 and result["range"][2] == 2.0
    # 25th percentile with linear interpolation: 1 + 0.5 (2 - 1).
    assert result["p25"][1] == pytest.approx(1.5)
    # Unknown statistics are rejected.
    with pytest.raises(ValueError):
        # Not a statistic.
        ZonalStatistics().calculate(values, zones, stats=["mode"])


# Zonal records with nodata, majority and minority.
def test_zonal_records() -> None:
    # Two zones; zone 1 holds 1, 1, 2 and a NaN; zone 2 holds 3, 3, 3, 5 and a -9999.
    values = np.array([[1, 1, 2, np.nan, -9999], [3, 3, 3, 5, 7]])
    # Zones; the last cell of row 2 has no zone (NaN).
    zones = np.array([[1, 1, 1, 1, 2], [2, 2, 2, 2, np.nan]])
    # Statistics with -9999 as nodata.
    first, second = zonal_statistics(values, zones, percentiles=[50], nodata=-9999)
    # Zone 1: three valid cells.
    assert first.zone_id == 1 and first.count == 3 and first.sum == pytest.approx(4.0)
    # Population standard deviation of 1, 1, 2.
    assert first.std == pytest.approx(np.std([1, 1, 2]))
    # Most and least frequent values.
    assert first.majority == 1.0 and first.minority == 2.0 and first.variety == 2
    # Zone 2: 3, 3, 3, 5 (nodata and the zone-less 7 are ignored).
    assert second.count == 4 and second.mean == pytest.approx(3.5) and second.median == 3.0
    # Range and percentiles.
    assert second.range == 2.0 and second.percentiles == {50.0: 3.0}


# Polygon zones are burnt into the raster grid.
def test_zonal_polygons() -> None:
    # rasterio is optional.
    pytest.importorskip("rasterio")
    # Raster class of the library.
    from unbihexium.core.raster import Raster

    # 2 x 4 raster with 10 m cells.
    raster = Raster.from_array(
        np.array([[1, 1, 2, 9], [3, 3, 3, 5]], dtype=np.float32),  # Values.
        crs="EPSG:32635",  # UTM zone 35N.
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0),  # North-up grid.
    )  # End of the raster.
    # Corners of a 20 m square over the first two columns of both rows.
    ring = [(500000, 7000000), (500020, 7000000), (500020, 6999980), (500000, 6999980)]
    # Close the ring.
    ring.append(ring[0])
    # GeoJSON polygon.
    square = {
        "type": "Polygon",  # GeoJSON geometry type.
        "coordinates": [ring],  # Exterior ring.
    }  # End of the polygon.
    # One record for the polygon.
    (record,) = zonal_statistics(raster, [square])
    # Cells 1, 1, 3, 3.
    assert record.zone_id == 1 and record.count == 4 and record.mean == pytest.approx(2.0)


# AHP weights of a consistent matrix and the consistency ratio.
def test_ahp() -> None:
    # Consistent judgements a_ij = w_i / w_j of the weights 0.5, 0.3, 0.2.
    w = np.array([0.5, 0.3, 0.2])
    # Model for three criteria.
    ahp = AHP(criteria=["cost", "quality", "time"])
    # Set the matrix.
    ahp.set_comparison_matrix(w[:, None] / w[None, :])
    # The weights are recovered exactly.
    assert np.allclose(ahp.calculate_weights(), w)
    # A consistent matrix has lambda_max = n and CR = 0.
    assert ahp.lambda_max() == pytest.approx(3.0)
    # Zero consistency ratio.
    assert ahp.consistency_ratio() == pytest.approx(0.0, abs=1e-12)
    # Named weights.
    assert ahp.weights_dict()["cost"] == pytest.approx(0.5)
    # The geometric mean method agrees on consistent matrices.
    geo = AHP(method="geometric_mean").fit(w[:, None] / w[None, :])
    # Same weights.
    assert np.allclose(geo.calculate_weights(), w)
    # A slightly inconsistent matrix.
    matrix = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    # Model of it.
    ahp = AHP(["A", "B", "C"]).set_comparison_matrix(matrix)
    # Largest eigenvalue of the matrix.
    lam = max(np.linalg.eigvals(matrix).real)
    # CR = (lambda_max - 3) / 2 / 0.58.
    assert ahp.consistency_ratio() == pytest.approx((lam - 3) / 2 / 0.58)
    # Acceptable consistency, positive weights summing to one.
    assert ahp.is_consistent() and np.all(ahp.calculate_weights() > 0)
    # Weights sum to one.
    assert ahp.calculate_weights().sum() == pytest.approx(1.0)
    # Judgements by name build the same matrix.
    named = AHP.from_judgements(["A", "B", "C"], {("A", "B"): 3, ("A", "C"): 5, ("B", "C"): 3})
    # Same weights.
    assert np.allclose(named.calculate_weights(), ahp.calculate_weights())
    # Non-reciprocal matrices are rejected.
    with pytest.raises(ValueError):
        # a_12 a_21 = 4.
        AHP().set_comparison_matrix(np.array([[1, 2], [2, 1]]))


# Standardisation of factors.
def test_standardisation() -> None:
    # Values 0, 5, 10, 20.
    x = np.array([0.0, 5.0, 10.0, 20.0])
    # Linear rescaling between 0 and 10, clipped.
    assert np.allclose(rescale_linear(x, 0, 10), [0, 0.5, 1, 1])
    # Decreasing factor.
    assert np.allclose(rescale_linear(x, 0, 10, increasing=False), [1, 0.5, 0, 0])
    # Sigmoidal membership is 0.5 half-way and 0 or 1 at the control points.
    assert np.allclose(fuzzy_membership(x, 0, 10), [0, 0.5, 1, 1])
    # Decreasing sigmoidal membership (a > b).
    assert np.allclose(fuzzy_membership(np.array([2.5]), 10, 0), [np.cos(np.pi / 8) ** 2])
    # Reclassification: < 5 -> 1, [5, 15) -> 2, >= 15 -> 3.
    assert np.allclose(reclassify(x, [5, 15], [1, 2, 3]), [1, 2, 2, 3])


# Weighted overlay with normalised weights and constraints.
def test_weighted_overlay() -> None:
    # Two layers.
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Second layer.
    b = np.array([[4, 3], [2, 1]], dtype=np.float32)
    # Equal weights give the mean.
    assert np.allclose(WeightedOverlay().calculate([a, b], [0.5, 0.5]), 2.5)
    # 0.8 * 10 + 0.2 * 0 = 8.
    tens, zeros = np.full((2, 2), 10.0), np.zeros((2, 2))
    # Weighted sum.
    assert WeightedOverlay().calculate([tens, zeros], [0.8, 0.2])[0, 0] == pytest.approx(8.0)
    # Weights are normalised: [1, 1] behaves as [0.5, 0.5].
    ones = np.ones((1, 1))
    # Normalised weights.
    assert WeightedOverlay().calculate([ones, ones], [1.0, 1.0])[0, 0] == pytest.approx(1.0)
    # A constraint excludes a cell.
    mask = np.array([[True, False], [True, True]])
    # Excluded cells score zero.
    assert WeightedOverlay().calculate([a, b], [1, 1], constraints=[mask])[0, 1] == 0.0
    # Mismatched weights are rejected.
    with pytest.raises(ValueError):
        # Two layers, one weight.
        WeightedOverlay().calculate([a, b], [1.0])


# Weighted overlay of rasters keeps the georeferencing.
def test_weighted_overlay_rasters() -> None:
    # Raster class of the library.
    from unbihexium.core.raster import Raster

    # A georeferenced layer.
    layer = Raster.from_array(np.array([[0.0, 5.0], [10.0, 10.0]]), crs="EPSG:3067")
    # Normalised overlay of the layer with itself, weights 1 and 3.
    result = weighted_overlay([layer, layer], [1, 3])
    # Rescaled values 0, 0.5, 1, 1.
    assert np.allclose(result.suitability, [[0.0, 0.5], [1.0, 1.0]])
    # Normalised weights.
    assert result.weights == {"layer_0": 0.25, "layer_1": 0.75}
    # Output raster with the input CRS.
    assert result.raster is not None and result.raster.metadata.crs == "EPSG:3067"


# =============================================================================
# End of module tests/unit/test_analysis.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
