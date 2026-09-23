# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for analysis modules (network, zonal, suitability).

This module provides comprehensive tests for:
- Dijkstra shortest path algorithm
- A* pathfinding (when implemented)
- AHP multi-criteria decision analysis
- Weighted overlay suitability analysis
- Zonal statistics calculations

All tests use small synthetic fixtures for speed.
"""

from __future__ import annotations

import numpy as np
import pytest

from unbihexium.analysis.network import AccessibilityResult, NetworkAnalyzer, Route
from unbihexium.analysis.suitability import AHP, WeightedOverlay
from unbihexium.analysis.zonal import ZonalStatistics

# Network Analysis Tests


class TestRoute:
    """Tests for Route dataclass."""

    def test_route_creation(self) -> None:
        """Test route creation."""
        route = Route(nodes=[1, 2, 3], distance=10.5, time=5.0)
        assert len(route.nodes) == 3
        assert route.distance == 10.5
        assert route.time == 5.0

    def test_route_with_geometry(self) -> None:
        """Test route with geometry."""
        route = Route(
            nodes=[1, 2],
            distance=5.0,
            geometry=[(0.0, 0.0), (1.0, 1.0)],
        )
        assert len(route.geometry) == 2


class TestNetworkAnalyzer:
    """Tests for NetworkAnalyzer class."""

    @pytest.fixture
    def simple_network(self) -> NetworkAnalyzer:
        """Create a simple network for testing.

        Network topology:
            1 --1.0-- 2 --1.0-- 3
                      |
                     1.5
                      |
                      4 --0.5-- 3
        """
        network = NetworkAnalyzer()
        network.add_node(1, 0.0, 0.0)
        network.add_node(2, 1.0, 0.0)
        network.add_node(3, 2.0, 0.0)
        network.add_node(4, 1.0, 1.0)

        network.add_edge(1, 2, 1.0)
        network.add_edge(2, 3, 1.0)
        network.add_edge(2, 4, 1.5)
        network.add_edge(4, 3, 0.5)

        return network

    @pytest.fixture
    def disconnected_network(self) -> NetworkAnalyzer:
        """Create a network with disconnected components."""
        network = NetworkAnalyzer()
        network.add_node(1, 0.0, 0.0)
        network.add_node(2, 1.0, 0.0)
        network.add_node(3, 10.0, 10.0)  # Disconnected

        network.add_edge(1, 2, 1.0)
        # Node 3 has no edges

        return network

    def test_add_node(self) -> None:
        """Test adding nodes to network."""
        network = NetworkAnalyzer()
        network.add_node(1, 0.0, 0.0)
        network.add_node(2, 1.0, 1.0)
        assert 1 in network._nodes
        assert 2 in network._nodes

    def test_add_edge(self) -> None:
        """Test adding edges to network."""
        network = NetworkAnalyzer()
        network.add_node(1, 0.0, 0.0)
        network.add_node(2, 1.0, 0.0)
        network.add_edge(1, 2, 1.0)
        assert len(network._edges) == 1

    def test_shortest_path_direct(self, simple_network: NetworkAnalyzer) -> None:
        """Test shortest path on direct route."""
        route = simple_network.shortest_path(1, 2)
        assert route is not None
        assert route.nodes == [1, 2]
        assert route.distance == pytest.approx(1.0, rel=0.01)

    def test_shortest_path_multiple_hops(self, simple_network: NetworkAnalyzer) -> None:
        """Test shortest path with multiple hops."""
        route = simple_network.shortest_path(1, 3)
        assert route is not None
        assert route.nodes[0] == 1
        assert route.nodes[-1] == 3
        assert route.distance == pytest.approx(2.0, rel=0.01)

    def test_shortest_path_same_node(self, simple_network: NetworkAnalyzer) -> None:
        """Test shortest path to same node."""
        route = simple_network.shortest_path(1, 1)
        assert route is not None
        assert route.distance == pytest.approx(0.0, rel=0.01)

    def test_shortest_path_no_path(self, disconnected_network: NetworkAnalyzer) -> None:
        """Test shortest path when no path exists."""
        route = disconnected_network.shortest_path(1, 3)
        assert route is None

    def test_shortest_path_invalid_node(self, simple_network: NetworkAnalyzer) -> None:
        """Test shortest path with invalid node."""
        route = simple_network.shortest_path(1, 999)
        assert route is None

    def test_dijkstra_correctness(self, simple_network: NetworkAnalyzer) -> None:
        """Test Dijkstra produces known optimal path.

        From 1 to 3, direct path (1-2-3) has cost 2.0.
        Alternative path (1-2-4-3) has cost 1.0 + 1.5 + 0.5 = 3.0.
        Optimal should be 2.0.
        """
        route = simple_network.shortest_path(1, 3)
        assert route is not None
        assert route.distance == pytest.approx(2.0, rel=0.01)
        assert route.nodes == [1, 2, 3]

    def test_service_area(self, simple_network: NetworkAnalyzer) -> None:
        """Test service area calculation."""
        nodes = simple_network.service_area(1, max_cost=1.5)
        assert 1 in nodes
        assert 2 in nodes
        # Node 3 is 2.0 away, should not be included
        assert 3 not in nodes

    def test_service_area_all_reachable(self, simple_network: NetworkAnalyzer) -> None:
        """Test service area with large threshold."""
        nodes = simple_network.service_area(1, max_cost=10.0)
        assert len(nodes) == 4  # All nodes reachable

    def test_closest_facility(self, simple_network: NetworkAnalyzer) -> None:
        """Test closest facility calculation."""
        result = simple_network.closest_facility(1, facilities=[3, 4])
        assert result is not None
        facility, distance = result
        # Node 3 is distance 2.0, Node 4 is distance 2.5
        assert facility == 3
        assert distance == pytest.approx(2.0, rel=0.01)

    def test_closest_facility_no_facilities(self, simple_network: NetworkAnalyzer) -> None:
        """Test closest facility with empty list."""
        result = simple_network.closest_facility(1, facilities=[])
        assert result is None


# Zonal Statistics Tests


class TestZonalStatistics:
    """Tests for ZonalStatistics class."""

    @pytest.fixture
    def sample_raster(self) -> np.ndarray:
        """Create sample raster for testing."""
        return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    @pytest.fixture
    def sample_zones(self) -> np.ndarray:
        """Create sample zones for testing."""
        return np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int32)

    def test_initialization(self) -> None:
        """Test ZonalStatistics initialization."""
        zs = ZonalStatistics()
        assert zs is not None

    def test_calculate_mean(self, sample_raster: np.ndarray, sample_zones: np.ndarray) -> None:
        """Test mean calculation for zones."""
        zs = ZonalStatistics()
        result = zs.calculate(sample_raster, sample_zones, stats=["mean"])

        assert "mean" in result
        assert result["mean"][1] == pytest.approx(2.0, rel=0.01)
        assert result["mean"][2] == pytest.approx(5.0, rel=0.01)
        assert result["mean"][3] == pytest.approx(8.0, rel=0.01)

    def test_calculate_sum(self, sample_raster: np.ndarray, sample_zones: np.ndarray) -> None:
        """Test sum calculation for zones."""
        zs = ZonalStatistics()
        result = zs.calculate(sample_raster, sample_zones, stats=["sum"])

        assert "sum" in result
        assert result["sum"][1] == pytest.approx(6.0, rel=0.01)  # 1+2+3
        assert result["sum"][2] == pytest.approx(15.0, rel=0.01)  # 4+5+6
        assert result["sum"][3] == pytest.approx(24.0, rel=0.01)  # 7+8+9

    def test_calculate_multiple_stats(
        self, sample_raster: np.ndarray, sample_zones: np.ndarray
    ) -> None:
        """Test multiple statistics calculation."""
        zs = ZonalStatistics()
        result = zs.calculate(sample_raster, sample_zones, stats=["mean", "sum", "min", "max"])

        assert "mean" in result
        assert "sum" in result
        assert "min" in result
        assert "max" in result


# AHP Tests


class TestAHP:
    """Tests for Analytic Hierarchy Process."""

    def test_initialization(self) -> None:
        """Test AHP initialization."""
        criteria = ["cost", "quality", "time"]
        ahp = AHP(criteria=criteria)
        assert len(ahp.criteria) == 3

    def test_consistency_ratio_consistent(self) -> None:
        """Test consistency ratio for consistent matrix."""
        criteria = ["A", "B", "C"]
        ahp = AHP(criteria=criteria)

        # Consistent pairwise comparison matrix
        matrix = np.array([[1, 2, 3], [1 / 2, 1, 2], [1 / 3, 1 / 2, 1]], dtype=np.float64)
        ahp.set_comparison_matrix(matrix)

        cr = ahp.consistency_ratio()
        # Consistent matrix should have CR < 0.1
        assert cr < 0.1

    def test_weights_sum_to_one(self) -> None:
        """Test that calculated weights sum to 1."""
        criteria = ["A", "B"]
        ahp = AHP(criteria=criteria)
        matrix = np.array([[1, 2], [1 / 2, 1]], dtype=np.float64)
        ahp.set_comparison_matrix(matrix)

        weights = ahp.calculate_weights()
        assert np.sum(weights) == pytest.approx(1.0, rel=0.01)

    def test_weights_positive(self) -> None:
        """Test that all weights are positive."""
        criteria = ["A", "B", "C"]
        ahp = AHP(criteria=criteria)
        matrix = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]], dtype=np.float64)
        ahp.set_comparison_matrix(matrix)

        weights = ahp.calculate_weights()
        assert np.all(weights > 0)


# Weighted Overlay Tests


class TestWeightedOverlay:
    """Tests for WeightedOverlay analysis."""

    def test_initialization(self) -> None:
        """Test WeightedOverlay initialization."""
        wo = WeightedOverlay()
        assert wo is not None

    def test_overlay_equal_weights(self) -> None:
        """Test weighted overlay with equal weights."""
        wo = WeightedOverlay()

        layers = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[4, 3], [2, 1]], dtype=np.float32),
        ]
        weights = [0.5, 0.5]

        result = wo.calculate(layers, weights)

        assert result.shape == (2, 2)
        # With equal weights, result should be average
        assert result[0, 0] == pytest.approx(2.5, rel=0.01)
        assert result[1, 1] == pytest.approx(2.5, rel=0.01)

    def test_overlay_unequal_weights(self) -> None:
        """Test weighted overlay with unequal weights."""
        wo = WeightedOverlay()

        layers = [
            np.array([[10, 10], [10, 10]], dtype=np.float32),
            np.array([[0, 0], [0, 0]], dtype=np.float32),
        ]
        weights = [0.8, 0.2]

        result = wo.calculate(layers, weights)

        # 0.8 * 10 + 0.2 * 0 = 8
        assert result[0, 0] == pytest.approx(8.0, rel=0.01)

    def test_overlay_weights_sum_validation(self) -> None:
        """Test that weights are normalized if needed."""
        wo = WeightedOverlay()

        layers = [
            np.array([[1]], dtype=np.float32),
            np.array([[1]], dtype=np.float32),
        ]
        weights = [1.0, 1.0]  # Sum to 2, should be normalized

        result = wo.calculate(layers, weights)
        # After normalization: 0.5 * 1 + 0.5 * 1 = 1
        assert result[0, 0] == pytest.approx(1.0, rel=0.01)
