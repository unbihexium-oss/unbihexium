# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/network/__init__.py
# Title       : Network and cost-surface routing
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Routing on graphs and rasters:
#
#   graph         NetworkAnalyzer: Dijkstra and A* shortest paths, service
#                 areas, accessibility, closest facility, OD cost matrices
#   a_star        A* search with Euclidean, Manhattan and haversine
#                 heuristics
#   cost_surface  accumulated cost distance and least-cost paths over a
#                 friction raster
#
# This package replaces the former module unbihexium/analysis/network.py;
# its public names (NetworkAnalyzer, Route, AccessibilityResult) are
# importable from here as before.
# =============================================================================

# A* search.
from unbihexium.analysis.network.a_star import (
    AStarPathfinder,  # Graph with A* queries.
    AStarResult,  # A* result record.
    Heuristic,  # Heuristic names.
    astar,  # A* on an adjacency list.
    euclidean_distance,  # Planar distance.
    haversine_distance,  # Great-circle distance.
    manhattan_distance,  # Grid distance.
)  # End of the A* imports.

# Raster cost distance.
from unbihexium.analysis.network.cost_surface import cost_distance, least_cost_path

# Graph routing.
from unbihexium.analysis.network.graph import AccessibilityResult, NetworkAnalyzer, Route

# Public names of the package.
__all__ = [
    "AStarPathfinder",  # Graph with A* queries.
    "AStarResult",  # A* result record.
    "AccessibilityResult",  # Costs from one origin.
    "Heuristic",  # Heuristic names.
    "NetworkAnalyzer",  # Graph routing.
    "Route",  # Path record.
    "astar",  # A* on an adjacency list.
    "cost_distance",  # Accumulated cost surface.
    "euclidean_distance",  # Planar distance.
    "haversine_distance",  # Great-circle distance.
    "least_cost_path",  # Cheapest raster path.
    "manhattan_distance",  # Grid distance.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/analysis/network/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
