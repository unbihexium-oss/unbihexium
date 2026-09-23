# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/analysis/__init__.py
# Title       : Spatial analysis: routing, suitability and zonal statistics
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# GIS analysis tools:
#
#   network      graph routing (Dijkstra, A*, service areas, closest
#                facility, OD cost matrices) and raster cost distance
#   suitability  AHP weights, factor standardisation and weighted overlay
#   zonal        zonal statistics of rasters
# =============================================================================

# Routing.
from unbihexium.analysis.network import (
    AccessibilityResult,  # Costs from one origin.
    AStarPathfinder,  # Graph with A* queries.
    NetworkAnalyzer,  # Graph routing.
    Route,  # Path record.
    cost_distance,  # Accumulated cost surface.
    least_cost_path,  # Cheapest raster path.
)  # End of the routing imports.

# Suitability analysis.
from unbihexium.analysis.suitability import (
    AHP,  # Analytic Hierarchy Process.
    SuitabilityResult,  # Overlay result.
    WeightedOverlay,  # Weighted linear combination.
    fuzzy_membership,  # Fuzzy standardisation.
    reclassify,  # Class scores.
    rescale_linear,  # Linear standardisation.
    weighted_overlay,  # Overlay of arrays or rasters.
)  # End of the suitability imports.

# Zonal statistics.
from unbihexium.analysis.zonal import (
    ZonalResult,  # Record per zone.
    ZonalStatistics,  # Table per statistic.
    rasterize_zones,  # Polygons to zones.
    zonal_statistics,  # Records per zone.
    zonal_table,  # Table of all zones.
)  # End of the zonal imports.

# Public names of the package.
__all__ = [
    "AHP",  # Analytic Hierarchy Process.
    "AStarPathfinder",  # Graph with A* queries.
    "AccessibilityResult",  # Costs from one origin.
    "NetworkAnalyzer",  # Graph routing.
    "Route",  # Path record.
    "SuitabilityResult",  # Overlay result.
    "WeightedOverlay",  # Weighted linear combination.
    "ZonalResult",  # Record per zone.
    "ZonalStatistics",  # Table per statistic.
    "cost_distance",  # Accumulated cost surface.
    "fuzzy_membership",  # Fuzzy standardisation.
    "least_cost_path",  # Cheapest raster path.
    "rasterize_zones",  # Polygons to zones.
    "reclassify",  # Class scores.
    "rescale_linear",  # Linear standardisation.
    "weighted_overlay",  # Overlay of arrays or rasters.
    "zonal_statistics",  # Records per zone.
    "zonal_table",  # Table of all zones.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/analysis/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
