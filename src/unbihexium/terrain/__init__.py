# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/terrain/__init__.py
# Title       : Terrain analysis of digital elevation models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Array functions for gridded digital elevation models (north-up, NaN for
# nodata):
#
#   derivatives  gradient, slope, aspect, hillshade, curvatures, TPI, TRI,
#                roughness and vector ruggedness
#   hydrology    depression filling, D8 flow direction and accumulation,
#                watersheds, streams and the topographic wetness index
#   visibility   line-of-sight viewshed
# =============================================================================

# Local surface derivatives.
from unbihexium.terrain.derivatives import (
    aspect,  # Downslope direction.
    curvature,  # Profile and plan curvature.
    gradient,  # Horn gradient.
    hillshade,  # Shaded relief.
    roughness,  # Elevation range of the window.
    slope,  # Slope angle or percent.
    total_curvature,  # Sum of the curvatures.
    tpi,  # Topographic position index.
    tri,  # Terrain ruggedness index.
    vrm,  # Vector ruggedness measure.
)  # End of the derivative imports.

# Flow routing.
from unbihexium.terrain.hydrology import (
    D8_NEIGHBOURS,  # D8 offsets and codes.
    extract_streams,  # Stream mask.
    fill_depressions,  # Priority-flood filling.
    flow_accumulation,  # Upstream cells.
    flow_direction_d8,  # D8 codes.
    twi,  # Topographic wetness index.
    watershed,  # Cells upstream of an outlet.
)  # End of the hydrology imports.

# Visibility.
from unbihexium.terrain.visibility import viewshed

# Public names of the package.
__all__ = [
    "D8_NEIGHBOURS",  # D8 offsets and codes.
    "aspect",  # Downslope direction.
    "curvature",  # Profile and plan curvature.
    "extract_streams",  # Stream mask.
    "fill_depressions",  # Priority-flood filling.
    "flow_accumulation",  # Upstream cells.
    "flow_direction_d8",  # D8 codes.
    "gradient",  # Horn gradient.
    "hillshade",  # Shaded relief.
    "roughness",  # Elevation range of the window.
    "slope",  # Slope angle or percent.
    "total_curvature",  # Sum of the curvatures.
    "tpi",  # Topographic position index.
    "tri",  # Terrain ruggedness index.
    "twi",  # Topographic wetness index.
    "viewshed",  # Line-of-sight visibility.
    "vrm",  # Vector ruggedness measure.
    "watershed",  # Cells upstream of an outlet.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/terrain/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
