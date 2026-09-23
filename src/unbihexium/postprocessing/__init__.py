# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/postprocessing/__init__.py
# Title       : Refinement of model outputs into map products
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy, SciPy, rasterio and
#               Shapely
# =============================================================================
#
# Abstract
# --------
# Turns scores and class maps into clean, publishable map products:
#
#   activations   thresholds, argmax, confidence masks, entropy and margin
#   morphology    opening and closing, small object and hole removal,
#                 minimum mapping unit sieve, majority filter, connected
#                 components and their statistics
#   vectorize     raster to polygons, simplification, GeoDataFrame output
#   tiles         tile windows and blended stitching of tiled predictions
#
# Usage
# -----
#   from unbihexium.postprocessing import confidence_mask, sieve
#   labels = confidence_mask(probabilities, min_confidence=0.6)
#   labels = sieve(labels, min_size=9, nodata=255)
# =============================================================================

# Activations and class maps.
from unbihexium.postprocessing.activations import (
    argmax,  # Class map from scores.
    confidence_mask,  # Class map with rejected pixels.
    margin,  # Best minus second-best probability.
    prediction_entropy,  # Normalised Shannon entropy.
    sigmoid,  # Logistic function.
    softmax,  # Softmax.
    threshold,  # Binary map.
)  # End of the activation imports.

# Morphology and regions.
from unbihexium.postprocessing.morphology import (
    component_statistics,  # Statistics per region.
    connected_components,  # Region labelling.
    fill_small_holes,  # Hole filling.
    majority_filter,  # Modal filter.
    morphology_clean,  # Binary morphology.
    remove_small_objects,  # Small object removal.
    sieve,  # Minimum mapping unit.
    structuring_element,  # Footprints.
)  # End of the morphology imports.

# Tiles.
from unbihexium.postprocessing.tiles import (
    blend_weights,  # Tile weights.
    stitch_tiles,  # Mosaicking.
    tile_positions,  # Tile origins.
)  # End of the tile imports.

# Vectorisation.
from unbihexium.postprocessing.vectorize import (
    as_affine,  # Transform conversion.
    polygons_to_geodataframe,  # GeoDataFrame output.
    raster_to_polygons,  # Polygonisation.
    simplify_polygons,  # Douglas-Peucker simplification.
)  # End of the vectorisation imports.

# Public names of the package.
__all__ = [
    "argmax",  # Class map from scores.
    "as_affine",  # Transform conversion.
    "blend_weights",  # Tile weights.
    "component_statistics",  # Statistics per region.
    "confidence_mask",  # Class map with rejected pixels.
    "connected_components",  # Region labelling.
    "fill_small_holes",  # Hole filling.
    "majority_filter",  # Modal filter.
    "margin",  # Best minus second-best probability.
    "morphology_clean",  # Binary morphology.
    "polygons_to_geodataframe",  # GeoDataFrame output.
    "prediction_entropy",  # Normalised Shannon entropy.
    "raster_to_polygons",  # Polygonisation.
    "remove_small_objects",  # Small object removal.
    "sieve",  # Minimum mapping unit.
    "sigmoid",  # Logistic function.
    "simplify_polygons",  # Douglas-Peucker simplification.
    "softmax",  # Softmax.
    "stitch_tiles",  # Mosaicking.
    "structuring_element",  # Footprints.
    "threshold",  # Binary map.
    "tile_positions",  # Tile origins.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/postprocessing/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
