# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_postprocessing.py
# Title       : Tests of class map refinement and vectorisation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest
# =============================================================================
#
# Abstract
# --------
# Checks the postprocessing package on small hand-made rasters: thresholds,
# confidence masks, entropy and margin, morphology, the minimum mapping unit
# sieve, the majority filter, connected component statistics, polygon
# areas in map units, and blended stitching of overlapping tiles.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Functions under test.
from unbihexium.postprocessing import (
    argmax,  # Class map.
    blend_weights,  # Tile weights.
    component_statistics,  # Region statistics.
    confidence_mask,  # Rejection of uncertain pixels.
    connected_components,  # Region labelling.
    fill_small_holes,  # Hole filling.
    majority_filter,  # Modal filter.
    margin,  # Probability margin.
    morphology_clean,  # Binary morphology.
    polygons_to_geodataframe,  # GeoDataFrame output.
    prediction_entropy,  # Entropy.
    raster_to_polygons,  # Polygonisation.
    remove_small_objects,  # Small object removal.
    sieve,  # Minimum mapping unit.
    sigmoid,  # Logistic function.
    simplify_polygons,  # Simplification.
    softmax,  # Softmax.
    stitch_tiles,  # Mosaicking.
    structuring_element,  # Footprints.
    threshold,  # Binary map.
    tile_positions,  # Tile origins.
)  # End of the imports under test.


# Activations, thresholds and argmax.
def test_activations_and_threshold() -> None:
    # Logistic of zero is one half.
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    # Softmax of equal logits is uniform.
    np.testing.assert_allclose(softmax(np.zeros((4, 1, 1)), axis=0), 0.25)
    # Values strictly above the threshold.
    np.testing.assert_array_equal(threshold(np.array([0.4, 0.5, 0.6])), [0, 0, 1])
    # Values strictly below the threshold.
    np.testing.assert_array_equal(threshold(np.array([0.4, 0.6]), 0.5, above=False), [1, 0])
    # Class of the largest score.
    assert argmax(np.array([[0.1], [0.7], [0.2]]))[0] == 1


# Confidence, entropy and margin of class probabilities.
def test_uncertainty_layers() -> None:
    # Two pixels: a confident one and a uniform one, three classes.
    p = np.array([[1.0, 1 / 3], [0.0, 1 / 3], [0.0, 1 / 3]])
    # Entropy is 0 for certainty and 1 for the uniform distribution.
    np.testing.assert_allclose(prediction_entropy(p), [0.0, 1.0], atol=1e-12)
    # Margin is 1 and 0.
    np.testing.assert_allclose(margin(p), [1.0, 0.0], atol=1e-12)
    # The uniform pixel is rejected.
    np.testing.assert_array_equal(confidence_mask(p, 0.5, nodata=255), [0, 255])
    # A margin requirement rejects close calls.
    close = np.array([[0.55], [0.45]])
    # 0.55 passes the confidence but not a 0.2 margin.
    assert confidence_mask(close, 0.5, min_margin=0.2)[0] == 255


# Morphology removes specks and fills gaps.
def test_morphology() -> None:
    # A 3 x 3 square and an isolated pixel.
    mask = np.zeros((7, 7), dtype=np.uint8)
    # Square.
    mask[1:4, 1:4] = 1
    # Speck.
    mask[5, 5] = 1
    # Opening with a 3 x 3 square removes only the speck.
    opened = morphology_clean(mask, "open", 3)
    # Nine pixels remain.
    assert opened.sum() == 9 and opened[5, 5] == 0
    # Removal by size keeps components of at least 9 pixels.
    np.testing.assert_array_equal(remove_small_objects(mask, 9), opened)
    # A ring with a one-pixel hole.
    ring = np.ones((5, 5), dtype=np.uint8)
    # Hole.
    ring[2, 2] = 0
    # The hole is filled; the output keeps the type.
    filled = fill_small_holes(ring, max_size=1)
    # Every pixel set.
    assert filled.all() and filled.dtype == np.uint8
    # A disk of size 5 has 13 pixels.
    assert structuring_element(5, "disk").sum() == 13
    # Unknown operations are rejected.
    with pytest.raises(ValueError, match="operation"):
        # Misspelled name.
        morphology_clean(mask, "opening")


# The sieve merges small regions into their largest neighbour.
def test_sieve_minimum_mapping_unit() -> None:
    # Class 1 with an island of class 2 of two pixels and one of class 3 of six.
    labels = np.ones((6, 6), dtype=np.uint8)
    # Two-pixel island.
    labels[1, 1:3] = 2
    # Six-pixel island.
    labels[3:5, 2:5] = 3
    # Minimum mapping unit of three pixels.
    out = sieve(labels, 3)
    # The small island disappears.
    assert (out == 2).sum() == 0
    # The large island stays.
    assert (out == 3).sum() == 6
    # Type is preserved.
    assert out.dtype == np.uint8
    # Nodata pixels are neither changed nor used.
    labels[0, 5] = 255
    # Sieve with nodata.
    assert sieve(labels, 3, nodata=255)[0, 5] == 255


# The majority filter replaces isolated labels and keeps ties.
def test_majority_filter() -> None:
    # Class 1 with a single pixel of class 2.
    labels = np.ones((5, 5), dtype=np.int16)
    # Isolated pixel.
    labels[2, 2] = 2
    # 8 of 9 votes for class 1.
    assert (majority_filter(labels) == 1).all()
    # Nodata does not vote and is preserved.
    labels[0, 0] = 0
    # Filter with nodata.
    out = majority_filter(labels, nodata=0)
    # Nodata stays.
    assert out[0, 0] == 0 and out[2, 2] == 1
    # Two vertical halves of classes 1 and 2.
    halves = np.array([[1, 1, 2, 2]] * 3, dtype=np.int32)
    # Interior boundary pixels see 6 votes against 3 and keep their class.
    np.testing.assert_array_equal(majority_filter(halves), halves)


# Components of different classes are labelled separately.
def test_connected_components_statistics() -> None:
    # Two touching classes and a second region of class 1.
    image = np.array([[1, 1, 2], [0, 0, 2], [1, 0, 0]])
    # Label with 4-connectivity.
    labels, n = connected_components(image, connectivity=4)
    # Three regions.
    assert n == 3
    # Statistics with 100 square metre pixels.
    stats = component_statistics(labels, values=image * 10.0, pixel_area=100.0, source=image)
    # Region sizes.
    assert sorted(s["pixels"] for s in stats) == [1, 2, 2]
    # Record of the vertical class 2 region.
    region = next(s for s in stats if s["value"] == 2)
    # Area, bounding box, centroid and mean value.
    assert region["area"] == 200.0 and region["bbox"] == (0, 2, 2, 3)
    # Centroid of rows 0 and 1 in column 2.
    assert region["centroid"] == (0.5, 2.0) and region["mean"] == 20.0


# Polygons carry map coordinates and areas.
def test_raster_to_polygons() -> None:
    # A 2 x 2 block of class 1.
    image = np.zeros((4, 4), dtype=np.uint8)
    # Block.
    image[1:3, 1:3] = 1
    # 10 m pixels, upper-left corner (100, 200), north up.
    pairs = raster_to_polygons(image, (10.0, 0.0, 100.0, 0.0, -10.0, 200.0))
    # One polygon of class 1.
    assert len(pairs) == 1 and pairs[0][1] == 1.0
    # Area of four 10 m pixels.
    assert pairs[0][0].area == pytest.approx(400.0)
    # Bounds in map coordinates.
    assert pairs[0][0].bounds == (110.0, 170.0, 130.0, 190.0)
    # Simplification keeps the square.
    assert simplify_polygons(pairs, 1.0)[0][0].area == pytest.approx(400.0)
    # Data frame with an area column.
    frame = polygons_to_geodataframe(image, crs="EPSG:32635")
    # Pixel units without a transform.
    assert frame["area"].iloc[0] == pytest.approx(4.0) and frame.crs.to_epsg() == 32635


# Tiles cover the image and stitch back exactly.
def test_tiles_round_trip() -> None:
    # Tile origins of a 10 x 10 image with 4 x 4 tiles and overlap 1.
    positions = tile_positions((10, 10), 4, overlap=1)
    # Step 3 reaches the edge-aligned origin 6 exactly, so rows are 0, 3 and 6.
    assert sorted({r for r, _ in positions}) == [0, 3, 6]
    # Image with two bands.
    image = np.arange(200, dtype=float).reshape(2, 10, 10)
    # Cut the tiles.
    tiles = [image[:, r : r + 4, c : c + 4] for r, c in positions]
    # Stitch with the linear blend.
    out = stitch_tiles(tiles, positions, image.shape, overlap=1, blend="linear")
    # Identical tiles in the overlaps reproduce the image.
    np.testing.assert_allclose(out, image)
    # Linear weights fall towards the edges.
    w = blend_weights((4, 4), overlap=1, blend="linear")
    # Corner weight 0.25, centre weight 1.
    assert w[0, 0] == pytest.approx(0.25) and w[1, 1] == pytest.approx(1.0)
    # Uncovered pixels are NaN.
    partial = stitch_tiles([np.ones((2, 2))], [(0, 0)], (3, 3))
    # Only the tile area is defined.
    assert np.isfinite(partial).sum() == 4


# =============================================================================
# End of module tests/unit/test_postprocessing.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
