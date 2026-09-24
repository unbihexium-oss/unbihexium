# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_ai.py
# Title       : Tests of the task APIs and tiled inference
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and PyTorch; the ONNX
#               tests also need onnx and ONNX Runtime
# =============================================================================
#
# Abstract
# --------
# Runs every task API on synthetic rasters with the tiny starter models and
# checks the result layout and georeferencing: detection, segmentation,
# change detection, dense and scene regression, spectral indices,
# enhancement and super-resolution. Checks that tiled inference agrees with
# a single forward pass away from tile borders, that missing input pixels
# are NaN in the output, that the ONNX Runtime backend reproduces PyTorch,
# and that models of the wrong task or band count are rejected.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Arrays.
import numpy as np

# Test framework.
import pytest

# PyTorch is optional for the library; skip these tests without it.
torch = pytest.importorskip("torch")

# Task APIs under test.
from unbihexium.ai import (  # noqa: E402 - imported after the skip check
    ChangeDetector,  # Change maps.
    DetectionResult,  # Detection result.
    Enhancer,  # Image-to-image models.
    NDVICalculator,  # NDVI formula.
    Predictor,  # Tiled inference.
    RegressionResult,  # Regression result.
    SegmentationResult,  # Segmentation result.
    ShipDetector,  # Ship detector.
    SuperResolution,  # Upscaling.
    TreeHeightEstimator,  # Canopy height.
    WaterDetector,  # Water maps.
    YieldPredictor,  # Crop yield.
    predict,  # Generic prediction.
    write_result,  # Result output.
)  # End of the API imports.

# Model construction.
from unbihexium.ai.models import build_model  # noqa: E402 - imported after the skip check

# Raster container.
from unbihexium.core.raster import Raster  # noqa: E402 - imported after the skip check

# 10 m UTM grid used by the georeferenced test rasters.
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)


# Georeferenced random raster.
def raster(bands: int, height: int = 64, width: int = 64, seed: int = 0) -> Raster:
    # Seeded random reflectances.
    data = np.random.default_rng(seed).random((bands, height, width), dtype=np.float32)
    # Raster on the UTM grid.
    return Raster.from_array(data, crs="EPSG:32635", transform=TRANSFORM)


# Detection returns boxes with pixel and map coordinates.
def test_ship_detector() -> None:
    # Low threshold so that the untrained model returns boxes.
    detector = ShipDetector(variant="tiny", threshold=0.05)
    # The model id is known without loading the model.
    assert detector.model_id == "ship_detector_tiny" and detector._predictor is None
    # Run on a raster.
    result = detector.predict(raster(3))
    # Result type and model.
    assert isinstance(result, DetectionResult) and result.model_id == "ship_detector_tiny"
    # Deterministic predictions.
    assert result.count == detector.predict(raster(3)).count
    # Every detection lies inside the image and has a map box.
    for d in result.detections:
        # Pixel box inside the image.
        assert 0 <= d.bbox[0] <= d.bbox[2] <= 64 and 0 <= d.bbox[1] <= d.bbox[3] <= 64
        # Map box on the UTM grid.
        assert d.geo_bbox is not None and 500000 <= d.geo_bbox[0] <= 500640
    # GeoJSON uses the raster CRS.
    assert detector.predict(raster(3)).to_geojson()["crs"] == "EPSG:32635"


# Segmentation keeps the input size and georeferencing.
def test_water_detector() -> None:
    # Water detector with probabilities.
    segmenter = WaterDetector(variant="tiny", return_probabilities=True)
    # Four-band raster of odd size.
    result = segmenter.predict(raster(4, 45, 70))
    # Result type.
    assert isinstance(result, SegmentationResult)
    # Mask and probabilities match the input size.
    assert result.mask.shape == (45, 70) and result.probabilities.shape == (2, 45, 70)
    # Class names from the catalogue.
    assert result.classes == ["background", "water"]
    # Probabilities sum to one.
    assert np.allclose(result.probabilities.sum(axis=0), 1, atol=1e-5)
    # The written raster keeps the transform.
    assert result.to_raster().metadata.transform == TRANSFORM


# Missing input pixels are marked in the output.
def test_nodata_propagation() -> None:
    # Raster with a missing pixel in every band.
    r = raster(4)
    # Mark the pixel.
    r.data[:, 10, 20] = np.nan
    # Segment.
    result = WaterDetector(variant="tiny").predict(r)
    # The pixel has the no-data label.
    assert result.mask[10, 20] == 255 and result.mask[0, 0] != 255


# Change detection stacks the two dates.
def test_change_detector() -> None:
    # Generic change detector: three bands per date.
    detector = ChangeDetector(variant="tiny")
    # Two dates.
    result = detector.predict_pair(raster(3, seed=1), raster(3, seed=2))
    # Change map of the input size.
    assert result.mask.shape == (64, 64) and result.classes == ["no_change", "change"]
    # Dates of different size are rejected.
    with pytest.raises(ValueError, match="differ"):
        # Mismatched shapes.
        detector.predict_pair(raster(3, 64, 64), raster(3, 32, 32))


# Regression APIs return named values with units.
def test_regression_apis() -> None:
    # Canopy height map.
    height = TreeHeightEstimator(variant="tiny").predict(raster(12))
    # Dense result inside the catalogue range [0, 60] m.
    assert isinstance(height, RegressionResult) and height.values.shape == (1, 64, 64)
    # Range of the outputs.
    assert 0 <= np.nanmin(height.values) <= np.nanmax(height.values) <= 60
    # Units from the catalogue.
    assert height.units == ["m"]
    # Scene regression.
    yield_ = YieldPredictor(variant="tiny").predict(raster(10, 40, 40))
    # One value per output.
    assert yield_.values.shape == (1,) and not yield_.is_dense


# The NDVI model is exact.
def test_ndvi_exact() -> None:
    # Red and near-infrared bands.
    data = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.5)]).astype(np.float32)
    # Model output.
    values = NDVICalculator().predict(data).values
    # (0.5 - 0.1) / (0.5 + 0.1).
    assert np.allclose(values, 0.4 / 0.6)


# Super-resolution and enhancement.
def test_image_to_image() -> None:
    # Catalogue super-resolution upscales by 4.
    sr = SuperResolution(variant="tiny").enhance(raster(3, 24, 20))
    # Output size and pixel size.
    assert sr.raster.shape == (3, 96, 80) and sr.raster.metadata.transform[0] == 2.5
    # Another factor builds a customised model.
    doubled = SuperResolution(variant="tiny", scale_factor=2).enhance(raster(3, 16, 16))
    # Twice the size.
    assert doubled.raster.shape == (3, 32, 32)
    # Pansharpening keeps the grid and returns four bands.
    enhanced = Enhancer(variant="tiny").predict(raster(5, 32, 32))
    # Four output bands on the input grid.
    assert enhanced.raster.shape == (4, 32, 32) and enhanced.bands == [
        "blue",
        "green",
        "red",
        "nir",
    ]


# Tiled inference agrees with one forward pass away from the tile borders.
def test_tiling_matches_single_pass() -> None:
    # Tiny regression model.
    model = build_model("tree_height_estimator_tiny")
    # Input of 96 x 96 pixels.
    x = np.random.default_rng(3).random((12, 96, 96), dtype=np.float32)
    # Reference: one forward pass.
    with torch.no_grad():
        # Model output.
        reference = model(torch.from_numpy(x)[None])[0].numpy()
    # Tiles of 64 pixels with half overlap.
    tiled = Predictor(model, tile_size=64, overlap=0.5).dense(x)
    # Same shape.
    assert tiled.shape == reference.shape
    # The centre of the image is far from every tile border.
    centre = (slice(None), slice(40, 56), slice(40, 56))
    # Close agreement in the centre.
    assert np.allclose(tiled[centre], reference[centre], atol=0.5)


# Detection over several tiles returns each object once.
def test_tiled_detection_boxes_in_image() -> None:
    # Predictor with small tiles.
    predictor = Predictor(build_model("ship_detector_tiny"), tile_size=64, overlap=0.25)
    # Image larger than one tile.
    boxes, scores, classes = predictor.detect(raster(3, 150, 130).data, threshold=0.05)
    # Boxes inside the image.
    assert (boxes[:, 2] <= 130).all() and (boxes[:, 3] <= 150).all()
    # Sorted by decreasing score.
    assert np.all(np.diff(scores) <= 1e-9)


# Inputs with the wrong band count or task are rejected.
def test_rejections() -> None:
    # Three bands instead of four.
    with pytest.raises(ValueError, match="expects 4 bands"):
        # Run the water detector.
        WaterDetector(variant="tiny").predict(raster(3))
    # A segmentation model used as a detector.
    with pytest.raises(ValueError, match="needs a detection model"):
        # Wrong task.
        ShipDetector(model="lulc_classifier_tiny").predict(raster(10))


# The generic predict function dispatches on the task and writes files.
def test_predict_and_write(tmp_path: Path) -> None:
    # Detection result.
    boxes = predict("ship_detector_tiny", raster(3), threshold=0.05)
    # Detection result type.
    assert isinstance(boxes, DetectionResult)
    # Written as GeoJSON.
    written = write_result(boxes, tmp_path / "ships.geojson")
    # JSON object text.
    assert written.read_text(encoding="utf-8").startswith("{")
    # Segmentation result written as GeoTIFF.
    seg = predict("water_surface_detector_tiny", raster(4))
    # Write the class map.
    path = write_result(seg, tmp_path / "water.tif")
    # Read it back.
    written = Raster.from_file(path)
    # Same grid.
    assert written.shape == (1, 64, 64) and written.metadata.transform[:6] == TRANSFORM
    # Scene values written as JSON.
    scene = write_result(predict("yield_predictor_tiny", raster(10)), tmp_path / "y.json")
    # The output name is in the file.
    assert "yield" in scene.read_text(encoding="utf-8")


# The ONNX Runtime backend reproduces the PyTorch backend.
def test_onnx_backend(tmp_path: Path) -> None:
    # ONNX Runtime is optional.
    pytest.importorskip("onnxruntime")
    # onnx is needed for the export metadata.
    pytest.importorskip("onnx")
    # Export function.
    from unbihexium.zoo.export import export_onnx

    # Tiny segmentation model.
    model = build_model("water_surface_detector_tiny")
    # Export it.
    path = export_onnx(model, tmp_path / "water.onnx")
    # Input image.
    x = raster(4, 50, 60).data
    # PyTorch probabilities.
    expected = Predictor(model, tile_size=64).dense(x)
    # ONNX probabilities.
    actual = Predictor(path, tile_size=64).dense(x)
    # Same results.
    assert np.allclose(actual, expected, atol=1e-4)
    # The task API accepts the ONNX file as weights.
    assert WaterDetector(weights=path).predict(x).model_id == "water_surface_detector_tiny"


# =============================================================================
# End of module tests/unit/test_ai.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
