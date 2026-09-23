# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for AI detection and segmentation modules.

This module provides comprehensive tests for:
- Object detection (ships, buildings, aircraft, vehicles)
- Semantic segmentation (water, crops, change detection)
- Super-resolution baseline
- Model wrapper functionality

All tests use synthetic fixtures to avoid large data dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from unbihexium.ai.detection import (
    AircraftDetector,
    BuildingDetector,
    Detection,
    DetectionResult,
    ShipDetector,
    VehicleDetector,
)
from unbihexium.ai.segmentation import (
    ChangeDetector,
    CropDetector,
    GreenhouseDetector,
    SegmentationResult,
    WaterDetector,
)
from unbihexium.ai.super_resolution import SuperResolution, SuperResolutionResult
from unbihexium.core.raster import Raster


# Fixtures


@pytest.fixture
def seed() -> int:
    """Deterministic seed for reproducibility."""
    return 42


@pytest.fixture
def synthetic_image(seed: int) -> NDArray[np.floating]:
    """Generate synthetic 3-band image for testing."""
    np.random.seed(seed)
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def sample_raster(synthetic_image: NDArray[np.floating]) -> Raster:
    """Create a sample raster from synthetic image."""
    return Raster.from_array(synthetic_image, crs="EPSG:4326")


@pytest.fixture
def larger_raster(seed: int) -> Raster:
    """Create a larger raster for super-resolution tests."""
    np.random.seed(seed)
    data = np.random.rand(3, 128, 128).astype(np.float32)
    return Raster.from_array(data, crs="EPSG:4326")


# Detection Tests


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self) -> None:
        """Test creating a Detection object."""
        det = Detection(
            bbox=(10.0, 20.0, 50.0, 60.0),
            confidence=0.95,
            class_id=0,
            class_name="ship",
        )
        assert det.bbox == (10.0, 20.0, 50.0, 60.0)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "ship"

    def test_detection_confidence_range(self) -> None:
        """Test that confidence is in valid range."""
        det = Detection(
            bbox=(0, 0, 10, 10),
            confidence=0.5,
            class_id=0,
            class_name="test",
        )
        assert 0.0 <= det.confidence <= 1.0


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty detection result."""
        result = DetectionResult(detections=[], count=0, model_id="test_model")
        assert result.count == 0
        assert len(result.detections) == 0
        assert result.model_id == "test_model"

    def test_result_with_detections(self) -> None:
        """Test detection result with multiple detections."""
        detections = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="ship"),
            Detection(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="ship"),
        ]
        result = DetectionResult(detections=detections, count=2, model_id="test")
        assert result.count == 2
        assert result.detections[0].confidence > result.detections[1].confidence


class TestShipDetector:
    """Tests for ShipDetector."""

    def test_initialization_default(self) -> None:
        """Test detector initialization with defaults."""
        detector = ShipDetector()
        assert detector.threshold == 0.5

    def test_initialization_custom_threshold(self) -> None:
        """Test detector initialization with custom threshold."""
        detector = ShipDetector(threshold=0.7)
        assert detector.threshold == 0.7

    def test_predict_returns_result(self, sample_raster: Raster) -> None:
        """Test that predict returns DetectionResult."""
        detector = ShipDetector(threshold=0.5)
        result = detector.predict(sample_raster)
        assert isinstance(result, DetectionResult)
        assert result.model_id == "ship_detector_tiny"

    def test_predict_deterministic(self, sample_raster: Raster) -> None:
        """Test that predictions are deterministic."""
        detector = ShipDetector(threshold=0.5)
        result1 = detector.predict(sample_raster)
        result2 = detector.predict(sample_raster)
        assert result1.count == result2.count


class TestBuildingDetector:
    """Tests for BuildingDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = BuildingDetector(threshold=0.6)
        assert detector.threshold == 0.6

    def test_predict_returns_result(self, sample_raster: Raster) -> None:
        """Test that predict returns DetectionResult."""
        detector = BuildingDetector(threshold=0.5)
        result = detector.predict(sample_raster)
        assert isinstance(result, DetectionResult)


class TestAircraftDetector:
    """Tests for AircraftDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = AircraftDetector(threshold=0.6)
        assert detector.threshold == 0.6


class TestVehicleDetector:
    """Tests for VehicleDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = VehicleDetector(threshold=0.7)
        assert detector.threshold == 0.7


# Segmentation Tests


class TestSegmentationResult:
    """Tests for SegmentationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test segmentation result creation."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = SegmentationResult(
            mask=mask,
            classes=["background", "water"],
            model_id="test_seg",
        )
        assert result.mask.shape == (64, 64)
        assert len(result.classes) == 2
        assert result.model_id == "test_seg"

    def test_mask_dtype(self) -> None:
        """Test that mask has correct dtype."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = SegmentationResult(mask=mask, classes=["bg"], model_id="test")
        assert result.mask.dtype == np.uint8


class TestWaterDetector:
    """Tests for WaterDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = WaterDetector(threshold=0.5)
        assert detector.threshold == 0.5

    def test_predict_returns_result(self, sample_raster: Raster) -> None:
        """Test that predict returns SegmentationResult."""
        detector = WaterDetector(threshold=0.5)
        result = detector.predict(sample_raster)
        assert isinstance(result, SegmentationResult)

    def test_mask_shape_matches_input(self, sample_raster: Raster) -> None:
        """Test that output mask matches input dimensions."""
        detector = WaterDetector(threshold=0.5)
        result = detector.predict(sample_raster)
        # Mask should match spatial dimensions
        assert result.mask.shape[0] == sample_raster.shape[1]
        assert result.mask.shape[1] == sample_raster.shape[2]


class TestChangeDetector:
    """Tests for ChangeDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = ChangeDetector(threshold=0.5)
        assert detector.threshold == 0.5


class TestCropDetector:
    """Tests for CropDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = CropDetector(threshold=0.5)
        assert detector.threshold == 0.5


class TestGreenhouseDetector:
    """Tests for GreenhouseDetector."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = GreenhouseDetector(threshold=0.5)
        assert detector.threshold == 0.5


# Super-Resolution Tests


class TestSuperResolutionResult:
    """Tests for SuperResolutionResult dataclass."""

    def test_result_creation(self) -> None:
        """Test super-resolution result creation."""
        result = SuperResolutionResult(
            raster=None,
            scale_factor=2,
            source="test.tif",
            model_id="sr_test",
        )
        assert result.scale_factor == 2
        assert result.model_id == "sr_test"


class TestSuperResolution:
    """Tests for SuperResolution class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        sr = SuperResolution()
        assert sr.scale_factor == 2
        assert sr.tile_size == 256

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        sr = SuperResolution(scale_factor=4, tile_size=128)
        assert sr.scale_factor == 4
        assert sr.tile_size == 128

    def test_enhance_returns_result(self, sample_raster: Raster) -> None:
        """Test that enhance returns SuperResolutionResult."""
        sr = SuperResolution(scale_factor=2)
        result = sr.enhance(sample_raster)
        assert isinstance(result, SuperResolutionResult)

    def test_enhance_upscales_correctly(self, sample_raster: Raster) -> None:
        """Test that output is upscaled by scale_factor."""
        sr = SuperResolution(scale_factor=2)
        result = sr.enhance(sample_raster)
        assert result.raster is not None
        # Output should be 2x input in spatial dimensions
        assert result.raster.shape[1] == sample_raster.shape[1] * 2
        assert result.raster.shape[2] == sample_raster.shape[2] * 2

    def test_enhance_preserves_bands(self, sample_raster: Raster) -> None:
        """Test that number of bands is preserved."""
        sr = SuperResolution(scale_factor=2)
        result = sr.enhance(sample_raster)
        assert result.raster is not None
        assert result.raster.shape[0] == sample_raster.shape[0]

    def test_enhance_output_finite(self, sample_raster: Raster) -> None:
        """Test that output contains only finite values."""
        sr = SuperResolution(scale_factor=2)
        result = sr.enhance(sample_raster)
        assert result.raster is not None
        result.raster.load()
        assert np.all(np.isfinite(result.raster.data))


# Integration Tests


class TestDetectionIntegration:
    """Integration tests for detection pipeline."""

    def test_multiple_detectors_same_input(self, sample_raster: Raster) -> None:
        """Test running multiple detectors on same input."""
        ship_det = ShipDetector(threshold=0.5)
        building_det = BuildingDetector(threshold=0.5)

        ship_result = ship_det.predict(sample_raster)
        building_result = building_det.predict(sample_raster)

        assert isinstance(ship_result, DetectionResult)
        assert isinstance(building_result, DetectionResult)
        assert ship_result.model_id != building_result.model_id
