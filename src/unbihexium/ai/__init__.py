"""AI domain module for detection, segmentation, and super-resolution."""

from unbihexium.ai.detection import (
    ShipDetector,
    BuildingDetector,
    AircraftDetector,
    VehicleDetector,
    ObjectDetector,
)
from unbihexium.ai.segmentation import (
    SemanticSegmenter,
    ChangeDetector,
    WaterDetector,
    CropDetector,
    GreenhouseDetector,
)
from unbihexium.ai.super_resolution import SuperResolution

__all__ = [
    "AircraftDetector",
    "BuildingDetector",
    "ChangeDetector",
    "CropDetector",
    "GreenhouseDetector",
    "ObjectDetector",
    "SemanticSegmenter",
    "ShipDetector",
    "SuperResolution",
    "VehicleDetector",
    "WaterDetector",
]
