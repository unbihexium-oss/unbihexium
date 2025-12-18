"""AI domain module for detection, segmentation, and super-resolution."""

from unbihexium.ai.detection import (
    AircraftDetector,
    BuildingDetector,
    ObjectDetector,
    ShipDetector,
    VehicleDetector,
)
from unbihexium.ai.segmentation import (
    ChangeDetector,
    CropDetector,
    GreenhouseDetector,
    SemanticSegmenter,
    WaterDetector,
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
