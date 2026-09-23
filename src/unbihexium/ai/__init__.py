# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/__init__.py
# Title       : Machine learning for Earth observation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14; running models needs PyTorch or ONNX
#               Runtime, training needs PyTorch
# =============================================================================
#
# Abstract
# --------
# Task APIs, inference, training and evaluation for the model zoo:
#
#   detection         ObjectDetector and subclasses (ships, buildings, ...)
#   segmentation      SemanticSegmenter, ChangeDetector and subclasses
#   regression        DenseRegressor, SceneRegressor and subclasses
#   super_resolution  SuperResolution and Enhancer
#   predict           run any model and write its result
#   inference         tiled Predictor for PyTorch and ONNX Runtime
#   training          ChipDataset, Trainer, train and evaluate (PyTorch)
#   data              dataset folders and synthetic datasets
#   evaluation        metrics of every task
#   models            network architectures and the model factory (PyTorch)
#
# Importing this package does not import PyTorch; the models, losses and
# training modules load it when they are imported.
# =============================================================================

# Detection APIs.
from unbihexium.ai.detection import (
    AircraftDetector,  # Aircraft.
    BuildingDetector,  # Buildings.
    CropDetector,  # Crop fields.
    FireDetector,  # Fires and burn scars.
    GreenhouseDetector,  # Greenhouses.
    ObjectDetector,  # Generic detector.
    PivotDetector,  # Centre-pivot fields.
    SARShipDetector,  # Ships in SAR imagery.
    ShipDetector,  # Ships.
    VehicleDetector,  # Vehicles.
)  # End of the detection imports.

# Inference.
from unbihexium.ai.inference import Predictor

# Generic prediction and output.
from unbihexium.ai.predict import predict, task_api, write_result

# Regression APIs.
from unbihexium.ai.regression import (
    DenseRegressor,  # Per-pixel values.
    LandSurfaceTemperature,  # Land surface temperature.
    NDVICalculator,  # NDVI formula.
    SceneRegressor,  # Scene values.
    TreeHeightEstimator,  # Canopy height.
    YieldPredictor,  # Crop yield.
)  # End of the regression imports.

# Result records.
from unbihexium.ai.results import (
    Detection,  # One box.
    DetectionResult,  # Boxes of an image.
    EnhancementResult,  # Enhanced bands.
    RegressionResult,  # Values.
    SegmentationResult,  # Class map.
    SuperResolutionResult,  # Upscaled bands.
)  # End of the result imports.

# Segmentation APIs.
from unbihexium.ai.segmentation import (
    ChangeDetector,  # Change maps.
    CloudMasker,  # Clouds.
    CropClassifier,  # Crop types.
    FloodMapper,  # SAR floods.
    LandCoverClassifier,  # Land cover.
    OilSpillDetector,  # SAR oil spills.
    SemanticSegmenter,  # Generic segmenter.
    WaterDetector,  # Water surfaces.
)  # End of the segmentation imports.

# Super-resolution and enhancement APIs.
from unbihexium.ai.super_resolution import Enhancer, SuperResolution

# Public names of the package.
__all__ = [
    "AircraftDetector",  # Aircraft.
    "BuildingDetector",  # Buildings.
    "ChangeDetector",  # Change maps.
    "CloudMasker",  # Clouds.
    "CropClassifier",  # Crop types.
    "CropDetector",  # Crop fields.
    "DenseRegressor",  # Per-pixel values.
    "Detection",  # One box.
    "DetectionResult",  # Boxes of an image.
    "EnhancementResult",  # Enhanced bands.
    "Enhancer",  # Image-to-image models.
    "FireDetector",  # Fires and burn scars.
    "FloodMapper",  # SAR floods.
    "GreenhouseDetector",  # Greenhouses.
    "LandCoverClassifier",  # Land cover.
    "LandSurfaceTemperature",  # Land surface temperature.
    "NDVICalculator",  # NDVI formula.
    "ObjectDetector",  # Generic detector.
    "OilSpillDetector",  # SAR oil spills.
    "PivotDetector",  # Centre-pivot fields.
    "Predictor",  # Tiled inference.
    "RegressionResult",  # Values.
    "SARShipDetector",  # Ships in SAR imagery.
    "SceneRegressor",  # Scene values.
    "SegmentationResult",  # Class map.
    "SemanticSegmenter",  # Generic segmenter.
    "ShipDetector",  # Ships.
    "SuperResolution",  # Upscaling.
    "SuperResolutionResult",  # Upscaled bands.
    "TreeHeightEstimator",  # Canopy height.
    "VehicleDetector",  # Vehicles.
    "WaterDetector",  # Water surfaces.
    "YieldPredictor",  # Crop yield.
    "predict",  # Run any model.
    "task_api",  # Task API of a model.
    "write_result",  # Write a result.
]  # End of the export list.

# =============================================================================
# End of module src/unbihexium/ai/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
