# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/segmentation.py
# Title       : Semantic segmentation and change detection
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and PyTorch or ONNX
#               Runtime
# =============================================================================
#
# Abstract
# --------
# SemanticSegmenter assigns a class to every pixel with a U-Net of the model
# zoo and returns a georeferenced SegmentationResult with the class map,
# optional class probabilities, class fractions and areas.
#
# Thresholds
# ----------
# For two-class models (background and target) a pixel belongs to the target
# class when its probability is at least `threshold` (default 0.5). For
# models with more classes, `threshold` is the minimum probability of the
# winning class; pixels below it are set to the no-data label 255.
#
# ChangeDetector compares two acquisitions of the same area. predict_pair
# stacks the bands of both dates in the order the model expects (first date,
# then second date) and returns the change map.
#
# The subclasses select common catalogue families: water, land cover,
# clouds, crops, SAR flood and oil spill mapping. Pass a trained checkpoint
# through `weights`; the starter weights are untrained.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Task API base and pipeline registration.
from unbihexium.ai.base import ZooTask, register_task_pipeline

# Detectors that earlier releases exported from this module.
from unbihexium.ai.detection import CropDetector, GreenhouseDetector

# Result record.
from unbihexium.ai.results import SegmentationResult

# Label of pixels without a class.
from unbihexium.ai.transforms import IGNORE_INDEX

# Raster container.
from unbihexium.core.raster import Raster

# Task enumeration.
from unbihexium.zoo.catalog import Task

# Names kept for backward compatibility of imports.
__all__ = [
    "ChangeDetector",  # Bi-temporal change detection.
    "CloudMasker",  # Cloud and shadow masks.
    "CropClassifier",  # Crop type maps.
    "CropDetector",  # Crop field detector, re-exported.
    "FloodMapper",  # SAR flood maps.
    "GreenhouseDetector",  # Greenhouse detector, re-exported.
    "LandCoverClassifier",  # Land use and land cover maps.
    "OilSpillDetector",  # SAR oil spill maps.
    "SegmentationResult",  # Result record.
    "SemanticSegmenter",  # Generic segmenter.
    "WaterDetector",  # Water surface maps.
]  # End of the export list.


# Per-pixel classification with a model zoo U-Net.
class SemanticSegmenter(ZooTask):
    # Land use and land cover by default.
    default_model = "lulc_classifier"
    # Segmentation and change detection models.
    tasks = (Task.SEGMENTATION, Task.CHANGE_DETECTION)

    # Configure the segmenter.
    def __init__(
        self,  # The segmenter.
        model: Any = None,  # Family, model id, checkpoint, ONNX file or ZooModel.
        threshold: float | None = 0.5,  # Probability threshold, see the module header.
        return_probabilities: bool = False,  # Keep the class probabilities.
        **kwargs: Any,  # Options of ZooTask (variant, weights, device, ...).
    ) -> None:  # The constructor returns nothing.
        # Model selection and inference options.
        super().__init__(model, **kwargs)
        # Probability threshold.
        self.threshold = threshold
        # Whether to keep the probabilities.
        self.return_probabilities = return_probabilities

    # Class map from class probabilities (K, H, W).
    def labels(self, probabilities: NDArray[np.float32]) -> NDArray[np.uint8]:
        # Pixels without valid input have NaN probabilities.
        invalid = ~np.isfinite(probabilities).all(axis=0)
        # Replace NaN so that argmax is defined.
        p = np.nan_to_num(probabilities, nan=0.0)
        # Two-class models: threshold on the target class.
        if p.shape[0] == 2 and self.threshold is not None:
            # Target where its probability reaches the threshold.
            labels = (p[1] >= self.threshold).astype(np.uint8)
        # More classes: most likely class.
        else:
            # Winning class.
            labels = np.argmax(p, axis=0).astype(np.uint8)
            # Uncertain pixels get the no-data label.
            if self.threshold is not None:
                # Probability of the winning class below the threshold.
                labels[p.max(axis=0) < self.threshold] = IGNORE_INDEX
        # Invalid pixels get the no-data label.
        labels[invalid] = IGNORE_INDEX
        # Return the class map.
        return labels

    # Segment an image, raster or raster file.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> SegmentationResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # Blended class probabilities.
        probabilities = self.predictor.dense(data)
        # Result with the class map.
        return SegmentationResult(
            mask=self.labels(probabilities),  # Class map.
            classes=self.outputs,  # Class names.
            model_id=self.model_id,  # Model.
            probabilities=probabilities if self.return_probabilities else None,  # Probabilities.
            source=source,  # Input.
            crs=crs,  # Coordinate system.
            transform=transform,  # Georeferencing.
            nodata=IGNORE_INDEX,  # No-data label.
        )  # End of the result.


# Bi-temporal change detection.
class ChangeDetector(SemanticSegmenter):
    # Generic change detector of the catalogue.
    default_model = "change_detector"
    # Change detection models only.
    tasks = (Task.CHANGE_DETECTION,)

    # Detect changes between two acquisitions of the same grid.
    def predict_pair(
        self,  # The detector.
        before: Raster | NDArray[Any] | str | Path,  # First acquisition.
        after: Raster | NDArray[Any] | str | Path,  # Second acquisition.
    ) -> SegmentationResult:  # Change map.
        # First date with its georeferencing.
        first, crs, transform, source = self.prepare(before)
        # Second date.
        second, _, _, _ = self.prepare(after)
        # Both dates must share the grid.
        if first.shape != second.shape:
            # Explain the problem.
            raise ValueError(f"the images differ in shape: {first.shape} and {second.shape}")
        # Stack the dates on the band axis.
        stacked = np.concatenate([first, second], axis=0)
        # Georeferenced raster of the stack.
        raster = Raster.from_array(stacked, crs=crs, transform=transform)
        # Remember the sources.
        raster.source = source
        # Run the segmentation on the stack.
        return self.predict(raster)


# Water surfaces.
class WaterDetector(SemanticSegmenter):
    # Catalogue family.
    default_model = "water_surface_detector"


# Land use and land cover.
class LandCoverClassifier(SemanticSegmenter):
    # Catalogue family.
    default_model = "lulc_classifier"


# Clouds and cloud shadows.
class CloudMasker(SemanticSegmenter):
    # Catalogue family.
    default_model = "cloud_mask"


# Crop types.
class CropClassifier(SemanticSegmenter):
    # Catalogue family.
    default_model = "crop_classifier"


# Floods in SAR imagery.
class FloodMapper(SemanticSegmenter):
    # Catalogue family.
    default_model = "sar_flood_detector"


# Oil spills in SAR imagery.
class OilSpillDetector(SemanticSegmenter):
    # Catalogue family.
    default_model = "sar_oil_spill_detector"


# Pipeline: change detection between two raster files.
create_change_detection_pipeline = register_task_pipeline(
    "change_detection",  # Registry id.
    "Change Detection Pipeline",  # Name.
    "Bi-temporal change detection",  # Description.
    ["ai", "change"],  # Domains.
    ChangeDetector,  # Task API.
    inputs=("input1", "input2"),  # Two acquisitions.
    method="predict_pair",  # Pairwise method.
)  # End of the registration.

# Pipeline: water mapping on a raster file.
create_water_detection_pipeline = register_task_pipeline(
    "water_detection",  # Registry id.
    "Water Detection Pipeline",  # Name.
    "Detect water surfaces in satellite imagery",  # Description.
    ["ai", "water"],  # Domains.
    WaterDetector,  # Task API.
)  # End of the registration.


# =============================================================================
# End of module src/unbihexium/ai/segmentation.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
