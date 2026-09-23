# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/detection.py
# Title       : Object detection in Earth observation imagery
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
# ObjectDetector runs a CenterNet detector of the model zoo over an image of
# any size with tiled inference and returns a DetectionResult: boxes in
# pixel coordinates and, for georeferenced rasters, in map coordinates,
# with class names and scores, exportable as GeoJSON.
#
# The subclasses select the catalogue family of common targets: ships,
# ships in SAR amplitude images, buildings, aircraft, vehicles, greenhouses
# and centre-pivot fields among others. Every detector accepts a trained
# checkpoint or ONNX file through `weights`:
#
#   detector = ShipDetector(weights="runs/ship_detector_base/best.pt")
#   result = detector.predict(Raster.from_file("scene.tif"))
#   geojson = result.to_geojson()
#
# The starter weights of the catalogue are untrained; without `weights` the
# detections are meaningless.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Array type annotations.
from numpy.typing import NDArray

# Task API base and pipeline registration.
from unbihexium.ai.base import ZooTask, register_task_pipeline

# Result records and georeferencing.
from unbihexium.ai.results import Detection, DetectionResult, pixel_to_map

# Raster container.
from unbihexium.core.raster import Raster

# Task enumeration.
from unbihexium.zoo.catalog import Task


# Object detector backed by a model zoo CenterNet.
class ObjectDetector(ZooTask):
    # Generic multi-class detector of the catalogue.
    default_model = "object_detector"
    # Detection models only.
    tasks = (Task.DETECTION,)

    # Configure the detector.
    def __init__(
        self,  # The detector.
        model: Any = None,  # Family, model id, checkpoint, ONNX file or ZooModel.
        threshold: float = 0.5,  # Minimum detection score.
        iou_threshold: float = 0.5,  # Overlap for non-maximum suppression.
        max_detections: int = 1000,  # Maximum boxes per image.
        **kwargs: Any,  # Options of ZooTask (variant, weights, device, ...).
    ) -> None:  # The constructor returns nothing.
        # Model selection and inference options.
        super().__init__(model, **kwargs)
        # Minimum detection score.
        self.threshold = threshold
        # Overlap for non-maximum suppression.
        self.iou_threshold = iou_threshold
        # Maximum boxes per image.
        self.max_detections = max_detections

    # Detect objects in an image, raster or raster file.
    def predict(self, image: Raster | NDArray[Any] | str | Path) -> DetectionResult:
        # Array and georeferencing.
        data, crs, transform, source = self.prepare(image)
        # Tiled detection.
        boxes, scores, classes = self.predictor.detect(
            data,  # Image.
            threshold=self.threshold,  # Score threshold.
            max_detections=self.max_detections,  # Box limit.
            iou_threshold=self.iou_threshold,  # Suppression overlap.
        )  # End of the detection.
        # Class names of the model.
        names = self.outputs
        # Detection records.
        detections = []
        # Convert every box.
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            # Upper-left corner in map coordinates.
            mx1, my1 = pixel_to_map(transform, x1, y1)
            # Lower-right corner in map coordinates.
            mx2, my2 = pixel_to_map(transform, x2, y2)
            # Box in map coordinates, ordered from minimum to maximum.
            geo = (min(mx1, mx2), min(my1, my2), max(mx1, mx2), max(my1, my2))
            # Detection with pixel and map boxes.
            detections.append(
                Detection(  # Detection record.
                    bbox=(float(x1), float(y1), float(x2), float(y2)),  # Pixel box.
                    confidence=float(score),  # Score.
                    class_id=int(cls),  # Class index.
                    class_name=names[int(cls)],  # Class name.
                    geo_bbox=geo,  # Map box.
                )  # End of the detection.
            )  # End of the append.
        # Result with metadata.
        return DetectionResult(detections, source, self.model_id, crs)


# Ships in optical imagery.
class ShipDetector(ObjectDetector):
    # Catalogue family.
    default_model = "ship_detector"


# Ships in SAR amplitude imagery (VV, VH).
class SARShipDetector(ObjectDetector):
    # Catalogue family.
    default_model = "sar_ship_detector"


# Buildings in very high resolution imagery.
class BuildingDetector(ObjectDetector):
    # Catalogue family.
    default_model = "building_detector"


# Aircraft on airfields.
class AircraftDetector(ObjectDetector):
    # Catalogue family.
    default_model = "aircraft_detector"


# Cars, trucks and buses.
class VehicleDetector(ObjectDetector):
    # Catalogue family.
    default_model = "vehicle_detector"


# Greenhouses.
class GreenhouseDetector(ObjectDetector):
    # Catalogue family.
    default_model = "greenhouse_detector"


# Crop fields and parcels.
class CropDetector(ObjectDetector):
    # Catalogue family.
    default_model = "crop_detector"


# Centre-pivot irrigation fields.
class PivotDetector(ObjectDetector):
    # Catalogue family.
    default_model = "pivot_inventory"


# Active fires and burn scars.
class FireDetector(ObjectDetector):
    # Catalogue family.
    default_model = "fire_monitor"


# Pipeline: ship detection on a raster file.
create_ship_detection_pipeline = register_task_pipeline(
    "ship_detection",  # Registry id.
    "Ship Detection Pipeline",  # Name.
    "Detect ships in optical satellite imagery",  # Description.
    ["ai", "maritime"],  # Domains.
    ShipDetector,  # Task API.
)  # End of the registration.

# Pipeline: building detection on a raster file.
create_building_detection_pipeline = register_task_pipeline(
    "building_detection",  # Registry id.
    "Building Detection Pipeline",  # Name.
    "Detect buildings in very high resolution imagery",  # Description.
    ["ai", "urban"],  # Domains.
    BuildingDetector,  # Task API.
)  # End of the registration.


# =============================================================================
# End of module src/unbihexium/ai/detection.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
