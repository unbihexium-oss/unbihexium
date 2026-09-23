# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model inference service for REST API."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.ai.detection import BuildingDetector, ShipDetector
from unbihexium.ai.segmentation import WaterDetector
from unbihexium.core.raster import Raster
from unbihexium.zoo import list_models


class ModelInferenceService:
    """Service for running model inference.

    Provides a unified interface for running inference with different
    model types through the REST API.
    """

    def __init__(self) -> None:
        """Initialize inference service."""
        self._loaded_models: dict[str, Any] = {}

    def list_available_models(self) -> list[dict[str, str]]:
        """List available models for inference.

        Returns:
            List of model information dictionaries.
        """
        models = []

        # Detection models
        models.extend(
            [
                {
                    "model_id": "ship_detector_tiny",
                    "task": "detection",
                    "description": "Ship detection",
                },
                {
                    "model_id": "building_detector_tiny",
                    "task": "detection",
                    "description": "Building detection",
                },
            ]
        )

        # Segmentation models
        models.extend(
            [
                {
                    "model_id": "water_detector_tiny",
                    "task": "segmentation",
                    "description": "Water segmentation",
                },
            ]
        )

        return models

    def run_detection(
        self,
        model_id: str,
        image_data: list[list[list[float]]] | NDArray[np.floating[Any]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Run object detection.

        Args:
            model_id: Detection model identifier.
            image_data: Image data as [C, H, W] array.
            threshold: Detection confidence threshold.

        Returns:
            Detection results dictionary.
        """
        # Convert to numpy if needed
        if isinstance(image_data, list):
            data = np.array(image_data, dtype=np.float32)
        else:
            data = image_data

        # Create raster from data
        raster = Raster.from_array(data, crs="EPSG:4326")

        # Select detector
        if "ship" in model_id.lower():
            detector = ShipDetector(threshold=threshold)
        elif "building" in model_id.lower():
            detector = BuildingDetector(threshold=threshold)
        else:
            detector = ShipDetector(threshold=threshold)

        # Run inference
        result = detector.predict(raster)

        return {
            "model_id": result.model_id,
            "count": result.count,
            "detections": [
                {
                    "x1": d.bbox[0],
                    "y1": d.bbox[1],
                    "x2": d.bbox[2],
                    "y2": d.bbox[3],
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in result.detections
            ],
        }

    def run_segmentation(
        self,
        model_id: str,
        image_data: list[list[list[float]]] | NDArray[np.floating[Any]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Run semantic segmentation.

        Args:
            model_id: Segmentation model identifier.
            image_data: Image data as [C, H, W] array.
            threshold: Segmentation threshold.

        Returns:
            Segmentation results dictionary.
        """
        # Convert to numpy if needed
        if isinstance(image_data, list):
            data = np.array(image_data, dtype=np.float32)
        else:
            data = image_data

        # Create raster from data
        raster = Raster.from_array(data, crs="EPSG:4326")

        # Select segmenter
        segmenter = WaterDetector(threshold=threshold)

        # Run inference
        result = segmenter.predict(raster)

        return {
            "model_id": result.model_id,
            "mask_shape": list(result.mask.shape),
            "classes": result.classes,
        }

    def run_inference(
        self,
        model_id: str,
        data: list[list[float]] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run generic inference.

        Args:
            model_id: Model identifier.
            data: Input data.
            parameters: Model parameters.

        Returns:
            Inference results dictionary.
        """
        params = parameters or {}

        # Determine task type from model_id
        if "detector" in model_id.lower() or "detection" in model_id.lower():
            # Need 3D data for detection
            if data is None:
                return {"error": "Image data required for detection"}

            # Reshape flat data to image if needed
            return self.run_detection(
                model_id,
                [data] if len(data) > 0 and isinstance(data[0], (int, float)) else data,
                threshold=params.get("threshold", 0.5),
            )

        elif "segment" in model_id.lower() or "water" in model_id.lower():
            if data is None:
                return {"error": "Image data required for segmentation"}

            return self.run_segmentation(
                model_id,
                [data] if len(data) > 0 and isinstance(data[0], (int, float)) else data,
                threshold=params.get("threshold", 0.5),
            )

        else:
            return {
                "model_id": model_id,
                "error": f"Unknown model type: {model_id}",
            }
