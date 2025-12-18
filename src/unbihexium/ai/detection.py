"""Object detection pipelines for satellite imagery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.model import ModelWrapper
from unbihexium.core.pipeline import Pipeline, PipelineConfig
from unbihexium.core.raster import Raster
from unbihexium.registry.pipelines import PipelineRegistry


@dataclass
class Detection:
    """A single object detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    geo_bbox: tuple[float, float, float, float] | None = None  # lon1, lat1, lon2, lat2


@dataclass
class DetectionResult:
    """Result of object detection."""

    detections: list[Detection] = field(default_factory=list)
    source: str = ""
    model_id: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(detections=filtered, source=self.source, model_id=self.model_id)

    def to_geojson(self) -> dict[str, Any]:
        features = []
        for det in self.detections:
            if det.geo_bbox:
                x1, y1, x2, y2 = det.geo_bbox
                coords = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                        },
                        "geometry": {"type": "Polygon", "coordinates": coords},
                    }
                )
        return {"type": "FeatureCollection", "features": features}


class ObjectDetector:
    """Base object detector."""

    def __init__(
        self,
        model_id: str = "generic_detector",
        class_names: list[str] | None = None,
        threshold: float = 0.5,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> None:
        self.model_id = model_id
        self.class_names = class_names or ["object"]
        self.threshold = threshold
        self.tile_size = tile_size
        self.overlap = overlap
        self._model: ModelWrapper | None = None

    def load_model(self, weights_path: Path | None = None) -> None:
        """Load the detection model."""
        # Model loading logic
        pass

    def predict(self, raster: Raster) -> DetectionResult:
        """Run detection on a raster."""
        raster.load()
        if raster.data is None:
            return DetectionResult(source=str(raster.source), model_id=self.model_id)

        detections = self._detect_tiles(raster)
        return DetectionResult(
            detections=detections,
            source=str(raster.source),
            model_id=self.model_id,
        )

    def _detect_tiles(self, raster: Raster) -> list[Detection]:
        """Detect objects using tiled inference."""
        from unbihexium.core.tile import TileGrid

        grid = TileGrid.from_raster(raster, tile_size=self.tile_size, overlap=self.overlap)
        detections = []

        for tile in grid.tiles(raster):
            tile_detections = self._detect_single(tile.data)
            # Adjust bounding boxes to global coordinates
            for det in tile_detections:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (
                    x1 + tile.col_offset,
                    y1 + tile.row_offset,
                    x2 + tile.col_offset,
                    y2 + tile.row_offset,
                )
            detections.extend(tile_detections)

        return self._nms(detections)

    def _detect_single(self, data: NDArray[np.floating[Any]]) -> list[Detection]:
        """Detect objects in a single tile."""
        # Placeholder - actual model inference would go here
        return []

    def _nms(self, detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self._iou(best.bbox, d.bbox) < iou_threshold]

        return keep

    @staticmethod
    def _iou(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
        """Calculate intersection over union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0


class ShipDetector(ObjectDetector):
    """Ship detection in satellite imagery."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            model_id="ship_detector",
            class_names=["ship"],
            threshold=threshold,
        )


class BuildingDetector(ObjectDetector):
    """Building detection in satellite imagery."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            model_id="building_detector",
            class_names=["building"],
            threshold=threshold,
        )


class AircraftDetector(ObjectDetector):
    """Aircraft detection in satellite imagery."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            model_id="aircraft_detector",
            class_names=["aircraft"],
            threshold=threshold,
        )


class VehicleDetector(ObjectDetector):
    """Vehicle detection in satellite imagery."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            model_id="vehicle_detector",
            class_names=["vehicle", "car", "truck"],
            threshold=threshold,
        )


# Register detection pipelines
@PipelineRegistry.register(
    pipeline_id="ship_detection",
    name="Ship Detection Pipeline",
    description="Detect ships in satellite imagery",
    domains=["ai", "maritime"],
)
def create_ship_detection_pipeline(**kwargs: Any) -> Pipeline:
    config = PipelineConfig(
        pipeline_id="ship_detection",
        name="Ship Detection",
        parameters=kwargs,
    )
    pipeline = Pipeline(config)
    detector = ShipDetector(threshold=kwargs.get("threshold", 0.5))

    def detect_step(inputs: dict[str, Any]) -> dict[str, Any]:
        raster = Raster.from_file(inputs["input"])
        result = detector.predict(raster)
        return {"result": result, "input": inputs["input"]}

    pipeline.add_step(detect_step)
    return pipeline
