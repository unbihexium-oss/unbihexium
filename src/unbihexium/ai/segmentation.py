"""Segmentation pipelines for satellite imagery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.raster import Raster
from unbihexium.core.pipeline import Pipeline, PipelineConfig
from unbihexium.registry.pipelines import PipelineRegistry


@dataclass
class SegmentationResult:
    """Result of semantic segmentation."""

    mask: NDArray[np.floating[Any]] | None = None
    class_names: list[str] = field(default_factory=list)
    source: str = ""
    model_id: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def get_class_mask(self, class_id: int) -> NDArray[np.floating[Any]] | None:
        if self.mask is None:
            return None
        return (self.mask == class_id).astype(np.float32)

    def to_raster(self, metadata: Any = None) -> Raster:
        return Raster.from_array(self.mask if self.mask is not None else np.array([]))


class SemanticSegmenter:
    """Base semantic segmentation model."""

    def __init__(
        self,
        model_id: str = "segmenter",
        class_names: list[str] | None = None,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> None:
        self.model_id = model_id
        self.class_names = class_names or ["background", "foreground"]
        self.tile_size = tile_size
        self.overlap = overlap

    def predict(self, raster: Raster) -> SegmentationResult:
        """Run segmentation on a raster."""
        raster.load()
        if raster.data is None:
            return SegmentationResult(source=str(raster.source), model_id=self.model_id)

        mask = self._segment_tiles(raster)
        return SegmentationResult(
            mask=mask,
            class_names=self.class_names,
            source=str(raster.source),
            model_id=self.model_id,
        )

    def _segment_tiles(self, raster: Raster) -> NDArray[np.floating[Any]]:
        """Segment using tiled inference."""
        from unbihexium.core.tile import TileGrid

        grid = TileGrid.from_raster(raster, tile_size=self.tile_size, overlap=self.overlap)

        # Create output mask
        output = np.zeros((raster.height, raster.width), dtype=np.float32)

        for tile in grid.tiles(raster):
            tile_mask = self._segment_single(tile.data)
            row_off, col_off = tile.offset
            h, w = tile.height, tile.width
            output[row_off : row_off + h, col_off : col_off + w] = tile_mask[:h, :w]

        return output

    def _segment_single(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Segment a single tile."""
        # Placeholder - actual model inference
        _, h, w = data.shape
        return np.zeros((h, w), dtype=np.float32)


class ChangeDetector(SemanticSegmenter):
    """Bi-temporal change detection."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            model_id="change_detector",
            class_names=["no_change", "change"],
        )
        self.threshold = threshold

    def predict_pair(self, raster1: Raster, raster2: Raster) -> SegmentationResult:
        """Detect changes between two rasters."""
        raster1.load()
        raster2.load()

        if raster1.data is None or raster2.data is None:
            return SegmentationResult(model_id=self.model_id)

        # Simple difference-based change detection
        diff = np.abs(raster1.data.mean(axis=0) - raster2.data.mean(axis=0))
        mask = (diff > self.threshold).astype(np.float32)

        return SegmentationResult(
            mask=mask,
            class_names=self.class_names,
            model_id=self.model_id,
        )


class WaterDetector(SemanticSegmenter):
    """Water surface detection."""

    def __init__(self) -> None:
        super().__init__(
            model_id="water_detector",
            class_names=["land", "water"],
        )


class CropDetector(SemanticSegmenter):
    """Crop detection and classification."""

    def __init__(self, crop_classes: list[str] | None = None) -> None:
        classes = crop_classes or ["background", "crop"]
        super().__init__(
            model_id="crop_detector",
            class_names=classes,
        )


class GreenhouseDetector(SemanticSegmenter):
    """Greenhouse detection."""

    def __init__(self) -> None:
        super().__init__(
            model_id="greenhouse_detector",
            class_names=["background", "greenhouse"],
        )


# Register segmentation pipelines
@PipelineRegistry.register(
    pipeline_id="change_detection",
    name="Change Detection Pipeline",
    description="Bi-temporal change detection",
    domains=["ai", "change"],
)
def create_change_detection_pipeline(**kwargs: Any) -> Pipeline:
    config = PipelineConfig(
        pipeline_id="change_detection",
        name="Change Detection",
        parameters=kwargs,
    )
    pipeline = Pipeline(config)
    detector = ChangeDetector(threshold=kwargs.get("threshold", 0.5))

    def detect_step(inputs: dict[str, Any]) -> dict[str, Any]:
        raster1 = Raster.from_file(inputs["input1"])
        raster2 = Raster.from_file(inputs["input2"])
        result = detector.predict_pair(raster1, raster2)
        return {"result": result}

    pipeline.add_step(detect_step)
    return pipeline


@PipelineRegistry.register(
    pipeline_id="water_detection",
    name="Water Detection Pipeline",
    description="Detect water surfaces in satellite imagery",
    domains=["ai", "water"],
)
def create_water_detection_pipeline(**kwargs: Any) -> Pipeline:
    config = PipelineConfig(
        pipeline_id="water_detection",
        name="Water Detection",
        parameters=kwargs,
    )
    pipeline = Pipeline(config)
    detector = WaterDetector()

    def detect_step(inputs: dict[str, Any]) -> dict[str, Any]:
        raster = Raster.from_file(inputs["input"])
        result = detector.predict(raster)
        return {"result": result, "input": inputs["input"]}

    pipeline.add_step(detect_step)
    return pipeline
