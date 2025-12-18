"""Super-resolution pipeline for satellite imagery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unbihexium.core.raster import Raster
from unbihexium.core.pipeline import Pipeline, PipelineConfig
from unbihexium.registry.pipelines import PipelineRegistry


@dataclass
class SuperResolutionResult:
    """Result of super-resolution."""

    raster: Raster | None = None
    scale_factor: int = 2
    source: str = ""
    model_id: str = ""


class SuperResolution:
    """Super-resolution model for satellite imagery."""

    def __init__(
        self,
        model_id: str = "super_resolution",
        scale_factor: int = 2,
        tile_size: int = 256,
    ) -> None:
        self.model_id = model_id
        self.scale_factor = scale_factor
        self.tile_size = tile_size

    def enhance(self, raster: Raster) -> SuperResolutionResult:
        """Enhance raster resolution."""
        raster.load()
        if raster.data is None:
            return SuperResolutionResult(model_id=self.model_id)

        enhanced = self._upscale(raster.data)
        enhanced_raster = Raster.from_array(
            enhanced,
            crs=raster.metadata.crs if raster.metadata else "EPSG:4326",
        )

        return SuperResolutionResult(
            raster=enhanced_raster,
            scale_factor=self.scale_factor,
            source=str(raster.source),
            model_id=self.model_id,
        )

    def _upscale(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Upscale image using the model."""
        from scipy.ndimage import zoom

        # Simple bicubic upscaling as placeholder
        # Real implementation would use neural network
        return zoom(data, (1, self.scale_factor, self.scale_factor), order=3)


# Register super-resolution pipeline
@PipelineRegistry.register(
    pipeline_id="super_resolution",
    name="Super Resolution Pipeline",
    description="Enhance satellite imagery resolution",
    domains=["ai", "imaging"],
)
def create_super_resolution_pipeline(**kwargs: Any) -> Pipeline:
    config = PipelineConfig(
        pipeline_id="super_resolution",
        name="Super Resolution",
        parameters=kwargs,
    )
    pipeline = Pipeline(config)
    sr = SuperResolution(scale_factor=kwargs.get("scale_factor", 2))

    def enhance_step(inputs: dict[str, Any]) -> dict[str, Any]:
        raster = Raster.from_file(inputs["input"])
        result = sr.enhance(raster)
        return {"result": result, "input": inputs["input"]}

    pipeline.add_step(enhance_step)
    return pipeline
