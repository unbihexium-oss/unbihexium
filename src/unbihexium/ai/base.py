# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/base.py
# Title       : Common base of the task APIs
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; running a model needs
#               PyTorch or ONNX Runtime
# =============================================================================
#
# Abstract
# --------
# ZooTask is the base class of the task APIs (ObjectDetector,
# SemanticSegmenter, DenseRegressor, SuperResolution, ...). It selects the
# model, opens it lazily on first use, converts inputs to arrays and keeps
# their georeferencing:
#
#   model     catalogue family or model id (starter weights), a trained
#             checkpoint (.pt) or an ONNX export (.onnx)
#   variant   size variant for catalogue families (default: base)
#   weights   trained checkpoint or ONNX file; overrides `model`
#
# Starter weights are not trained, so predictions of catalogue models are
# meaningless until the model is trained (see docs/model_zoo/training.md);
# pass the resulting checkpoint as `weights`.
#
# register_task_pipeline adds a task API to the pipeline registry, so that
# `unbihexium pipeline run <id>` can run it on files.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values and callables.
from typing import Any, Callable

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Tiled inference.
from unbihexium.ai.inference import Predictor

# Georeferencing helpers.
from unbihexium.ai.results import IDENTITY_TRANSFORM, georeference

# Pipelines.
from unbihexium.core.pipeline import Pipeline, PipelineConfig

# Raster container.
from unbihexium.core.raster import Raster

# Pipeline registry.
from unbihexium.registry.pipelines import PipelineRegistry

# Catalogue lookups.
from unbihexium.zoo.catalog import Task, Variant, get_spec, parse_model_id

# Input image with georeferencing and source description.
Prepared = tuple[NDArray[np.float32], str, tuple[float, ...], str]


# Common base of the task APIs.
class ZooTask:
    # Catalogue family used when no model is given.
    default_model = "object_detector"
    # Tasks accepted by this API.
    tasks: tuple[Task, ...] = ()

    # Configure the model; nothing is loaded yet.
    def __init__(
        self,  # The task API.
        model: Any = None,  # Family, model id, checkpoint, ONNX file or ZooModel.
        variant: str | None = None,  # Size variant of catalogue families.
        weights: str | Path | None = None,  # Trained checkpoint or ONNX file.
        device: str = "cpu",  # Torch device.
        backend: str = "auto",  # auto, torch or onnx.
        tile_size: int | None = None,  # Tile size of the inference.
        overlap: float = 0.25,  # Tile overlap fraction.
        batch_size: int = 4,  # Tiles per forward pass.
    ) -> None:  # The constructor returns nothing.
        # Trained weights take precedence over the model name.
        self.source = weights if weights is not None else (model or self.default_model)
        # Variant of catalogue families.
        self.variant = variant
        # Torch device.
        self.device = device
        # Inference backend.
        self.backend = backend
        # Tile size.
        self.tile_size = tile_size
        # Tile overlap.
        self.overlap = overlap
        # Tiles per forward pass.
        self.batch_size = batch_size
        # Predictor, opened on first use.
        self._predictor: Predictor | None = None

    # Model id, known without loading for catalogue names.
    @property
    def model_id(self) -> str:
        # Loaded models know their id.
        if self._predictor is not None:
            # Id from the configuration.
            return self._predictor.model_id
        # Model objects carry their configuration.
        config = getattr(self.source, "config", None)
        # Id of the object.
        if config is not None:
            # Id from its configuration.
            return str(config.model_id)
        # Catalogue names can be resolved without loading.
        if isinstance(self.source, str) and not Path(self.source).suffix:
            # Family and variant of the name.
            family, parsed = parse_model_id(self.source)
            # Explicit variant wins over the suffix.
            return get_spec(family).model_id(Variant(self.variant) if self.variant else parsed)
        # Files must be opened.
        return self.predictor.model_id

    # Predictor of the model, opened on first use.
    @property
    def predictor(self) -> Predictor:
        # Open the model once.
        if self._predictor is None:
            # Tiled predictor.
            self._predictor = Predictor(
                self.source,  # Model.
                variant=self.variant,  # Variant.
                device=self.device,  # Device.
                backend=self.backend,  # Backend.
                tile_size=self.tile_size,  # Tile size.
                overlap=self.overlap,  # Overlap.
                batch_size=self.batch_size,  # Batch size.
            )  # End of the predictor.
            # Reject models of another task.
            if self.tasks and self._predictor.config.task not in self.tasks:
                # Names of the accepted tasks.
                accepted = ", ".join(t.value for t in self.tasks)
                # Explain the mismatch.
                raise ValueError(
                    f"{type(self).__name__} needs a {accepted} model, "
                    f"{self._predictor.model_id} is {self._predictor.config.task.value}"
                )  # End of the error.
        # Return the predictor.
        return self._predictor

    # Class, target or band names of the model.
    @property
    def outputs(self) -> list[str]:
        # From the loaded configuration.
        return list(self.predictor.config.outputs)

    # Convert an input to an array with its georeferencing.
    @staticmethod
    def prepare(image: Raster | NDArray[Any] | str | Path) -> Prepared:
        # Paths are read as rasters.
        if isinstance(image, (str, Path)):
            # Read the file.
            image = Raster.from_file(image)
        # Rasters provide data and georeferencing.
        if isinstance(image, Raster):
            # Load lazy rasters.
            image.load()
            # Data must be present.
            if image.data is None:
                # Explain the problem.
                raise ValueError("the raster has no data")
            # CRS and transform.
            crs, transform = georeference(image)
            # Missing values become NaN.
            data = np.asarray(image.data, dtype=np.float32)
            # No-data value of the raster.
            nodata = image.metadata.nodata if image.metadata else None
            # Mark missing pixels.
            if nodata is not None and np.isfinite(nodata):
                # Pixels where every band equals the no-data value.
                data = np.where(np.all(data == nodata, axis=0), np.nan, data).astype(np.float32)
            # Return the array with its georeferencing.
            return data, crs, transform, str(image.source or "")
        # Plain arrays have pixel coordinates.
        data = np.asarray(image, dtype=np.float32)
        # Add a band axis to single-band arrays.
        if data.ndim == 2:
            # One band.
            data = data[None]
        # Return the array with the identity transform.
        return data, "EPSG:4326", IDENTITY_TRANSFORM, ""


# Register a pipeline that runs a task API on files.
def register_task_pipeline(
    pipeline_id: str,  # Registry id.
    name: str,  # Human-readable name.
    description: str,  # Description.
    domains: list[str],  # Capability domains.
    factory: Callable[..., ZooTask],  # Creates the task API from the parameters.
    inputs: tuple[str, ...] = ("input",),  # Keys of the input files.
    method: str = "predict",  # Method of the task API to call.
) -> Callable[..., Pipeline]:  # The registered pipeline factory.
    # Pipeline factory registered under the id.
    @PipelineRegistry.register(pipeline_id, name, description, domains)
    def create(**kwargs: Any) -> Pipeline:
        # Pipeline configuration with the parameters.
        config = PipelineConfig(pipeline_id=pipeline_id, name=name, parameters=kwargs)
        # Empty pipeline.
        pipeline = Pipeline(config)
        # Task API configured with the parameters.
        task = factory(**kwargs)
        # Keep the task API, so that callers can tell which model runs.
        pipeline.task = task  # type: ignore[attr-defined]

        # Single step: read the inputs and run the task.
        def run_task(values: dict[str, Any]) -> dict[str, Any]:
            # Read every input file.
            rasters = [Raster.from_file(values[key]) for key in inputs]
            # Run the task API.
            result = getattr(task, method)(*rasters)
            # Keep the result object for callers such as the command line.
            pipeline.last_result = result  # type: ignore[attr-defined]
            # Keep the inputs next to the result.
            return {"result": result, **{key: values[key] for key in inputs}}

        # Add the step.
        pipeline.add_step(run_task)
        # Return the pipeline.
        return pipeline

    # Return the factory.
    return create


# =============================================================================
# End of module src/unbihexium/ai/base.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
