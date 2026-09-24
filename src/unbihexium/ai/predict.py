# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/predict.py
# Title       : Run any model zoo model and write its result
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
# predict() opens a model, picks the task API that matches its task and runs
# it on an image. write_result() stores the result in the natural format of
# the task:
#
#   detection                GeoJSON FeatureCollection of boxes
#   segmentation, change     single-band GeoTIFF of class labels
#   dense regression,        multi-band float32 GeoTIFF
#   spectral index,
#   enhancement,
#   super-resolution
#   scene regression         JSON document with one value per output
#
# The `unbihexium predict` command is a thin wrapper around these two
# functions.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Keyword arguments of the task APIs.
import inspect

# JSON output.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Task API base.
from unbihexium.ai.base import ZooTask

# Detection API.
from unbihexium.ai.detection import ObjectDetector

# Regression APIs.
from unbihexium.ai.regression import DenseRegressor, SceneRegressor

# Result records.
from unbihexium.ai.results import (
    DetectionResult,  # Boxes.
    EnhancementResult,  # Enhanced bands.
    RegressionResult,  # Values.
    SegmentationResult,  # Class maps.
    SuperResolutionResult,  # Upscaled bands.
)  # End of the result imports.

# Segmentation and change detection APIs.
from unbihexium.ai.segmentation import ChangeDetector, SemanticSegmenter

# Enhancement and super-resolution APIs.
from unbihexium.ai.super_resolution import Enhancer, SuperResolution

# Task enumeration.
from unbihexium.zoo.catalog import Task

# Task API class per task.
TASK_APIS: dict[Task, type[ZooTask]] = {
    Task.DETECTION: ObjectDetector,  # Boxes.
    Task.SEGMENTATION: SemanticSegmenter,  # Class maps.
    Task.CHANGE_DETECTION: ChangeDetector,  # Change maps.
    Task.DENSE_REGRESSION: DenseRegressor,  # Value maps.
    Task.SPECTRAL_INDEX: DenseRegressor,  # Index maps.
    Task.SCENE_REGRESSION: SceneRegressor,  # Scene values.
    Task.ENHANCEMENT: Enhancer,  # Enhanced bands.
    Task.SUPER_RESOLUTION: SuperResolution,  # Upscaled bands.
}  # End of the mapping.

# Any result record of the task APIs.
Result = (
    DetectionResult  # Boxes.
    | SegmentationResult  # Class maps.
    | RegressionResult  # Values.
    | EnhancementResult  # Enhanced bands.
    | SuperResolutionResult  # Upscaled bands.
)


# Options of ZooTask.__init__.
_BASE_OPTIONS = ("variant", "weights", "device", "backend", "tile_size", "overlap", "batch_size")


# Task API for a model, with the model already opened.
def task_api(model: Any, **options: Any) -> ZooTask:
    # Generic API used to open the model and read its task.
    probe = ZooTask(model, **{k: v for k, v in options.items() if k in _BASE_OPTIONS})
    # Task of the model.
    task = probe.predictor.config.task
    # API class of the task.
    api_cls = TASK_APIS[task]
    # Keyword arguments of the API and of its base class.
    accepted = set(inspect.signature(api_cls.__init__).parameters) | set(_BASE_OPTIONS)
    # Options understood by that API; others (for example a detection threshold
    # passed to a regression model) are ignored.
    api = api_cls(model, **{k: v for k, v in options.items() if k in accepted})
    # Reuse the opened predictor.
    api._predictor = probe.predictor
    # Return the API.
    return api


# Run a model on an image, raster or raster file.
def predict(model: Any, image: Any, **options: Any) -> Result:
    # Task API with the opened model.
    api = task_api(model, **options)
    # Super-resolution uses enhance; every other API uses predict.
    return api.predict(image)  # type: ignore[attr-defined]


# File extensions accepted for each kind of result.
VECTOR_SUFFIXES = (".geojson", ".json")
# Scene values are written as JSON.
JSON_SUFFIXES = (".json",)
# Rasters are written as GeoTIFF.
RASTER_SUFFIXES = (".tif", ".tiff")


# Extensions that suit the results of a task.
def suffixes_for(task: Task) -> tuple[str, ...]:
    # Detections are vector features.
    if task is Task.DETECTION:
        # GeoJSON.
        return VECTOR_SUFFIXES
    # Scene-level values.
    if task is Task.SCENE_REGRESSION:
        # JSON.
        return JSON_SUFFIXES
    # Every other result is a raster.
    return RASTER_SUFFIXES


# Refuse an output file name whose extension does not suit the task.
def check_output_path(task: Task, path: str | Path) -> None:
    # Extensions of the task.
    allowed = suffixes_for(task)
    # Extension in lower case.
    suffix = Path(path).suffix.lower()
    # Reject other extensions instead of writing, for example, GeoJSON to .tif.
    if suffix not in allowed:
        # Explain the accepted names.
        raise ValueError(
            f"{Path(path).name}: a {task.value} result is written as "
            f"{' or '.join(allowed)}, not {suffix or 'a file without extension'}"
        )  # End of the error.


# Write a result to a file in the natural format of its task.
def write_result(result: Result, path: str | Path) -> Path:
    # Destination file.
    path = Path(path)
    # Kind of result: vector, scene values or raster.
    if isinstance(result, DetectionResult):
        # GeoJSON.
        allowed = VECTOR_SUFFIXES
    # Scene values.
    elif isinstance(result, RegressionResult) and not result.is_dense:
        # JSON.
        allowed = JSON_SUFFIXES
    # Rasters.
    else:
        # GeoTIFF.
        allowed = RASTER_SUFFIXES
    # The extension must suit the result.
    if path.suffix.lower() not in allowed:
        # Explain the accepted names.
        raise ValueError(
            f"{path.name}: this result is written as {' or '.join(allowed)}, "
            f"not {path.suffix or 'a file without extension'}"
        )  # End of the error.
    # Create the parent directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Detections as GeoJSON.
    if isinstance(result, DetectionResult):
        # Write the feature collection.
        path.write_text(json.dumps(result.to_geojson(), indent=2), encoding="utf-8")
        # Return the path.
        return path
    # Scene values as JSON.
    if isinstance(result, RegressionResult) and not result.is_dense:
        # Write the values.
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        # Return the path.
        return path
    # Class maps and value maps as rasters.
    if isinstance(result, (SegmentationResult, RegressionResult)):
        # Raster of the result.
        raster = result.to_raster()
    # Enhanced and upscaled bands.
    else:
        # Raster of the result.
        raster = result.raster
    # Every remaining result has a raster.
    if raster is None:
        # Explain the problem.
        raise ValueError("the result has no raster to write")
    # Write the GeoTIFF.
    return raster.to_file(path)


# =============================================================================
# End of module src/unbihexium/ai/predict.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
