# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/serving/inference.py
# Title       : Model inference service behind the REST API
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and PyTorch or ONNX
#               Runtime; FastAPI is not needed by this module
# =============================================================================
#
# Abstract
# --------
# ModelInferenceService runs any model of the model zoo (unbihexium.zoo,
# 520 catalogue models and user-registered ones) on an image sent to the
# service and returns a JSON-serialisable summary of the result:
#
#   1. decode_image      nested lists or a base64 NumPy .npy file to a
#                        float32 (bands, rows, cols) array; object, text and
#                        complex arrays and pickles are rejected
#   2. validate          the model must exist, the band count must match
#                        the catalogue, and the image must respect the pixel
#                        and value limits of the service
#   3. predict           the task API of the model (unbihexium.ai.predict.
#                        task_api), kept in a small LRU cache of opened
#                        models; request options are restricted to a
#                        whitelist, so a client can never choose weights
#                        files, devices or backends
#   4. summarise         detection: boxes with pixel and map coordinates,
#                        counts per class, optional GeoJSON;
#                        segmentation and change detection: pixel counts,
#                        fractions and areas per class, optional class map;
#                        dense regression and spectral indices: statistics
#                        per output, optional values; scene regression:
#                        one value per output; enhancement and
#                        super-resolution: output shape and band statistics
#
# NaN and infinite values are not valid JSON; they are returned as null.
# Georeferencing (CRS and affine transform) is optional; with it, boxes
# carry map coordinates and class areas are in squared CRS units.
#
# Errors: unknown models raise UnknownModelError (a KeyError), invalid
# inputs ValueError and inputs over the limits PayloadTooLargeError; the
# REST layer maps them to 404, 422 and 413.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Base64 decoding of NumPy files.
import base64

# In-memory NumPy files.
import io

# Serialise access to the model cache.
import threading

# Timing of predictions.
import time

# Least recently used cache of models.
from collections import OrderedDict

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Model descriptions and input validation.
from unbihexium.registry.models import ModelEntry, ModelRegistry

# Request options passed to the task APIs.
ALLOWED_OPTIONS = ("threshold", "iou_threshold", "max_detections", "tile_size", "overlap")

# Response flags handled by the service itself.
RESPONSE_FLAGS = ("return_mask", "return_values", "return_geojson")


# Unknown model id.
class UnknownModelError(KeyError):
    # No extra behaviour.
    pass


# Input larger than the limits of the service.
class PayloadTooLargeError(ValueError):
    # No extra behaviour.
    pass


# Replace NaN and infinity by None recursively, and NumPy scalars by Python ones.
def json_safe(value: Any) -> Any:
    # Dictionaries.
    if isinstance(value, dict):
        # Convert every value.
        return {str(k): json_safe(v) for k, v in value.items()}
    # Sequences.
    if isinstance(value, (list, tuple)):
        # Convert every item.
        return [json_safe(v) for v in value]
    # Arrays.
    if isinstance(value, np.ndarray):
        # Convert through nested lists.
        return json_safe(value.tolist())
    # NumPy scalars.
    if isinstance(value, np.generic):
        # Python scalar.
        value = value.item()
    # Non-finite floats.
    if isinstance(value, float) and not np.isfinite(value):
        # JSON null.
        return None
    # Everything else is already serialisable.
    return value


# Statistics of the finite values of an array.
def value_statistics(values: NDArray[Any]) -> dict[str, Any]:
    # Values as float64.
    v = np.asarray(values, dtype=np.float64)
    # Finite values only.
    finite = v[np.isfinite(v)]
    # Empty selections have no statistics.
    if finite.size == 0:
        # Only the counts.
        return {"count": 0, "missing": int(v.size), "mean": None, "min": None, "max": None}
    # Statistics.
    return {
        "count": int(finite.size),  # Finite values.
        "missing": int(v.size - finite.size),  # NaN or infinite values.
        "mean": float(finite.mean()),  # Mean.
        "std": float(finite.std()),  # Population standard deviation.
        "min": float(finite.min()),  # Minimum.
        "max": float(finite.max()),  # Maximum.
    }  # End of the statistics.


# Map coordinates of a pixel with affine coefficients (a, b, c, d, e, f).
def _pixel_to_map(transform: tuple[float, ...], col: float, row: float) -> tuple[float, float]:
    # Coefficients.
    a, b, c, d, e, f = transform
    # x = a * col + b * row + c, y = d * col + e * row + f.
    return (a * col + b * row + c, d * col + e * row + f)


# Inference service for the REST API.
class ModelInferenceService:
    # Configure the limits and the execution of models.
    def __init__(
        self,  # The service.
        max_pixels: int = 2048 * 2048,  # Largest image (rows times cols).
        max_values: int = 16 * 1024 * 1024,  # Largest input or returned array (values).
        cache_size: int = 4,  # Opened models kept in memory.
        device: str = "cpu",  # Torch device of the models.
        backend: str = "auto",  # auto, torch or onnx.
        batch_size: int = 4,  # Tiles per forward pass.
    ) -> None:  # The constructor returns nothing.
        # Limits must be positive.
        if min(max_pixels, max_values, cache_size) < 1:
            # Explain the problem.
            raise ValueError("max_pixels, max_values and cache_size must be positive")
        # Pixel limit.
        self.max_pixels = int(max_pixels)
        # Value limit.
        self.max_values = int(max_values)
        # Cache size.
        self.cache_size = int(cache_size)
        # Device.
        self.device = device
        # Backend.
        self.backend = backend
        # Batch size.
        self.batch_size = int(batch_size)
        # Opened task APIs by (model id, options).
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        # Lock of the cache.
        self._lock = threading.Lock()

    # Number of opened models.
    @property
    def loaded_count(self) -> int:
        # Size of the cache.
        return len(self._cache)

    # Descriptions of the available models, optionally filtered.
    def list_available_models(
        self,  # The service.
        task: str | None = None,  # Keep only this task.
        domain: str | None = None,  # Keep only this domain.
        variant: str | None = None,  # Keep only this variant.
    ) -> list[dict[str, Any]]:  # Model descriptions.
        # Zoo models; user descriptions without weights cannot be served.
        entries = ModelRegistry.list_all(task, domain, variant)
        # Keep the servable entries.
        entries = [e for e in entries if e.source != "external"]
        # Descriptions with the catalogue text.
        return [self._describe(e) for e in entries]

    # Description of one model entry.
    @staticmethod
    def _describe(entry: ModelEntry) -> dict[str, Any]:
        # Catalogue description of the family.
        from unbihexium.zoo import get_model

        # Zoo entry for the description text.
        zoo = get_model(entry.model_id)
        # Dictionary of the entry plus the description.
        return {**entry.to_dict(), "description": zoo.spec.description if zoo else entry.name}

    # Entry of a model id; unknown ids raise UnknownModelError.
    def model_entry(self, model_id: str) -> ModelEntry:
        # Registry lookup.
        entry = ModelRegistry.get(model_id)
        # Unknown ids.
        if entry is None:
            # Explain the problem.
            raise UnknownModelError(f"unknown model {model_id!r}; see GET /models")
        # Return the entry.
        return entry

    # Description of a model id.
    def model_info(self, model_id: str) -> dict[str, Any]:
        # Entry and description.
        return self._describe(self.model_entry(model_id))

    # Decode nested lists or a base64 .npy file to a (bands, rows, cols) float32 array.
    def decode_image(self, image: Any = None, npy_base64: str | None = None) -> NDArray[Any]:
        # Exactly one encoding.
        if (image is None) == (npy_base64 is None):
            # Explain the problem.
            raise ValueError("send exactly one of image and image_npy_base64")
        # Base64 NumPy files.
        if npy_base64 is not None:
            # Decode the text strictly.
            try:
                # Raw bytes of the .npy file.
                raw = base64.b64decode(npy_base64, validate=True)
                # Parse without pickles (no code execution).
                array = np.load(io.BytesIO(raw), allow_pickle=False)
            # Malformed input.
            except (ValueError, OSError) as exc:
                # Explain the problem.
                raise ValueError(f"image_npy_base64 is not a valid .npy file: {exc}") from exc
        # Nested lists.
        else:
            # Convert; ragged lists raise ValueError.
            try:
                # Array of the lists.
                array = np.asarray(image)
            # Ragged nesting.
            except ValueError as exc:
                # Explain the problem.
                raise ValueError("image must be a rectangular nested list") from exc
        # Only integer, unsigned, boolean and float data are images.
        if array.dtype.kind not in "biuf":
            # Explain the problem.
            raise ValueError(f"image must be numeric, got data type {array.dtype}")
        # Single bands get a band axis.
        if array.ndim == 2:
            # Add the axis.
            array = array[np.newaxis]
        # Only (bands, rows, cols) arrays are images.
        if array.ndim != 3:
            # Explain the expected layout.
            raise ValueError(f"image must be (bands, rows, cols), got shape {array.shape}")
        # Float32 as the models expect.
        return array.astype(np.float32, copy=False)

    # Check an image against the model and the limits; returns the entry.
    def validate(self, model_id: str, image: NDArray[Any]) -> ModelEntry:
        # Entry of the model.
        entry = self.model_entry(model_id)
        # Pixels of the image.
        pixels = int(image.shape[-1]) * int(image.shape[-2])
        # Pixel limit.
        if pixels > self.max_pixels:
            # Explain the limit.
            raise PayloadTooLargeError(f"image has {pixels} pixels, limit is {self.max_pixels}")
        # Value limit.
        if image.size > self.max_values:
            # Explain the limit.
            raise PayloadTooLargeError(f"image has {image.size} values, limit is {self.max_values}")
        # Infinite values are corrupt data (NaN marks missing pixels).
        if np.isinf(image).any():
            # Explain the problem.
            raise ValueError("image contains infinite values; use NaN or nodata for gaps")
        # Band count and spatial size against the catalogue.
        return ModelRegistry.check_input(entry.model_id, tuple(image.shape))

    # Task API of a model with the given options, from the cache.
    def _api(self, model_id: str, options: dict[str, Any]) -> Any:
        # Cache key.
        key = (model_id, tuple(sorted(options.items())))
        # Look the key up.
        with self._lock:
            # Cached APIs move to the end (most recently used).
            if key in self._cache:
                # Refresh the position.
                self._cache.move_to_end(key)
                # Return the API.
                return self._cache[key]
        # Imported lazily because it needs PyTorch or ONNX Runtime.
        from unbihexium.ai.predict import task_api

        # Open the model outside the lock (it may take a while).
        api = task_api(
            model_id,  # Model id.
            device=self.device,  # Device of the service.
            backend=self.backend,  # Backend of the service.
            batch_size=self.batch_size,  # Batch size of the service.
            **options,  # Whitelisted request options.
        )  # End of the task API.
        # Store it.
        with self._lock:
            # Insert as most recently used.
            self._cache[key] = api
            # Drop the least recently used APIs beyond the cache size.
            while len(self._cache) > self.cache_size:
                # Oldest entry.
                self._cache.popitem(last=False)
        # Return the API.
        return api

    # Run a model and return a JSON-serialisable summary.
    def predict(
        self,  # The service.
        model_id: str,  # Model id or family name (base variant).
        image: NDArray[Any],  # (bands, rows, cols) or (rows, cols) array.
        crs: str | None = None,  # CRS of the image.
        transform: tuple[float, ...] | list[float] | None = None,  # Affine coefficients.
        nodata: float | None = None,  # Value of missing pixels.
        parameters: dict[str, Any] | None = None,  # Options and response flags.
    ) -> dict[str, Any]:  # Summary with model_id, task, input_shape, elapsed_ms, result.
        # Float32 image with a band axis.
        data = self.decode_image(image=image) if not isinstance(image, np.ndarray) else image
        # Band axis for single bands.
        data = data[np.newaxis] if data.ndim == 2 else data
        # Model entry after validation.
        entry = self.validate(model_id, data)
        # Request parameters.
        params = dict(parameters or {})
        # Reject unknown parameters instead of ignoring them silently.
        unknown = sorted(set(params) - set(ALLOWED_OPTIONS) - set(RESPONSE_FLAGS))
        # Report them.
        if unknown:
            # Explain the accepted names.
            raise ValueError(f"unknown parameters {unknown}; accepted: {ALLOWED_OPTIONS}")
        # Options for the task API, without unset values.
        options = {k: params[k] for k in ALLOWED_OPTIONS if params.get(k) is not None}
        # Missing pixels: every band equals the no-data value.
        if nodata is not None and np.isfinite(nodata):
            # Mark them as NaN.
            data = np.where(np.all(data == nodata, axis=0), np.nan, data).astype(np.float32)
        # Affine coefficients, None for pixel coordinates.
        affine = tuple(float(v) for v in transform) if transform is not None else None
        # Validate the transform.
        if affine is not None and len(affine) != 6:
            # Explain the expected form.
            raise ValueError(f"transform needs 6 affine coefficients, got {len(affine)}")
        # Start of the timing.
        start = time.perf_counter()
        # Task API of the model.
        api = self._api(entry.model_id, options)
        # Run the model.
        result = api.predict(data)
        # Elapsed time in milliseconds.
        elapsed = (time.perf_counter() - start) * 1000.0
        # Summary of the result.
        summary = self.summarise(result, affine, crs, params, entry.outputs)
        # Response dictionary.
        response = {
            "model_id": entry.model_id,  # Canonical id.
            "task": entry.task,  # Task.
            "input_shape": list(data.shape),  # Input shape.
            "elapsed_ms": round(elapsed, 3),  # Time.
            "requires_training": entry.requires_training,  # Starter model flag.
            "result": summary,  # Summary.
        }  # End of the response.
        # Plain JSON values.
        return json_safe(response)

    # Refuse to return arrays larger than the value limit.
    def _check_output(self, size: int) -> None:
        # Value limit.
        if size > self.max_values:
            # Explain the limit.
            raise PayloadTooLargeError(f"output has {size} values, limit is {self.max_values}")

    # JSON summary of a task API result.
    def summarise(
        self,  # The service.
        result: Any,  # Result record of a task API.
        transform: tuple[float, ...] | None = None,  # Affine coefficients of the input.
        crs: str | None = None,  # CRS of the input.
        params: dict[str, Any] | None = None,  # Response flags.
        outputs: list[str] | None = None,  # Output names of the model.
    ) -> dict[str, Any]:  # Summary.
        # Result records.
        from unbihexium.ai.results import (
            DetectionResult,  # Boxes.
            RegressionResult,  # Values.
            SegmentationResult,  # Class maps.
        )  # End of the result imports.

        # Response flags.
        flags = params or {}
        # Detections.
        if isinstance(result, DetectionResult):
            # Boxes with map coordinates.
            return self._detections(result, transform, crs, bool(flags.get("return_geojson")))
        # Class maps.
        if isinstance(result, SegmentationResult):
            # Pixel counts, fractions and areas.
            return self._segmentation(result, transform, bool(flags.get("return_mask")))
        # Regression values.
        if isinstance(result, RegressionResult):
            # Statistics or scene values.
            return self._regression(result, bool(flags.get("return_values")))
        # Enhanced or upscaled bands.
        return self._image(result, bool(flags.get("return_values")), outputs or [])

    # Summary of a detection result.
    def _detections(
        self,  # The service.
        result: Any,  # DetectionResult.
        transform: tuple[float, ...] | None,  # Affine coefficients.
        crs: str | None,  # CRS.
        geojson: bool,  # Include a FeatureCollection.
    ) -> dict[str, Any]:  # Summary.
        # Boxes as dictionaries.
        boxes = []
        # Visit every detection.
        for det in result.detections:
            # Pixel box.
            x1, y1, x2, y2 = det.bbox
            # Map box when georeferenced.
            if transform is not None:
                # Upper-left corner.
                mx1, my1 = _pixel_to_map(transform, x1, y1)
                # Lower-right corner.
                mx2, my2 = _pixel_to_map(transform, x2, y2)
                # Ordered box.
                det.geo_bbox = (min(mx1, mx2), min(my1, my2), max(mx1, mx2), max(my1, my2))
            # Without georeferencing there are no map coordinates.
            else:
                # No map box.
                det.geo_bbox = None
            # Dictionary of the box.
            box = {
                "x1": x1,  # Left.
                "y1": y1,  # Top.
                "x2": x2,  # Right.
                "y2": y2,  # Bottom.
                "confidence": det.confidence,  # Score.
                "class_id": det.class_id,  # Class index.
                "class_name": det.class_name,  # Class name.
                "geo_bbox": list(det.geo_bbox) if det.geo_bbox else None,  # Map box.
            }  # End of the box.
            # Collect it.
            boxes.append(box)
        # Summary.
        summary: dict[str, Any] = {
            "count": result.count,  # Number of boxes.
            "counts_by_class": result.counts_by_class(),  # Boxes per class.
            "crs": crs if transform is not None else "pixel",  # Coordinates of geo_bbox.
            "detections": boxes,  # Boxes.
        }  # End of the summary.
        # GeoJSON on request.
        if geojson:
            # CRS of the collection.
            result.crs = crs or "pixel"
            # Map boxes when georeferenced, else pixel boxes.
            summary["geojson"] = result.to_geojson(pixel_coordinates=transform is None)
        # Return the summary.
        return summary

    # Summary of a segmentation or change detection result.
    def _segmentation(
        self,  # The service.
        result: Any,  # SegmentationResult.
        transform: tuple[float, ...] | None,  # Affine coefficients.
        return_mask: bool,  # Include the class map.
    ) -> dict[str, Any]:  # Summary.
        # Class map.
        mask = np.asarray(result.mask)
        # Pixels per class.
        counts = np.bincount(mask[mask != result.nodata].ravel(), minlength=len(result.classes))
        # Valid pixels.
        valid = int(counts.sum())
        # Pixel area in squared CRS units, None without georeferencing.
        area = abs(transform[0] * transform[4] - transform[1] * transform[3]) if transform else None
        # Share of the valid pixels per class.
        fractions = {n: float(counts[i]) / max(valid, 1) for i, n in enumerate(result.classes)}
        # Class names.
        names = list(result.classes)
        # Area per class; unknown without georeferencing.
        areas = None if area is None else {n: float(counts[i]) * area for i, n in enumerate(names)}
        # Summary.
        summary: dict[str, Any] = {
            "shape": list(mask.shape),  # Rows and columns.
            "classes": list(result.classes),  # Class names.
            "class_pixels": {n: int(counts[i]) for i, n in enumerate(result.classes)},  # Counts.
            "class_fractions": fractions,  # Share of the valid pixels.
            "class_areas": areas,  # Areas in squared CRS units, None without georeferencing.
            "nodata_pixels": int(mask.size - valid),  # Unclassified pixels.
            "nodata_value": int(result.nodata),  # Label of unclassified pixels.
        }  # End of the summary.
        # Class map on request.
        if return_mask:
            # Respect the output limit.
            self._check_output(mask.size)
            # Nested lists.
            summary["mask"] = mask.astype(int).tolist()
        # Return the summary.
        return summary

    # Summary of a regression result.
    def _regression(self, result: Any, return_values: bool) -> dict[str, Any]:
        # Output names.
        names = list(result.names)
        # Scene values: one number per output.
        if not result.is_dense:
            # Values by name.
            values = {n: float(v) for n, v in zip(names, np.asarray(result.values).ravel())}
            # Summary.
            return {"outputs": names, "units": list(result.units), "values": values}
        # Dense values: statistics per output.
        stats = {n: value_statistics(result.values[i]) for i, n in enumerate(names)}
        # Summary.
        summary: dict[str, Any] = {
            "outputs": names,  # Output names.
            "units": list(result.units),  # Units.
            "shape": list(np.asarray(result.values).shape),  # Output shape.
            "statistics": stats,  # Statistics per output.
        }  # End of the summary.
        # Values on request.
        if return_values:
            # Respect the output limit.
            self._check_output(int(np.asarray(result.values).size))
            # Nested lists.
            summary["values"] = np.asarray(result.values).tolist()
        # Return the summary.
        return summary

    # Summary of an enhancement or super-resolution result.
    def _image(self, result: Any, return_values: bool, outputs: list[str]) -> dict[str, Any]:
        # Output bands.
        values = np.asarray(result.raster.data) if result.raster is not None else np.zeros(0)
        # Band names from the result, else from the model, else numbered.
        bands = list(getattr(result, "bands", []) or outputs)
        # Numbered names when the counts disagree.
        if len(bands) != len(values):
            # band_1, band_2, ...
            bands = [f"band_{i + 1}" for i in range(len(values))]
        # Summary.
        summary: dict[str, Any] = {
            "shape": list(values.shape),  # Output shape.
            "bands": bands,  # Band names.
            "statistics": {b: value_statistics(values[i]) for i, b in enumerate(bands)},  # Stats.
        }  # End of the summary.
        # Upscaling factor of super-resolution results.
        if hasattr(result, "scale_factor"):
            # Store it.
            summary["scale_factor"] = int(result.scale_factor)
        # Values on request.
        if return_values:
            # Respect the output limit.
            self._check_output(int(values.size))
            # Nested lists.
            summary["values"] = values.tolist()
        # Return the summary.
        return summary

    # Detection with the response layout of earlier releases.
    def run_detection(
        self,  # The service.
        model_id: str,  # Detection model id.
        image_data: Any,  # (bands, rows, cols) image.
        threshold: float = 0.5,  # Score threshold.
    ) -> dict[str, Any]:  # model_id, count and detections.
        # Generic prediction.
        response = self.predict(model_id, image_data, parameters={"threshold": threshold})
        # Only detection models have boxes.
        if response["task"] != "detection":
            # Explain the problem.
            raise ValueError(f"{response['model_id']} is a {response['task']} model")
        # Earlier layout.
        return {
            "model_id": response["model_id"],  # Model.
            "count": response["result"]["count"],  # Boxes.
            "detections": response["result"]["detections"],  # Box list.
        }  # End of the layout.

    # Segmentation with the response layout of earlier releases.
    def run_segmentation(
        self,  # The service.
        model_id: str,  # Segmentation model id.
        image_data: Any,  # (bands, rows, cols) image.
        threshold: float = 0.5,  # Probability threshold.
    ) -> dict[str, Any]:  # model_id, mask_shape, classes and class_fractions.
        # Generic prediction.
        response = self.predict(model_id, image_data, parameters={"threshold": threshold})
        # Only class map models qualify.
        if response["task"] not in ("segmentation", "change_detection"):
            # Explain the problem.
            raise ValueError(f"{response['model_id']} is a {response['task']} model")
        # Summary of the class map.
        result = response["result"]
        # Earlier layout.
        return {
            "model_id": response["model_id"],  # Model.
            "mask_shape": result["shape"],  # Rows and columns.
            "classes": result["classes"],  # Class names.
            "class_fractions": result["class_fractions"],  # Class shares.
        }  # End of the layout.

    # Generic inference on a single-band image, as in earlier releases.
    def run_inference(
        self,  # The service.
        model_id: str,  # Model id.
        data: Any = None,  # (rows, cols) image.
        parameters: dict[str, Any] | None = None,  # Options.
    ) -> dict[str, Any]:  # Prediction summary.
        # Images are required.
        if data is None:
            # Explain the problem.
            raise ValueError("image data required")
        # Generic prediction; 2-D data is one band.
        return self.predict(model_id, data, parameters=parameters)


# =============================================================================
# End of module src/unbihexium/serving/inference.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
