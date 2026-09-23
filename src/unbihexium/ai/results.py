# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/results.py
# Title       : Georeferenced results of model inference
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Result records returned by the task APIs and the command line:
#
#   Detection, DetectionResult    boxes in pixel and map coordinates, with
#                                 GeoJSON export
#   SegmentationResult            class label map, optional probabilities,
#                                 class areas and export as a raster
#   RegressionResult              per-pixel or per-scene values with names
#                                 and units
#   EnhancementResult             enhanced image bands as a raster
#   SuperResolutionResult         upscaled image as a raster
#
# Georeferencing
# --------------
# Rasters store the affine transform as the six coefficients (a, b, c, d, e,
# f) of rasterio, optionally followed by (0, 0, 1). A pixel position (col,
# row) maps to x = a * col + b * row + c and y = d * col + e * row + f. The
# helpers below convert between the two coordinate systems and derive the
# transform of an output grid with a different resolution (detector heat
# maps, super-resolution output).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Result records.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Raster container of the library.
from unbihexium.core.raster import Raster

# Identity transform used by rasters created from bare arrays.
IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


# Return the six affine coefficients of a raster transform.
def affine_coefficients(transform: tuple[float, ...] | None) -> tuple[float, ...]:
    # Rasters without georeferencing use the identity of from_array.
    if not transform:
        # Default transform.
        return IDENTITY_TRANSFORM
    # Keep the first six coefficients; a trailing (0, 0, 1) row is dropped.
    return tuple(float(v) for v in transform[:6])


# Convert pixel coordinates to map coordinates.
def pixel_to_map(
    transform: tuple[float, ...] | None,  # Affine transform of the raster.
    col: float,  # Column position, may be fractional.
    row: float,  # Row position, may be fractional.
) -> tuple[float, float]:  # Map coordinates (x, y).
    # Affine coefficients.
    a, b, c, d, e, f = affine_coefficients(transform)
    # Apply the affine transform.
    return (a * col + b * row + c, d * col + e * row + f)


# Transform of a grid whose pixels are `factor` times smaller than the input.
def scaled_transform(transform: tuple[float, ...] | None, factor: float) -> tuple[float, ...]:
    # Affine coefficients of the input grid.
    a, b, c, d, e, f = affine_coefficients(transform)
    # Pixel size shrinks by the factor; the origin stays the same.
    return (a / factor, b / factor, c, d / factor, e / factor, f)


# Coordinate reference system and transform of a raster.
def georeference(raster: Raster) -> tuple[str, tuple[float, ...]]:
    # Rasters without metadata get the library defaults.
    if raster.metadata is None:
        # Geographic coordinates and identity transform.
        return ("EPSG:4326", IDENTITY_TRANSFORM)
    # CRS and transform of the metadata.
    return (raster.metadata.crs, affine_coefficients(raster.metadata.transform))


# A single detected object.
@dataclass
class Detection:
    # Box in pixel coordinates: x1, y1, x2, y2 (columns and rows).
    bbox: tuple[float, float, float, float]
    # Detection score in [0, 1].
    confidence: float
    # Index of the class in the model outputs.
    class_id: int
    # Name of the class.
    class_name: str
    # Box in map coordinates: min x, min y, max x, max y.
    geo_bbox: tuple[float, float, float, float] | None = None

    # Width of the box in pixels.
    @property
    def width(self) -> float:
        # Difference of the x coordinates.
        return self.bbox[2] - self.bbox[0]

    # Height of the box in pixels.
    @property
    def height(self) -> float:
        # Difference of the y coordinates.
        return self.bbox[3] - self.bbox[1]

    # Centre of the box in pixels.
    @property
    def center(self) -> tuple[float, float]:
        # Mean of the corners.
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "bbox": [float(v) for v in self.bbox],  # Pixel box.
            "confidence": float(self.confidence),  # Score.
            "class_id": int(self.class_id),  # Class index.
            "class_name": self.class_name,  # Class name.
            "geo_bbox": [float(v) for v in self.geo_bbox] if self.geo_bbox else None,  # Map box.
        }  # End of the dictionary.


# Result of object detection on one image.
@dataclass
class DetectionResult:
    # Detected objects, sorted by decreasing confidence.
    detections: list[Detection] = field(default_factory=list)
    # Source of the input, for example a file path.
    source: str = ""
    # Model id of the detector.
    model_id: str = ""
    # Coordinate reference system of geo_bbox.
    crs: str = "EPSG:4326"

    # Number of detections.
    @property
    def count(self) -> int:
        # Length of the list.
        return len(self.detections)

    # Keep only detections with at least the given confidence.
    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        # Detections above the threshold.
        kept = [d for d in self.detections if d.confidence >= threshold]
        # New result with the same metadata.
        return DetectionResult(kept, self.source, self.model_id, self.crs)

    # Keep only detections of the given classes.
    def filter_by_class(self, *names: str) -> DetectionResult:
        # Detections whose class name was requested.
        kept = [d for d in self.detections if d.class_name in names]
        # New result with the same metadata.
        return DetectionResult(kept, self.source, self.model_id, self.crs)

    # Number of detections per class name.
    def counts_by_class(self) -> dict[str, int]:
        # Counter dictionary.
        counts: dict[str, int] = {}
        # Count every detection.
        for d in self.detections:
            # Increment the class count.
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        # Return the counts.
        return counts

    # Boxes, scores and class ids as arrays, as used by the metrics.
    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        # Boxes as an (N, 4) array.
        boxes = np.array([d.bbox for d in self.detections], dtype=np.float64).reshape(-1, 4)
        # Scores as an (N,) array.
        scores = np.array([d.confidence for d in self.detections], dtype=np.float64)
        # Class ids as an (N,) array.
        classes = np.array([d.class_id for d in self.detections], dtype=np.int64)
        # Return the three arrays.
        return boxes, scores, classes

    # Export the detections as a GeoJSON FeatureCollection.
    def to_geojson(self, pixel_coordinates: bool = False) -> dict[str, Any]:
        # One feature per detection.
        features = []
        # Convert every detection.
        for d in self.detections:
            # Map box unless pixel coordinates were requested or none exists.
            box = d.bbox if pixel_coordinates or d.geo_bbox is None else d.geo_bbox
            # Corners of the box.
            x1, y1, x2, y2 = box
            # Closed polygon ring.
            ring = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            # Feature with the detection attributes.
            features.append(
                {  # GeoJSON feature.
                    "type": "Feature",  # GeoJSON object type.
                    "properties": {  # Attributes of the detection.
                        "class_id": d.class_id,  # Class index.
                        "class_name": d.class_name,  # Class name.
                        "confidence": d.confidence,  # Score.
                    },  # End of the properties.
                    "geometry": {"type": "Polygon", "coordinates": [ring]},  # Box polygon.
                }  # End of the feature.
            )  # End of the append.
        # Feature collection with the model and CRS as foreign members.
        return {
            "type": "FeatureCollection",  # GeoJSON object type.
            "model_id": self.model_id,  # Detector.
            "crs": "pixel" if pixel_coordinates else self.crs,  # Coordinate system.
            "features": features,  # Detections.
        }  # End of the collection.


# Result of semantic segmentation or change detection on one image.
@dataclass
class SegmentationResult:
    # Class index per pixel, shape (H, W), dtype uint8 or int32.
    mask: NDArray[Any]
    # Class names, indexed by the values of the mask.
    classes: list[str] = field(default_factory=list)
    # Model id of the segmenter.
    model_id: str = ""
    # Class probabilities, shape (K, H, W), when requested.
    probabilities: NDArray[np.float32] | None = None
    # Source of the input.
    source: str = ""
    # Coordinate reference system of the mask.
    crs: str = "EPSG:4326"
    # Affine transform of the mask.
    transform: tuple[float, ...] = IDENTITY_TRANSFORM
    # Value of pixels without valid input.
    nodata: int = 255

    # Number of classes.
    @property
    def num_classes(self) -> int:
        # Length of the class list.
        return len(self.classes)

    # Binary mask of one class, by index or name.
    def class_mask(self, cls: int | str) -> NDArray[np.bool_]:
        # Resolve a class name to its index.
        index = self.classes.index(cls) if isinstance(cls, str) else int(cls)
        # Pixels of that class.
        return self.mask == index

    # Fraction of the valid pixels per class.
    def class_fractions(self) -> dict[str, float]:
        # Valid pixels.
        valid = self.mask != self.nodata
        # Number of valid pixels, at least one to avoid division by zero.
        total = max(int(valid.sum()), 1)
        # Fraction per class.
        return {name: float((self.mask == i).sum()) / total for i, name in enumerate(self.classes)}

    # Area per class in square map units (square metres for projected CRS).
    def class_areas(self) -> dict[str, float]:
        # Affine coefficients.
        a, b, _, d, e, _ = affine_coefficients(self.transform)
        # Area of one pixel: absolute determinant of the linear part.
        pixel_area = abs(a * e - b * d)
        # Area per class.
        return {
            name: float((self.mask == i).sum()) * pixel_area  # Pixels times pixel area.
            for i, name in enumerate(self.classes)  # Every class.
        }  # End of the areas.

    # Export the mask as a single-band raster.
    def to_raster(self) -> Raster:
        # Raster with the georeferencing of the input.
        return Raster.from_array(
            self.mask.astype(np.uint8 if self.num_classes < 255 else np.int32),  # Labels.
            crs=self.crs,  # Coordinate system.
            transform=self.transform,  # Affine transform.
            nodata=self.nodata,  # No-data value.
        )  # End of the raster.


# Result of dense or scene-level regression on one image.
@dataclass
class RegressionResult:
    # Values: (K, H, W) for dense models, (K,) for scene models.
    values: NDArray[np.float32]
    # Output names.
    names: list[str] = field(default_factory=list)
    # Units of the outputs.
    units: list[str] = field(default_factory=list)
    # Model id of the regressor.
    model_id: str = ""
    # Source of the input.
    source: str = ""
    # Coordinate reference system of dense values.
    crs: str = "EPSG:4326"
    # Affine transform of dense values.
    transform: tuple[float, ...] = IDENTITY_TRANSFORM

    # Whether the values form a map.
    @property
    def is_dense(self) -> bool:
        # Dense values have spatial axes.
        return self.values.ndim == 3

    # Values of one output, by index or name.
    def output(self, name: int | str) -> NDArray[np.float32]:
        # Resolve a name to its index.
        index = self.names.index(name) if isinstance(name, str) else int(name)
        # Values of that output.
        return self.values[index]

    # Summary statistics per output, ignoring NaN.
    def summary(self) -> dict[str, dict[str, float]]:
        # Statistics per output.
        stats: dict[str, dict[str, float]] = {}
        # Iterate over the outputs.
        for i, name in enumerate(self.names):
            # Finite values of the output.
            v = np.asarray(self.values[i], dtype=np.float64)
            # Drop NaN and infinite values.
            v = v[np.isfinite(v)]
            # Statistics, or NaN when no value is valid.
            stats[name] = {
                "mean": float(v.mean()) if v.size else float("nan"),  # Mean.
                "min": float(v.min()) if v.size else float("nan"),  # Minimum.
                "max": float(v.max()) if v.size else float("nan"),  # Maximum.
                "std": float(v.std()) if v.size else float("nan"),  # Standard deviation.
            }  # End of the statistics.
        # Return the statistics.
        return stats

    # Plain dictionary of a scene-level result.
    def to_dict(self) -> dict[str, Any]:
        # Scene values are listed; dense values are summarised.
        values: Any = self.summary() if self.is_dense else self._named_values()
        # Dictionary with metadata.
        return {"model_id": self.model_id, "units": self.units, "values": values}

    # Scene values by output name.
    def _named_values(self) -> dict[str, float]:
        # Pair names and values.
        return dict(zip(self.names, self.values.tolist()))

    # Export dense values as a multi-band float32 raster.
    def to_raster(self) -> Raster:
        # Scene values have no spatial layout.
        if not self.is_dense:
            # Explain the limitation.
            raise ValueError("scene-level values cannot be written as a raster")
        # Raster with the georeferencing of the input.
        return Raster.from_array(
            self.values.astype(np.float32),  # Values.
            crs=self.crs,  # Coordinate system.
            transform=self.transform,  # Affine transform.
            nodata=float("nan"),  # Missing values are NaN.
        )  # End of the raster.


# Result of image enhancement (image-to-image models).
@dataclass
class EnhancementResult:
    # Enhanced image.
    raster: Raster | None = None
    # Output band names.
    bands: list[str] = field(default_factory=list)
    # Source of the input.
    source: str = ""
    # Model id of the enhancer.
    model_id: str = ""


# Result of super-resolution.
@dataclass
class SuperResolutionResult:
    # Upscaled image.
    raster: Raster | None = None
    # Upscaling factor.
    scale_factor: int = 2
    # Source of the input.
    source: str = ""
    # Model id of the network.
    model_id: str = ""


# =============================================================================
# End of module src/unbihexium/ai/results.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
