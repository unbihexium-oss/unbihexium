# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/serving/schemas.py
# Title       : Request and response models of the REST service
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pydantic 2
# =============================================================================
#
# Abstract
# --------
# Pydantic models of the REST API. Every field carries a description, which
# FastAPI publishes in the OpenAPI document at /openapi.json and /docs.
#
#   discovery    HealthResponse, CapabilityInfo, CapabilitiesResponse,
#                ModelInfo, ModelsResponse, PipelineInfo, PipelinesResponse
#   prediction   PredictParameters, PredictRequest, PredictResponse
#   legacy       InferenceRequest/Response, DetectionRequest/Response,
#                SegmentationRequest/Response of earlier releases
#   errors       ErrorResponse
#
# Images are sent either as nested JSON lists (bands, rows, cols) or, more
# compactly, as a base64-encoded NumPy .npy file, which keeps the data type
# and needs about a third of the bytes of JSON text.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Model base class and field descriptions.
from pydantic import BaseModel, Field

# Example prediction request: two bands (red, nir) of a 1 x 2 image.
EXAMPLE_REQUEST = {"image": [[[0.1, 0.2]], [[0.4, 0.6]]], "parameters": {}}


# Health check.
class HealthResponse(BaseModel):
    # Service state.
    status: str = Field(description="Service health status, 'healthy' when serving.")
    # Library version.
    version: str = Field(description="Version of the unbihexium library.")
    # Readiness.
    ready: bool = Field(description="Whether the service accepts requests.")
    # Zoo size.
    models_available: int = Field(default=0, description="Number of models that can be served.")
    # Cache size.
    models_loaded: int = Field(default=0, description="Number of models open in memory.")


# One capability.
class CapabilityInfo(BaseModel):
    # Id.
    id: str = Field(description="Capability identifier.")
    # Name.
    name: str = Field(description="Human-readable name.")
    # Domain.
    domain: str = Field(description="Capability domain, for example 'water' or 'sar'.")
    # Maturity.
    maturity: str = Field(description="Maturity level: stable, beta, research or deprecated.")
    # Description.
    description: str = Field(default="", description="What the capability does.")
    # Task.
    task: str | None = Field(default=None, description="Model task of model capabilities.")
    # Models.
    models: list[str] = Field(default_factory=list, description="Model ids that provide it.")
    # Pipeline.
    pipeline_id: str | None = Field(default=None, description="Pipeline that runs it, if any.")


# List of capabilities.
class CapabilitiesResponse(BaseModel):
    # Count.
    count: int = Field(description="Number of capabilities returned.")
    # Items.
    capabilities: list[CapabilityInfo] = Field(description="Capabilities of the library.")


# One model.
class ModelInfo(BaseModel):
    # Id.
    model_id: str = Field(description="Model identifier, for example 'ship_detector_base'.")
    # Task.
    task: str = Field(description="Task: detection, segmentation, dense_regression, ...")
    # Description.
    description: str = Field(description="What the model predicts once trained.")
    # Name.
    name: str = Field(default="", description="Human-readable name with the variant.")
    # Domain.
    domain: str = Field(default="", description="Capability domain of the model.")
    # Variant.
    variant: str = Field(default="", description="Size variant: tiny, base, large or mega.")
    # Channel count.
    in_channels: int = Field(default=0, description="Number of input bands expected.")
    # Channel names.
    channels: list[str] = Field(default_factory=list, description="Names of the input bands.")
    # Outputs.
    outputs: list[str] = Field(default_factory=list, description="Classes, targets or bands.")
    # Units.
    units: list[str] = Field(default_factory=list, description="Units of regression outputs.")
    # Training flag.
    requires_training: bool = Field(
        # Default value.
        default=True,
        # Description in the OpenAPI document.
        description="True for starter weights, whose predictions are not meaningful yet.",
    )


# List of models.
class ModelsResponse(BaseModel):
    # Count.
    count: int = Field(description="Number of models in this page.")
    # Total.
    total: int = Field(default=0, description="Number of models matching the filters.")
    # Offset.
    offset: int = Field(default=0, description="Index of the first model of this page.")
    # Items.
    models: list[ModelInfo] = Field(description="Models of this page.")


# One pipeline.
class PipelineInfo(BaseModel):
    # Id.
    pipeline_id: str = Field(description="Pipeline identifier.")
    # Name.
    name: str = Field(description="Human-readable name.")
    # Description.
    description: str = Field(default="", description="What the pipeline does.")
    # Domains.
    domains: list[str] = Field(default_factory=list, description="Capability domains.")


# List of pipelines.
class PipelinesResponse(BaseModel):
    # Count.
    count: int = Field(description="Number of pipelines.")
    # Items.
    pipelines: list[PipelineInfo] = Field(description="Registered pipelines.")


# Options of a prediction.
class PredictParameters(BaseModel):
    # Score or probability threshold.
    threshold: float | None = Field(
        # Default value.
        default=None,
        # Lower bound.
        ge=0.0,
        # Upper bound.
        le=1.0,
        # Description in the OpenAPI document.
        description="Detection score or segmentation probability threshold.",
    )
    # NMS overlap.
    iou_threshold: float | None = Field(
        # Default value.
        default=None,
        # Lower bound.
        ge=0.0,
        # Upper bound.
        le=1.0,
        # Description in the OpenAPI document.
        description="Overlap above which weaker boxes are suppressed (detection).",
    )
    # Box limit.
    max_detections: int | None = Field(
        # Default value.
        default=None,
        # Lower bound.
        ge=1,
        # Upper bound.
        le=10000,
        # Description in the OpenAPI document.
        description="Maximum number of boxes returned (detection).",
    )
    # Tile size.
    tile_size: int | None = Field(
        # Default value.
        default=None,
        # Lower bound.
        ge=32,
        # Upper bound.
        le=2048,
        # Description in the OpenAPI document.
        description="Tile size of the inference; default from the model.",
    )
    # Overlap.
    overlap: float | None = Field(
        # Default value.
        default=None,
        # Lower bound.
        ge=0.0,
        # Upper bound.
        le=0.9,
        # Description in the OpenAPI document.
        description="Overlap between tiles as a fraction of the tile size.",
    )
    # Class map in the response.
    return_mask: bool = Field(
        # Default value.
        default=False,
        # Description in the OpenAPI document.
        description="Include the class map as nested lists (segmentation).",
    )
    # Values in the response.
    return_values: bool = Field(
        # Default value.
        default=False,
        # Description in the OpenAPI document.
        description="Include the output values as nested lists (dense outputs).",
    )
    # GeoJSON in the response.
    return_geojson: bool = Field(
        # Default value.
        default=False,
        # Description in the OpenAPI document.
        description="Include the detections as a GeoJSON FeatureCollection.",
    )


# Prediction request.
class PredictRequest(BaseModel):
    # Nested lists.
    image: list[Any] | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Image as nested lists (bands, rows, cols), or (rows, cols) for one band.",
    )
    # Base64 NumPy file.
    image_npy_base64: str | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Image as a base64-encoded NumPy .npy file (numeric types only).",
    )
    # Coordinate system.
    crs: str | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="CRS of the image, for example 'EPSG:32632'; used for map coordinates.",
    )
    # Affine transform.
    transform: list[float] | None = Field(
        # Default value.
        default=None,
        # Minimum length.
        min_length=6,
        # Maximum length.
        max_length=6,
        # Description in the OpenAPI document.
        description="Affine coefficients (a, b, c, d, e, f): x = a*col + b*row + c.",
    )
    # No-data value.
    nodata: float | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Value of missing pixels; pixels where every band has it are ignored.",
    )
    # Options.
    parameters: PredictParameters = Field(
        # Default factory.
        default_factory=PredictParameters,
        # Description in the OpenAPI document.
        description="Options of the prediction.",
    )

    # Example shown in the OpenAPI document: a two-band 1 x 2 image.
    model_config = {"json_schema_extra": {"examples": [EXAMPLE_REQUEST]}}


# Prediction response.
class PredictResponse(BaseModel):
    # Id.
    model_id: str = Field(description="Model that produced the result.")
    # Task.
    task: str = Field(description="Task of the model.")
    # Success.
    success: bool = Field(default=True, description="Whether the prediction succeeded.")
    # Input shape.
    input_shape: list[int] = Field(description="Shape (bands, rows, cols) of the input.")
    # Time.
    elapsed_ms: float = Field(description="Wall-clock time of the prediction in milliseconds.")
    # Training flag.
    requires_training: bool = Field(
        # Description in the OpenAPI document.
        description="True when the model has starter weights; its output is not meaningful.",
    )
    # Result.
    result: dict[str, Any] = Field(description="Task-specific result, see the API reference.")


# Generic inference request of earlier releases.
class InferenceRequest(BaseModel):
    # Single band.
    data: list[list[float]] | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Single-band image as nested lists (rows, cols).",
    )
    # Options.
    parameters: dict[str, Any] = Field(
        # Default factory.
        default_factory=dict,
        # Description in the OpenAPI document.
        description="Options of the prediction, see PredictParameters.",
    )


# Generic inference response of earlier releases.
class InferenceResponse(BaseModel):
    # Id.
    model_id: str = Field(description="Model used for inference.")
    # Success.
    success: bool = Field(description="Whether inference succeeded.")
    # Result.
    result: dict[str, Any] = Field(description="Inference result.")
    # Error.
    error: str | None = Field(default=None, description="Error message if inference failed.")


# One detected object.
class DetectionBox(BaseModel):
    # Left.
    x1: float = Field(description="Left pixel coordinate.")
    # Top.
    y1: float = Field(description="Top pixel coordinate.")
    # Right.
    x2: float = Field(description="Right pixel coordinate.")
    # Bottom.
    y2: float = Field(description="Bottom pixel coordinate.")
    # Score.
    confidence: float = Field(description="Detection score in [0, 1].")
    # Class index.
    class_id: int = Field(description="Class index.")
    # Class name.
    class_name: str = Field(description="Class name.")
    # Map box.
    geo_bbox: list[float] | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Box (min x, min y, max x, max y) in map coordinates, if georeferenced.",
    )


# Detection request of earlier releases.
class DetectionRequest(BaseModel):
    # Image.
    image_data: list[list[list[float]]] | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Image as nested lists (bands, rows, cols).",
    )
    # Threshold.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection threshold.")


# Detection response.
class DetectionResponse(BaseModel):
    # Id.
    model_id: str = Field(description="Model used.")
    # Count.
    count: int = Field(description="Number of detections.")
    # Boxes.
    detections: list[DetectionBox] = Field(description="Detected objects.")


# Segmentation request of earlier releases.
class SegmentationRequest(BaseModel):
    # Image.
    image_data: list[list[list[float]]] | None = Field(
        # Default value.
        default=None,
        # Description in the OpenAPI document.
        description="Image as nested lists (bands, rows, cols).",
    )
    # Threshold.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability threshold.")


# Segmentation response.
class SegmentationResponse(BaseModel):
    # Id.
    model_id: str = Field(description="Model used.")
    # Shape.
    mask_shape: tuple[int, int] = Field(description="Shape (rows, cols) of the class map.")
    # Classes.
    classes: list[str] = Field(description="Class names.")
    # Fractions.
    class_fractions: dict[str, float] = Field(
        # Default factory.
        default_factory=dict,
        # Description in the OpenAPI document.
        description="Fraction of the valid pixels in each class.",
    )


# Pipeline execution request (pipelines run from the command line).
class PipelineRequest(BaseModel):
    # Id.
    pipeline_id: str = Field(description="Pipeline identifier.")
    # Inputs.
    inputs: dict[str, Any] = Field(description="Pipeline inputs.")
    # Options.
    parameters: dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters.")


# Pipeline execution response.
class PipelineResponse(BaseModel):
    # Run id.
    run_id: str = Field(description="Pipeline run identifier.")
    # Id.
    pipeline_id: str = Field(description="Pipeline identifier.")
    # Status.
    status: str = Field(description="Execution status.")
    # Outputs.
    outputs: dict[str, Any] = Field(default_factory=dict, description="Pipeline outputs.")
    # Error.
    error: str | None = Field(default=None, description="Error message if the run failed.")


# Error body.
class ErrorResponse(BaseModel):
    # Message.
    error: str = Field(description="Error message.")
    # Details.
    detail: str | None = Field(default=None, description="Detailed error information.")
    # Status.
    status_code: int = Field(description="HTTP status code.")


# =============================================================================
# End of module src/unbihexium/serving/schemas.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
