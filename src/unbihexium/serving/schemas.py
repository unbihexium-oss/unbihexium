# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service health status")
    version: str = Field(description="Library version")
    ready: bool = Field(description="Whether service is ready to handle requests")


class CapabilityInfo(BaseModel):
    """Information about a capability."""

    id: str = Field(description="Capability identifier")
    name: str = Field(description="Human-readable name")
    domain: str = Field(description="Domain category")
    maturity: str = Field(description="Maturity level")


class CapabilitiesResponse(BaseModel):
    """List of available capabilities."""

    count: int = Field(description="Number of capabilities")
    capabilities: list[CapabilityInfo] = Field(description="Available capabilities")


class ModelInfo(BaseModel):
    """Information about a model."""

    model_id: str = Field(description="Model identifier")
    task: str = Field(description="Task type")
    description: str = Field(description="Model description")


class ModelsResponse(BaseModel):
    """List of available models."""

    count: int = Field(description="Number of models")
    models: list[ModelInfo] = Field(description="Available models")


class InferenceRequest(BaseModel):
    """Generic inference request."""

    data: list[list[float]] | None = Field(
        default=None,
        description="Input data as nested list",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters",
    )


class InferenceResponse(BaseModel):
    """Generic inference response."""

    model_id: str = Field(description="Model used for inference")
    success: bool = Field(description="Whether inference succeeded")
    result: dict[str, Any] = Field(description="Inference results")
    error: str | None = Field(default=None, description="Error message if failed")


class DetectionBox(BaseModel):
    """Detected object bounding box."""

    x1: float = Field(description="Left coordinate")
    y1: float = Field(description="Top coordinate")
    x2: float = Field(description="Right coordinate")
    y2: float = Field(description="Bottom coordinate")
    confidence: float = Field(description="Detection confidence")
    class_id: int = Field(description="Class identifier")
    class_name: str = Field(description="Class name")


class DetectionRequest(BaseModel):
    """Detection request."""

    image_data: list[list[list[float]]] | None = Field(
        default=None,
        description="Image data as [C, H, W] nested list",
    )
    threshold: float = Field(default=0.5, description="Detection threshold")


class DetectionResponse(BaseModel):
    """Detection response."""

    model_id: str = Field(description="Model used")
    count: int = Field(description="Number of detections")
    detections: list[DetectionBox] = Field(description="Detected objects")


class SegmentationRequest(BaseModel):
    """Segmentation request."""

    image_data: list[list[list[float]]] | None = Field(
        default=None,
        description="Image data as [C, H, W] nested list",
    )
    threshold: float = Field(default=0.5, description="Segmentation threshold")


class SegmentationResponse(BaseModel):
    """Segmentation response."""

    model_id: str = Field(description="Model used")
    mask_shape: tuple[int, int] = Field(description="Output mask shape")
    classes: list[str] = Field(description="Class names")


class PipelineRequest(BaseModel):
    """Pipeline execution request."""

    pipeline_id: str = Field(description="Pipeline identifier")
    inputs: dict[str, Any] = Field(description="Pipeline inputs")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline parameters",
    )


class PipelineResponse(BaseModel):
    """Pipeline execution response."""

    run_id: str = Field(description="Pipeline run identifier")
    pipeline_id: str = Field(description="Pipeline identifier")
    status: str = Field(description="Execution status")
    outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline outputs",
    )
    error: str | None = Field(default=None, description="Error message if failed")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Detailed error info")
    status_code: int = Field(description="HTTP status code")
