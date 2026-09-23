# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FastAPI application factory for unbihexium serving.

This module provides the main FastAPI application with endpoints for:
- Health checks
- Model listing and capabilities
- Inference execution
- Pipeline execution

Usage:
    uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000

Or programmatically:
    from unbihexium.serving.app import create_app
    app = create_app()
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import unbihexium
from unbihexium.registry.capabilities import CapabilityRegistry
from unbihexium.serving.inference import ModelInferenceService
from unbihexium.serving.schemas import (
    CapabilitiesResponse,
    CapabilityInfo,
    DetectionRequest,
    DetectionResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ModelsResponse,
    PipelineRequest,
    PipelineResponse,
    SegmentationRequest,
    SegmentationResponse,
)
from unbihexium.serving.security import PayloadSizeMiddleware


def create_app(
    title: str = "Unbihexium API",
    version: str | None = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        title: API title.
        version: API version (defaults to library version).
        enable_cors: Whether to enable CORS.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description="REST API for Earth Observation and Geospatial AI",
        version=version or unbihexium.__version__,
        license_info={"name": "MPL-2.0", "url": "https://mozilla.org/MPL/2.0/"},
    )

    # Middleware
    app.add_middleware(PayloadSizeMiddleware)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Services
    inference_service = ModelInferenceService()

    # Health endpoint
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=unbihexium.__version__,
            ready=True,
        )

    # Capabilities endpoint
    @app.get("/capabilities", response_model=CapabilitiesResponse, tags=["Discovery"])
    async def capabilities() -> CapabilitiesResponse:
        """List available capabilities."""
        caps = CapabilityRegistry.list_capabilities()
        return CapabilitiesResponse(
            count=len(caps),
            capabilities=[
                CapabilityInfo(
                    id=c.capability_id,
                    name=c.name,
                    domain=c.domain,
                    maturity=c.maturity,
                )
                for c in caps
            ],
        )

    # Models endpoint
    @app.get("/models", response_model=ModelsResponse, tags=["Discovery"])
    async def models() -> ModelsResponse:
        """List available models."""
        model_list = inference_service.list_available_models()
        return ModelsResponse(
            count=len(model_list),
            models=[
                ModelInfo(
                    model_id=m["model_id"],
                    task=m["task"],
                    description=m["description"],
                )
                for m in model_list
            ],
        )

    # Generic inference endpoint
    @app.post("/infer/{model_id}", response_model=InferenceResponse, tags=["Inference"])
    async def infer(
        model_id: str,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Run inference with specified model.

        Args:
            model_id: Model identifier.
            request: Inference request with data and parameters.

        Returns:
            Inference results.
        """
        try:
            result = inference_service.run_inference(
                model_id=model_id,
                data=request.data,
                parameters=request.parameters,
            )

            if "error" in result:
                return InferenceResponse(
                    model_id=model_id,
                    success=False,
                    result={},
                    error=result["error"],
                )

            return InferenceResponse(
                model_id=model_id,
                success=True,
                result=result,
            )

        except Exception as e:
            return InferenceResponse(
                model_id=model_id,
                success=False,
                result={},
                error=str(e),
            )

    # Detection endpoint
    @app.post("/detect/{model_id}", response_model=DetectionResponse, tags=["Inference"])
    async def detect(
        model_id: str,
        request: DetectionRequest,
    ) -> DetectionResponse:
        """Run object detection.

        Args:
            model_id: Detection model identifier.
            request: Detection request with image data.

        Returns:
            Detection results with bounding boxes.
        """
        if request.image_data is None:
            raise HTTPException(status_code=400, detail="Image data required")

        try:
            result = inference_service.run_detection(
                model_id=model_id,
                image_data=request.image_data,
                threshold=request.threshold,
            )

            return DetectionResponse(
                model_id=result["model_id"],
                count=result["count"],
                detections=[
                    {
                        "x1": d["x1"],
                        "y1": d["y1"],
                        "x2": d["x2"],
                        "y2": d["y2"],
                        "confidence": d["confidence"],
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                    }
                    for d in result["detections"]
                ],
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Segmentation endpoint
    @app.post("/segment/{model_id}", response_model=SegmentationResponse, tags=["Inference"])
    async def segment(
        model_id: str,
        request: SegmentationRequest,
    ) -> SegmentationResponse:
        """Run semantic segmentation.

        Args:
            model_id: Segmentation model identifier.
            request: Segmentation request with image data.

        Returns:
            Segmentation results with mask info.
        """
        if request.image_data is None:
            raise HTTPException(status_code=400, detail="Image data required")

        try:
            result = inference_service.run_segmentation(
                model_id=model_id,
                image_data=request.image_data,
                threshold=request.threshold,
            )

            return SegmentationResponse(
                model_id=result["model_id"],
                mask_shape=tuple(result["mask_shape"]),
                classes=result["classes"],
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
