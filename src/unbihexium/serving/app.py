# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/serving/app.py
# Title       : FastAPI application of the REST service
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires FastAPI, Starlette and
#               PyTorch or ONNX Runtime for predictions
# =============================================================================
#
# Abstract
# --------
# create_app() builds the REST service:
#
#   GET  /health                    liveness and readiness
#   GET  /capabilities              capabilities, optionally of one domain
#   GET  /capabilities/{id}         one capability
#   GET  /models                    models with task, domain and variant
#                                   filters and pagination
#   GET  /models/{model_id}         one model: bands, outputs, units
#   GET  /pipelines                 registered pipelines
#   POST /predict/{model_id}        run any model zoo model on an image
#   POST /infer/{model_id}          single-band inference (earlier API)
#   POST /detect/{model_id}         detection (earlier API)
#   POST /segment/{model_id}        segmentation (earlier API)
#
# Settings come from unbihexium.config (ServingConfig): the request body
# limit (413 above it), image pixel and value limits, an optional API key
# (X-API-Key header, required on every route except /health), an optional
# per-client rate limit (429) and CORS origins. Unknown models give 404,
# invalid inputs 422. Prediction routes are plain functions, so FastAPI runs
# them in its thread pool and the event loop stays responsive.
#
# Usage
# -----
#   uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000
#
#   from unbihexium.serving import create_app
#   app = create_app()
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# Web framework.
from fastapi import Depends, FastAPI, HTTPException, Query

# Cross-origin resource sharing.
from fastapi.middleware.cors import CORSMiddleware

# Library version.
import unbihexium

# Service settings.
from unbihexium.config import ServingConfig, get_settings

# Capability registry.
from unbihexium.registry.capabilities import Capability, CapabilityRegistry

# Pipeline registry.
from unbihexium.registry.pipelines import PipelineRegistry

# Inference service and its errors.
from unbihexium.serving.inference import (
    ModelInferenceService,  # Runs the models.
    PayloadTooLargeError,  # Inputs over the limits.
    UnknownModelError,  # Unknown model ids.
)  # End of the inference imports.

# Request and response models.
from unbihexium.serving.schemas import (
    CapabilitiesResponse,  # Capability list.
    CapabilityInfo,  # One capability.
    DetectionRequest,  # Earlier detection request.
    DetectionResponse,  # Earlier detection response.
    HealthResponse,  # Health check.
    InferenceRequest,  # Earlier inference request.
    InferenceResponse,  # Earlier inference response.
    ModelInfo,  # One model.
    ModelsResponse,  # Model list.
    PipelineInfo,  # One pipeline.
    PipelinesResponse,  # Pipeline list.
    PredictRequest,  # Prediction request.
    PredictResponse,  # Prediction response.
    SegmentationRequest,  # Earlier segmentation request.
    SegmentationResponse,  # Earlier segmentation response.
)  # End of the schema imports.

# Security helpers.
from unbihexium.serving.security import APIKeyAuth, RateLimiter, RequestSizeLimitMiddleware

# Description of the service in the OpenAPI document.
API_DESCRIPTION = (
    "REST API for Earth observation and geospatial AI. Runs any model of the Unbihexium "
    "model zoo on images posted as JSON lists or base64 NumPy files and returns "
    "JSON summaries: boxes, class statistics or value statistics."
)  # End of the description.


# API description of a capability.
def _capability_info(c: Capability) -> CapabilityInfo:
    # Copy the public fields.
    return CapabilityInfo(
        id=c.capability_id,  # Id.
        name=c.name,  # Name.
        domain=c.domain.value,  # Domain.
        maturity=c.maturity.value,  # Maturity.
        description=c.description,  # Description.
        task=c.task,  # Task.
        models=c.models,  # Model ids.
        pipeline_id=c.pipeline_id,  # Pipeline.
    )  # End of the description.


# Convert service errors to HTTP errors.
def _http_error(exc: Exception) -> HTTPException:
    # Unknown models.
    if isinstance(exc, (UnknownModelError, KeyError)):
        # Not found; KeyError messages carry quotes, so use the argument.
        return HTTPException(status_code=404, detail=str(exc.args[0] if exc.args else exc))
    # Inputs over the limits.
    if isinstance(exc, PayloadTooLargeError):
        # Content too large.
        return HTTPException(status_code=413, detail=str(exc))
    # Invalid inputs.
    if isinstance(exc, ValueError):
        # Unprocessable content.
        return HTTPException(status_code=422, detail=str(exc))
    # Anything else is a server error without internal details.
    return HTTPException(status_code=500, detail=f"prediction failed: {type(exc).__name__}")


# Build the FastAPI application.
def create_app(
    title: str = "Unbihexium API",  # Title in the OpenAPI document.
    version: str | None = None,  # API version; default the library version.
    enable_cors: bool = True,  # Add the CORS middleware.
    config: ServingConfig | None = None,  # Settings; default from unbihexium.config.
    service: ModelInferenceService | None = None,  # Inference service to use.
) -> FastAPI:  # Configured application.
    # Settings of the service.
    settings = config or get_settings().serving
    # Model settings (device, backend, batch size).
    model_settings = get_settings().model
    # Application with its OpenAPI metadata.
    app = FastAPI(
        title=title,  # Title.
        description=API_DESCRIPTION,  # Description.
        version=version or unbihexium.__version__,  # Version.
        license_info={"name": "MPL-2.0", "url": "https://mozilla.org/MPL/2.0/"},  # Licence.
    )  # End of the application.
    # Request body limit.
    app.add_middleware(RequestSizeLimitMiddleware, max_size=settings.max_request_bytes)
    # Cross-origin requests.
    if enable_cors:
        # Credentials cannot be combined with a wildcard origin.
        wildcard = "*" in settings.cors_origins
        # Add the middleware.
        app.add_middleware(
            CORSMiddleware,  # Middleware class.
            allow_origins=list(settings.cors_origins),  # Origins.
            allow_credentials=not wildcard,  # Cookies only for listed origins.
            allow_methods=["GET", "POST"],  # Methods of the API.
            allow_headers=["*"],  # Headers.
        )  # End of the middleware.
    # Inference service.
    inference = service or ModelInferenceService(
        max_pixels=settings.max_pixels,  # Pixel limit.
        max_values=settings.max_values,  # Value limit.
        cache_size=settings.model_cache_size,  # Opened models.
        device=model_settings.device,  # Device.
        backend=model_settings.backend,  # Backend.
        batch_size=model_settings.batch_size,  # Batch size.
    )  # End of the service.
    # Keep it reachable, for example for tests.
    app.state.inference = inference
    # Dependencies of protected routes.
    guards: list[Any] = [Depends(APIKeyAuth(settings.api_key))]
    # Rate limit when configured.
    if settings.rate_limit_per_minute > 0:
        # Token bucket per client.
        guards.append(Depends(RateLimiter(settings.rate_limit_per_minute)))

    # Health check.
    @app.get(
        "/health",  # Path.
        response_model=HealthResponse,  # Response model.
        tags=["System"],  # OpenAPI group.
        summary="Health check",  # Short title.
        description="Liveness and readiness of the service; needs no API key.",  # Text.
    )  # End of the route.
    def health() -> HealthResponse:
        # Models that can be served.
        available = len(inference.list_available_models())
        # Service state.
        return HealthResponse(
            status="healthy",  # State.
            version=unbihexium.__version__,  # Version.
            ready=True,  # Accepting requests.
            models_available=available,  # Servable models.
            models_loaded=inference.loaded_count,  # Opened models.
        )  # End of the response.

    # Capability listing.
    @app.get(
        "/capabilities",  # Path.
        response_model=CapabilitiesResponse,  # Response model.
        tags=["Discovery"],  # OpenAPI group.
        summary="List capabilities",  # Short title.
        description="Capabilities of the library, optionally of one domain.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def capabilities(
        domain: str | None = Query(None, description="Domain, for example 'water'."),  # Filter.
    ) -> CapabilitiesResponse:  # Capability list.
        # Filtered or complete listing; unknown domains are 422.
        try:
            # Capabilities.
            caps = CapabilityRegistry.by_domain(domain) if domain else CapabilityRegistry.list_all()
        # Unknown domain names.
        except ValueError as exc:
            # Unprocessable.
            raise HTTPException(status_code=422, detail=f"unknown domain {domain!r}") from exc
        # Descriptions.
        items = [_capability_info(c) for c in caps]
        # Response with the count.
        return CapabilitiesResponse(count=len(items), capabilities=items)

    # One capability.
    @app.get(
        "/capabilities/{capability_id}",  # Path.
        response_model=CapabilityInfo,  # Response model.
        tags=["Discovery"],  # OpenAPI group.
        summary="Get a capability",  # Short title.
        description="One capability with its models and pipeline.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def capability(capability_id: str) -> CapabilityInfo:
        # Registry lookup.
        found = CapabilityRegistry.get(capability_id)
        # Unknown ids.
        if found is None:
            # Not found.
            raise HTTPException(status_code=404, detail=f"unknown capability {capability_id!r}")
        # Description.
        return _capability_info(found)

    # Model listing.
    @app.get(
        "/models",  # Path.
        response_model=ModelsResponse,  # Response model.
        tags=["Discovery"],  # OpenAPI group.
        summary="List models",  # Short title.
        description="Models that can be served, with filters and pagination.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def models(
        task: str | None = Query(None, description="Task, for example 'detection'."),  # Task.
        domain: str | None = Query(None, description="Domain, for example 'water'."),  # Domain.
        variant: str | None = Query(None, description="tiny, base, large or mega."),  # Variant.
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of models."),  # Size.
        offset: int = Query(0, ge=0, description="Index of the first model."),  # Start.
    ) -> ModelsResponse:  # Page of models.
        # Filtered listing; invalid filter values are 422.
        try:
            # Descriptions.
            listing = inference.list_available_models(task, domain, variant)
        # Unknown task or variant names.
        except ValueError as exc:
            # Unprocessable.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Page of the listing.
        page = listing[offset : offset + limit]
        # Response.
        return ModelsResponse(
            count=len(page),  # Page size.
            total=len(listing),  # Matches.
            offset=offset,  # First index.
            models=[ModelInfo(**m) for m in page],  # Models.
        )  # End of the response.

    # One model.
    @app.get(
        "/models/{model_id}",  # Path.
        response_model=ModelInfo,  # Response model.
        tags=["Discovery"],  # OpenAPI group.
        summary="Get a model",  # Short title.
        description="Input bands, outputs and units of one model.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def model(model_id: str) -> ModelInfo:
        # Service lookup.
        try:
            # Description.
            return ModelInfo(**inference.model_info(model_id))
        # Unknown ids.
        except UnknownModelError as exc:
            # Not found.
            raise _http_error(exc) from exc

    # Pipeline listing.
    @app.get(
        "/pipelines",  # Path.
        response_model=PipelinesResponse,  # Response model.
        tags=["Discovery"],  # OpenAPI group.
        summary="List pipelines",  # Short title.
        description="Pipelines registered by the task APIs; run them with the CLI.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def pipelines() -> PipelinesResponse:
        # Task APIs register their pipelines on import.
        import unbihexium.ai

        # Registered pipelines.
        entries = PipelineRegistry.list_all()
        # Response.
        return PipelinesResponse(
            count=len(entries),  # Count.
            pipelines=[PipelineInfo(**p.to_dict()) for p in entries],  # Pipelines.
        )  # End of the response.

    # Generic prediction.
    @app.post(
        "/predict/{model_id}",  # Path.
        response_model=PredictResponse,  # Response model.
        tags=["Inference"],  # OpenAPI group.
        summary="Run a model",  # Short title.
        description=(  # Text.
            "Run any model zoo model on an image. The band count must match the model "
            "(see GET /models/{model_id}). Returns boxes for detection, class statistics "
            "for segmentation and change detection, value statistics for regression and "
            "spectral indices, and band statistics for enhancement and super-resolution."
        ),  # End of the text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def predict(model_id: str, request: PredictRequest) -> PredictResponse:
        # Run the model; service errors become HTTP errors.
        try:
            # Decode the image.
            image = inference.decode_image(request.image, request.image_npy_base64)
            # Prediction summary.
            response = inference.predict(
                model_id,  # Model.
                image,  # Image.
                crs=request.crs,  # Coordinate system.
                transform=request.transform,  # Affine coefficients.
                nodata=request.nodata,  # No-data value.
                parameters=request.parameters.model_dump(exclude_none=True),  # Options.
            )  # End of the prediction.
        # Map the errors.
        except Exception as exc:
            # HTTP error.
            raise _http_error(exc) from exc
        # Response model.
        return PredictResponse(**response)

    # Single-band inference of earlier releases.
    @app.post(
        "/infer/{model_id}",  # Path.
        response_model=InferenceResponse,  # Response model.
        tags=["Inference"],  # OpenAPI group.
        summary="Run a model on one band (earlier API)",  # Short title.
        description="Earlier API: single-band image; errors are reported in the body.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def infer(model_id: str, request: InferenceRequest) -> InferenceResponse:
        # Errors are reported in the body, as in earlier releases.
        try:
            # Prediction summary.
            result = inference.run_inference(model_id, request.data, request.parameters)
        # Any failure.
        except Exception as exc:
            # Message of the mapped error.
            detail = _http_error(exc).detail
            # Failed response.
            return InferenceResponse(model_id=model_id, success=False, result={}, error=detail)
        # Successful response.
        return InferenceResponse(model_id=result["model_id"], success=True, result=result)

    # Detection of earlier releases.
    @app.post(
        "/detect/{model_id}",  # Path.
        response_model=DetectionResponse,  # Response model.
        tags=["Inference"],  # OpenAPI group.
        summary="Detect objects (earlier API)",  # Short title.
        description="Earlier API: boxes of a detection model; prefer POST /predict.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def detect(model_id: str, request: DetectionRequest) -> DetectionResponse:
        # Images are required.
        if request.image_data is None:
            # Bad request.
            raise HTTPException(status_code=400, detail="image data required")
        # Run the detector.
        try:
            # Earlier layout.
            result = inference.run_detection(model_id, request.image_data, request.threshold)
        # Map the errors.
        except Exception as exc:
            # HTTP error.
            raise _http_error(exc) from exc
        # Response model.
        return DetectionResponse(**result)

    # Segmentation of earlier releases.
    @app.post(
        "/segment/{model_id}",  # Path.
        response_model=SegmentationResponse,  # Response model.
        tags=["Inference"],  # OpenAPI group.
        summary="Segment an image (earlier API)",  # Short title.
        description="Earlier API: class map summary; prefer POST /predict.",  # Text.
        dependencies=guards,  # Authentication and rate limit.
    )  # End of the route.
    def segment(model_id: str, request: SegmentationRequest) -> SegmentationResponse:
        # Images are required.
        if request.image_data is None:
            # Bad request.
            raise HTTPException(status_code=400, detail="image data required")
        # Run the segmenter.
        try:
            # Earlier layout.
            result = inference.run_segmentation(model_id, request.image_data, request.threshold)
        # Map the errors.
        except Exception as exc:
            # HTTP error.
            raise _http_error(exc) from exc
        # Response model.
        return SegmentationResponse(**result)

    # Return the application.
    return app


# Application instance for `uvicorn unbihexium.serving.app:app`.
app = create_app()


# Run a development server when executed as a script.
if __name__ == "__main__":
    # ASGI server, imported only here.
    import uvicorn

    # Settings of the service.
    serving = get_settings().serving
    # Serve the application.
    uvicorn.run(app, host=serving.host, port=serving.port)


# =============================================================================
# End of module src/unbihexium/serving/app.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
