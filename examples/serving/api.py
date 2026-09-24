# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : examples/serving/api.py
# Title       : Example: FastAPI REST service for Unbihexium models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires unbihexium[serving] and
#               python-multipart (file uploads)
# =============================================================================
#
# Abstract
# --------
# A simple REST API for serving Unbihexium models. It demonstrates how to
# expose detection and spectral index calculation over HTTP with FastAPI.
# Uploaded images are written to a temporary file, processed and deleted,
# also when processing fails. Client errors get 4xx answers: 422 for files
# that are not readable rasters, band numbers outside the image and images
# that do not fit the model; other failures get 500 without internal
# details.
#
# Endpoints
# ---------
#   GET  /health            Health check
#   GET  /info              Library information
#   POST /detect/ships      Ship detection
#   POST /detect/buildings  Building detection
#   POST /index/ndvi        NDVI calculation
#
# Usage
# -----
#   uvicorn examples.serving.api:app --host 0.0.0.0 --port 8000
#
# The interactive OpenAPI documentation is then served at /docs.
#
# Notes
# -----
# This is an example, not a hardened service: it has no authentication and
# no upload size limit. The service of the library is unbihexium.serving
# (see docker-compose.yml).
# Descriptions shown in the OpenAPI documentation are passed explicitly to
# the decorators and model configurations below.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Temporary files for the uploaded images.
import tempfile

# Type of the generator of the upload context manager.
from collections.abc import Iterator

# Delete the temporary files whatever happens.
from contextlib import contextmanager

# Represent the temporary file path.
from pathlib import Path

# Compute summary statistics that ignore NaN pixels.
import numpy as np

# Web framework, file upload support, query validation and HTTP errors.
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

# Base class and configuration of the response models.
from pydantic import BaseModel, ConfigDict

# Library version for the health and info endpoints.
import unbihexium

# Detection model wrappers of Unbihexium.
from unbihexium.ai.detection import BuildingDetector, ObjectDetector, ShipDetector

# Spectral index calculator of Unbihexium.
from unbihexium.core.index import compute_index

# Raster container with pixel data and georeferencing metadata.
from unbihexium.core.raster import Raster

# FastAPI application
app = FastAPI(
    title="Unbihexium API",  # Title shown in the OpenAPI documentation.
    description="REST API for Earth Observation and Geospatial AI",  # API description.
    version=unbihexium.__version__,  # API version follows the library version.
    license_info={"name": "MPL-2.0", "url": "https://mozilla.org/MPL/2.0/"},  # Licence.
)  # End of the application definition.


# Response models
# Response of GET /health.
class HealthResponse(BaseModel):
    # Description of the schema in the OpenAPI documentation.
    model_config = ConfigDict(json_schema_extra={"description": "Health check response."})

    # Service status, "healthy" when the service answers.
    status: str
    # Version of the Unbihexium library.
    version: str


# Response of GET /info.
class InfoResponse(BaseModel):
    # Description of the schema in the OpenAPI documentation.
    model_config = ConfigDict(json_schema_extra={"description": "Library info response."})

    # Package name.
    name: str
    # Version of the Unbihexium library.
    version: str
    # One-line description of the library.
    description: str


# One detected object.
class Detection(BaseModel):
    # Description of the schema in the OpenAPI documentation.
    model_config = ConfigDict(json_schema_extra={"description": "Single detection."})

    # Bounding box in pixels as (xmin, ymin, xmax, ymax).
    bbox: tuple[float, float, float, float]
    # Model confidence between 0 and 1.
    confidence: float
    # Numeric class identifier.
    class_id: int
    # Human-readable class name.
    class_name: str


# Response of the detection endpoints.
class DetectionResponse(BaseModel):
    # Description of the schema in the OpenAPI documentation.
    model_config = ConfigDict(json_schema_extra={"description": "Detection response."})

    # Number of detections.
    count: int
    # Identifier of the model that produced the detections.
    model_id: str
    # The detections themselves.
    detections: list[Detection]


# Response of the index endpoint.
class IndexResponse(BaseModel):
    # Description of the schema in the OpenAPI documentation.
    model_config = ConfigDict(json_schema_extra={"description": "Index calculation response."})

    # Name of the spectral index, for example "NDVI".
    index_name: str
    # Minimum index value, ignoring NaN pixels.
    min_value: float
    # Maximum index value, ignoring NaN pixels.
    max_value: float
    # Mean index value, ignoring NaN pixels.
    mean_value: float
    # Image size as (rows, columns).
    shape: tuple[int, int]


# Open an upload as a raster; the temporary file is deleted on every path.
@contextmanager
def uploaded_raster(file: UploadFile) -> Iterator[Raster]:
    # Temporary GeoTIFF file that outlives the with block of its creation.
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        # Copy the upload into it.
        tmp.write(file.file.read())
        # Keep the path.
        tmp_path = Path(tmp.name)
    # Read the raster and hand it to the caller.
    try:
        # Files that are not rasters are client errors.
        try:
            # Open the raster.
            raster = Raster.from_file(tmp_path)
            # Read the pixels while the file still exists.
            raster.load()
        # Any failure to read the upload.
        except Exception as exc:
            # Unprocessable content.
            raise HTTPException(
                status_code=422,  # Unprocessable content.
                detail="the upload is not a readable raster",  # Explanation.
            ) from exc  # Keep the read error as the cause.
        # Processing by the caller.
        yield raster
    # Whatever happened.
    finally:
        # Delete the temporary file.
        tmp_path.unlink(missing_ok=True)


# Run a detector and convert its result, mapping errors to HTTP answers.
def run_detector(detector: ObjectDetector, raster: Raster) -> DetectionResponse:
    # Detection on the whole raster.
    try:
        # Result of the task API.
        result = detector.predict(raster)
    # Images that do not fit the model, for example a wrong band count.
    except ValueError as exc:
        # Unprocessable content with the reason.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Anything else is a server error.
    except Exception as exc:
        # Internal error without internal details.
        raise HTTPException(status_code=500, detail="detection failed") from exc
    # Convert to response items.
    detections = [
        Detection(  # One response item per detection.
            bbox=d.bbox,  # Bounding box.
            confidence=d.confidence,  # Model confidence.
            class_id=d.class_id,  # Numeric class identifier.
            class_name=d.class_name,  # Human-readable class name.
        )  # End of the response item.
        for d in result.detections  # Iterate over the detections.
    ]  # End of the list of detections.
    # Build the response body.
    return DetectionResponse(
        count=result.count,  # Number of detections.
        model_id=result.model_id,  # Model that produced them.
        detections=detections,  # The converted detections.
    )  # End of the response.


# Endpoints
# Health check: answers as long as the service runs.
@app.get("/health", response_model=HealthResponse, description="Health check endpoint.")
# Asynchronous handler of GET /health.
async def health() -> HealthResponse:
    # Report the status and the library version.
    return HealthResponse(status="healthy", version=unbihexium.__version__)


# Library information.
@app.get("/info", response_model=InfoResponse, description="Library information endpoint.")
# Asynchronous handler of GET /info.
async def info() -> InfoResponse:
    # Build the information response.
    return InfoResponse(
        name="unbihexium",  # Package name.
        version=unbihexium.__version__,  # Library version.
        # One-line description of the library.
        description="Python library for Earth observation, geospatial, remote sensing and SAR",
    )  # End of the information response.


# Ship detection on an uploaded GeoTIFF.
@app.post(
    "/detect/ships",  # Route of the endpoint.
    response_model=DetectionResponse,  # Schema of the response body.
    description=(  # Description shown in the OpenAPI documentation.
        "Detect ships in uploaded image.\n\n"  # Summary line.
        "Args:\n"  # Parameter section.
        "    file: GeoTIFF image file\n"  # Uploaded file.
        "    threshold: Detection confidence threshold (0.0-1.0)\n\n"  # Query parameter.
        "Returns:\n"  # Response section.
        "    DetectionResponse with ship detections"  # Response body.
    ),  # End of the description.
)  # End of the route decorator.
# Plain handler, run in the thread pool because inference blocks.
def detect_ships(
    file: UploadFile = File(...),  # Required multipart file upload.
    threshold: float = Query(0.5, ge=0.0, le=1.0),  # Minimum confidence.
) -> DetectionResponse:  # The handler returns a DetectionResponse.
    # Open the upload; the temporary file is always deleted.
    with uploaded_raster(file) as raster:
        # Detect the ships.
        return run_detector(ShipDetector(threshold=threshold), raster)


# Building detection on an uploaded GeoTIFF.
@app.post(
    "/detect/buildings",  # Route of the endpoint.
    response_model=DetectionResponse,  # Schema of the response body.
    description="Detect buildings in uploaded image.",  # OpenAPI description.
)  # End of the route decorator.
# Plain handler, run in the thread pool because inference blocks.
def detect_buildings(
    file: UploadFile = File(...),  # Required multipart file upload.
    threshold: float = Query(0.5, ge=0.0, le=1.0),  # Minimum confidence.
) -> DetectionResponse:  # The handler returns a DetectionResponse.
    # Open the upload; the temporary file is always deleted.
    with uploaded_raster(file) as raster:
        # Detect the buildings.
        return run_detector(BuildingDetector(threshold=threshold), raster)


# NDVI statistics of an uploaded multispectral GeoTIFF.
@app.post(
    "/index/ndvi",  # Route of the endpoint.
    response_model=IndexResponse,  # Schema of the response body.
    description=(  # Description shown in the OpenAPI documentation.
        "Calculate NDVI from uploaded multispectral image.\n\n"  # Summary line.
        "Args:\n"  # Parameter section.
        "    file: GeoTIFF image file with NIR and RED bands\n"  # Uploaded file.
        "    nir_band: Band number for NIR (1-indexed)\n"  # Query parameter.
        "    red_band: Band number for RED (1-indexed)\n\n"  # Query parameter.
        "Returns:\n"  # Response section.
        "    IndexResponse with NDVI statistics"  # Response body.
    ),  # End of the description.
)  # End of the route decorator.
# Plain handler, run in the thread pool because reading blocks.
def calculate_ndvi(
    nir_band: int = Query(4, ge=1),  # Near-infrared band number, counted from 1.
    red_band: int = Query(3, ge=1),  # Red band number, counted from 1.
    file: UploadFile = File(...),  # Required multipart file upload.
) -> IndexResponse:  # The handler returns an IndexResponse.
    # Open the upload; the temporary file is always deleted.
    with uploaded_raster(file) as raster:
        # Pixels as (bands, rows, columns).
        data = np.asarray(raster.data)
        # Number of bands of the upload.
        count = data.shape[0] if data.ndim == 3 else 1
        # Band numbers must exist in the image.
        if max(nir_band, red_band) > count:
            # Unprocessable content.
            raise HTTPException(
                status_code=422,  # Unprocessable content.
                detail=f"band numbers must be between 1 and {count}",  # Reason.
            )  # End of the error.
        # Single-band images have no band axis to index.
        data = data if data.ndim == 3 else data[None]
        # Bands converted from 1-based numbers to 0-based indices.
        bands = {"NIR": data[nir_band - 1], "RED": data[red_band - 1]}
        # Per-pixel NDVI = (NIR - RED) / (NIR + RED).
        try:
            # Index values.
            ndvi = np.asarray(compute_index("NDVI", bands), dtype=np.float64)
        # Anything else is a server error.
        except Exception as exc:
            # Internal error without internal details.
            raise HTTPException(status_code=500, detail="NDVI calculation failed") from exc
    # Images without a single valid pixel have no statistics.
    if not np.isfinite(ndvi).any():
        # Unprocessable content.
        raise HTTPException(status_code=422, detail="the image has no valid NDVI pixels")
    # Build the response body with NaN-aware statistics.
    return IndexResponse(
        index_name="NDVI",  # Name of the index.
        min_value=float(np.nanmin(ndvi)),  # Minimum, as a JSON number.
        max_value=float(np.nanmax(ndvi)),  # Maximum, as a JSON number.
        mean_value=float(np.nanmean(ndvi)),  # Mean, as a JSON number.
        shape=(ndvi.shape[0], ndvi.shape[1]),  # (rows, columns).
    )  # End of the response.


# Run a development server when the file is executed as a script.
if __name__ == "__main__":
    # Imported here so that importing the module does not require uvicorn.
    import uvicorn

    # Listen on all interfaces on port 8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)

# =============================================================================
# End of module examples/serving/api.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
