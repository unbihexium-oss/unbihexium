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
# Python      : CPython 3.10 to 3.14, requires unbihexium[serving]
# =============================================================================
#
# Abstract
# --------
# A simple REST API for serving Unbihexium models. It demonstrates how to
# expose detection and spectral index calculation over HTTP with FastAPI.
# Uploaded images are written to a temporary file, processed and deleted.
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
# This is an example, not a hardened service: it has no authentication, no
# upload size limit and returns internal error messages to the client. The
# production service is unbihexium.serving (see docker-compose.yml).
# Descriptions shown in the OpenAPI documentation are passed explicitly to
# the decorators and model configurations below.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Imported by the original example for in-memory file handling (unused).
import io

# Temporary files for the uploaded images.
import tempfile

# Represent the temporary file path.
from pathlib import Path

# Type used by the original example for generic payloads (unused).
from typing import Any

# Compute summary statistics that ignore NaN pixels.
import numpy as np

# Web framework, file upload support and HTTP errors.
from fastapi import FastAPI, File, HTTPException, UploadFile

# Imported by the original example for custom JSON responses (unused).
from fastapi.responses import JSONResponse

# Base class and configuration of the response models.
from pydantic import BaseModel, ConfigDict

# Library version for the health and info endpoints.
import unbihexium

# Detection model wrappers of Unbihexium.
from unbihexium.ai.detection import BuildingDetector, ShipDetector

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

    # Bounding box as (xmin, ymin, xmax, ymax).
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
        description="Production-grade Earth Observation, Geospatial, Remote Sensing, and SAR Python library",
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
# Asynchronous handler of POST /detect/ships.
async def detect_ships(
    file: UploadFile = File(...),  # Required multipart file upload.
    threshold: float = 0.5,  # Minimum confidence, a query parameter.
) -> DetectionResponse:  # The handler returns a DetectionResponse.
    # Turn any processing error into an HTTP 500 response.
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Read the whole upload into memory.
            content = await file.read()
            # Write it to the temporary file.
            tmp.write(content)
            # Keep the path; delete=False keeps the file after closing.
            tmp_path = Path(tmp.name)

        # Load and process
        raster = Raster.from_file(tmp_path)
        # Create the detector with the requested threshold.
        detector = ShipDetector(threshold=threshold)
        # Run the detection on the whole raster.
        result = detector.predict(raster)

        # Clean up
        tmp_path.unlink()

        # Convert to response
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

    # Report the error message to the client.
    except Exception as e:
        # HTTP 500 Internal Server Error with the message as detail.
        raise HTTPException(status_code=500, detail=str(e))


# Building detection on an uploaded GeoTIFF.
@app.post(
    "/detect/buildings",  # Route of the endpoint.
    response_model=DetectionResponse,  # Schema of the response body.
    description="Detect buildings in uploaded image.",  # OpenAPI description.
)  # End of the route decorator.
# Asynchronous handler of POST /detect/buildings.
async def detect_buildings(
    file: UploadFile = File(...),  # Required multipart file upload.
    threshold: float = 0.5,  # Minimum confidence, a query parameter.
) -> DetectionResponse:  # The handler returns a DetectionResponse.
    # Turn any processing error into an HTTP 500 response.
    try:
        # Store the upload in a temporary GeoTIFF file.
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Read the whole upload into memory.
            content = await file.read()
            # Write it to the temporary file.
            tmp.write(content)
            # Keep the path; delete=False keeps the file after closing.
            tmp_path = Path(tmp.name)

        # Open the uploaded raster.
        raster = Raster.from_file(tmp_path)
        # Create the detector with the requested threshold.
        detector = BuildingDetector(threshold=threshold)
        # Run the detection on the whole raster.
        result = detector.predict(raster)

        # Delete the temporary file.
        tmp_path.unlink()

        # Convert the detections to response items.
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

    # Report the error message to the client.
    except Exception as e:
        # HTTP 500 Internal Server Error with the message as detail.
        raise HTTPException(status_code=500, detail=str(e))


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
# Asynchronous handler of POST /index/ndvi.
async def calculate_ndvi(
    nir_band: int = 4,  # Near-infrared band number, counted from 1.
    red_band: int = 3,  # Red band number, counted from 1.
    file: UploadFile = File(...),  # Required multipart file upload.
) -> IndexResponse:  # The handler returns an IndexResponse.
    # Turn any processing error into an HTTP 500 response.
    try:
        # Store the upload in a temporary GeoTIFF file.
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            # Read the whole upload into memory.
            content = await file.read()
            # Write it to the temporary file.
            tmp.write(content)
            # Keep the path; delete=False keeps the file after closing.
            tmp_path = Path(tmp.name)

        # Open the uploaded raster.
        raster = Raster.from_file(tmp_path)
        # Read the pixel data into memory.
        raster.load()

        # Extract bands (convert to 0-indexed)
        nir = raster.data[nir_band - 1]
        # Red band, converted from a 1-based band number to a 0-based index.
        red = raster.data[red_band - 1]

        # Compute NDVI
        bands = {"NIR": nir, "RED": red}
        # Per-pixel NDVI = (NIR - RED) / (NIR + RED).
        ndvi = compute_index("NDVI", bands)

        # Delete the temporary file.
        tmp_path.unlink()

        # Build the response body with NaN-aware statistics.
        return IndexResponse(
            index_name="NDVI",  # Name of the index.
            min_value=float(np.nanmin(ndvi)),  # Minimum, as a JSON number.
            max_value=float(np.nanmax(ndvi)),  # Maximum, as a JSON number.
            mean_value=float(np.nanmean(ndvi)),  # Mean, as a JSON number.
            shape=(ndvi.shape[0], ndvi.shape[1]),  # (rows, columns).
        )  # End of the response.

    # Report the error message to the client.
    except Exception as e:
        # HTTP 500 Internal Server Error with the message as detail.
        raise HTTPException(status_code=500, detail=str(e))


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
