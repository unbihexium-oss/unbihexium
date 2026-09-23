"""FastAPI serving example for unbihexium models.

This module provides a simple REST API for serving unbihexium models.
It demonstrates how to expose detection and index calculation capabilities
via HTTP endpoints.

Usage:
    uvicorn examples.serving.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          - Health check
    GET  /info            - Library information
    POST /detect/ships    - Ship detection
    POST /detect/buildings - Building detection
    POST /index/ndvi      - NDVI calculation
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import unbihexium
from unbihexium.ai.detection import BuildingDetector, ShipDetector
from unbihexium.core.index import compute_index
from unbihexium.core.raster import Raster

# FastAPI application
app = FastAPI(
    title="Unbihexium API",
    description="REST API for Earth Observation and Geospatial AI",
    version=unbihexium.__version__,
    license_info={"name": "MPL-2.0", "url": "https://mozilla.org/MPL/2.0/"},
)


# Response models
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class InfoResponse(BaseModel):
    """Library info response."""

    name: str
    version: str
    description: str


class Detection(BaseModel):
    """Single detection."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    """Detection response."""

    count: int
    model_id: str
    detections: list[Detection]


class IndexResponse(BaseModel):
    """Index calculation response."""

    index_name: str
    min_value: float
    max_value: float
    mean_value: float
    shape: tuple[int, int]


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=unbihexium.__version__)


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Library information endpoint."""
    return InfoResponse(
        name="unbihexium",
        version=unbihexium.__version__,
        description="Production-grade Earth Observation, Geospatial, Remote Sensing, and SAR Python library",
    )


@app.post("/detect/ships", response_model=DetectionResponse)
async def detect_ships(
    file: UploadFile = File(...),
    threshold: float = 0.5,
) -> DetectionResponse:
    """Detect ships in uploaded image.

    Args:
        file: GeoTIFF image file
        threshold: Detection confidence threshold (0.0-1.0)

    Returns:
        DetectionResponse with ship detections
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Load and process
        raster = Raster.from_file(tmp_path)
        detector = ShipDetector(threshold=threshold)
        result = detector.predict(raster)

        # Clean up
        tmp_path.unlink()

        # Convert to response
        detections = [
            Detection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in result.detections
        ]

        return DetectionResponse(
            count=result.count,
            model_id=result.model_id,
            detections=detections,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/buildings", response_model=DetectionResponse)
async def detect_buildings(
    file: UploadFile = File(...),
    threshold: float = 0.5,
) -> DetectionResponse:
    """Detect buildings in uploaded image."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        raster = Raster.from_file(tmp_path)
        detector = BuildingDetector(threshold=threshold)
        result = detector.predict(raster)

        tmp_path.unlink()

        detections = [
            Detection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
            )
            for d in result.detections
        ]

        return DetectionResponse(
            count=result.count,
            model_id=result.model_id,
            detections=detections,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/ndvi", response_model=IndexResponse)
async def calculate_ndvi(
    nir_band: int = 4,
    red_band: int = 3,
    file: UploadFile = File(...),
) -> IndexResponse:
    """Calculate NDVI from uploaded multispectral image.

    Args:
        file: GeoTIFF image file with NIR and RED bands
        nir_band: Band number for NIR (1-indexed)
        red_band: Band number for RED (1-indexed)

    Returns:
        IndexResponse with NDVI statistics
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        raster = Raster.from_file(tmp_path)
        raster.load()

        # Extract bands (convert to 0-indexed)
        nir = raster.data[nir_band - 1]
        red = raster.data[red_band - 1]

        # Compute NDVI
        bands = {"NIR": nir, "RED": red}
        ndvi = compute_index("NDVI", bands)

        tmp_path.unlink()

        return IndexResponse(
            index_name="NDVI",
            min_value=float(np.nanmin(ndvi)),
            max_value=float(np.nanmax(ndvi)),
            mean_value=float(np.nanmean(ndvi)),
            shape=(ndvi.shape[0], ndvi.shape[1]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
