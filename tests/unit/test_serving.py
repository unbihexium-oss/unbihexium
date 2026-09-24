# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_serving.py
# Title       : Tests of the REST service, its inference service and security
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, FastAPI, httpx and
#               PyTorch (skipped when missing)
# =============================================================================
#
# Abstract
# --------
# The NDVI model computes (NIR - red) / (NIR + red) exactly, so the /predict
# endpoint is checked against hand-computed values: red 0.1 and NIR 0.3 give
# 0.5, equal bands give 0, red 0.1 and NIR 0.9 give 0.8, and a no-data
# pixel is null. Result summaries are checked on hand-made result records
# (map boxes from the affine transform, class pixel counts, fractions and
# areas). The tests also cover band-count validation (422), unknown models
# (404), pixel and body limits (413), pickled .npy rejection, API keys
# (401/403), the token bucket rate limiter (429), CORS and the OpenAPI
# descriptions. Starter models (untrained) are only checked for the
# structure of their output.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Run the ASGI middleware directly.
import asyncio

# Base64 encoding of .npy files.
import base64

# In-memory .npy files.
import io

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Test framework.
import pytest

# The service needs FastAPI, the test client needs httpx, models need PyTorch.
pytest.importorskip("fastapi")
# HTTP client of the test client.
pytest.importorskip("httpx")
# Model backend.
pytest.importorskip("torch")

# HTTP errors.
from fastapi import HTTPException

# Test client of FastAPI.
from fastapi.testclient import TestClient

# Result records for the summary tests.
from unbihexium.ai.results import (
    Detection,  # One box.
    DetectionResult,  # Boxes.
    RegressionResult,  # Values.
    SegmentationResult,  # Class map.
)  # End of the result imports.

# Service settings.
from unbihexium.config import ServingConfig

# Application factory.
from unbihexium.serving import create_app

# Inference service under test.
from unbihexium.serving.inference import (
    ModelInferenceService,  # Service.
    PayloadTooLargeError,  # Limit errors.
    json_safe,  # JSON conversion.
    value_statistics,  # Statistics.
)  # End of the inference imports.

# Security helpers under test.
from unbihexium.serving.security import (
    APIKeyAuth,  # API keys.
    RateLimiter,  # Rate limiting.
    RequestSizeLimitMiddleware,  # Body limit.
    validate_content_type,  # Media types.
)  # End of the security imports.

# Red band of the NDVI test image; the last pixel is no-data.
RED = [[0.1, 0.2], [0.1, -1.0]]
# Near-infrared band of the NDVI test image.
NIR = [[0.3, 0.2], [0.9, -1.0]]
# Expected NDVI: (0.3 - 0.1) / 0.4, 0 / 0.4, (0.9 - 0.1) / 1.0 and no-data.
NDVI = [[0.5, 0.0], [0.8, None]]


# Small limits so that the limit tests need small requests.
@pytest.fixture(scope="module")
def client() -> TestClient:
    # Settings with a 4 KiB body limit.
    config = ServingConfig(max_request_bytes=4096, max_pixels=64, max_values=256)
    # Service with the same image limits.
    service = ModelInferenceService(max_pixels=64, max_values=256, cache_size=2)
    # Test client of the application.
    return TestClient(create_app(config=config, service=service))


# Encode an array as a base64 .npy file.
def npy_base64(array: Any, allow_pickle: bool = False) -> str:
    # In-memory file.
    buffer = io.BytesIO()
    # NumPy format.
    np.save(buffer, array, allow_pickle=allow_pickle)
    # Base64 text.
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


# Health, models, capabilities and pipelines.
def test_discovery(client: TestClient) -> None:
    # Health check.
    health = client.get("/health").json()
    # Healthy with every zoo model available.
    assert health["status"] == "healthy" and health["ready"] and health["models_available"] == 520
    # Spectral index models: 7 families in 4 variants.
    page = client.get("/models", params={"task": "spectral_index", "limit": 5, "offset": 25}).json()
    # 28 matches, 3 on the last page.
    assert page["total"] == 28 and page["count"] == 3 and page["offset"] == 25
    # One model.
    ndvi = client.get("/models/ndvi_calculator_tiny").json()
    # Bands and outputs.
    assert ndvi["channels"] == ["red", "nir"] and ndvi["outputs"] == ["ndvi"]
    # The formula needs no training.
    assert ndvi["requires_training"] is False and ndvi["in_channels"] == 2
    # Unknown models.
    assert client.get("/models/nothing_here").status_code == 404
    # Invalid filters.
    assert client.get("/models", params={"task": "astrology"}).status_code == 422
    # Capabilities of the input and output domain.
    caps = client.get("/capabilities", params={"domain": "io"}).json()
    # Five formats.
    assert caps["count"] == 5
    # One capability with its models.
    ship = client.get("/capabilities/ship_detector").json()
    # Pipeline and model ids.
    assert ship["pipeline_id"] == "ship_detection" and "ship_detector_tiny" in ship["models"]
    # Unknown capabilities.
    assert client.get("/capabilities/nothing").status_code == 404
    # Pipelines registered by the task APIs.
    ids = [p["pipeline_id"] for p in client.get("/pipelines").json()["pipelines"]]
    # Includes the ship detection pipeline.
    assert "ship_detection" in ids


# Every route and request field is documented in the OpenAPI document.
def test_openapi_descriptions(client: TestClient) -> None:
    # OpenAPI document.
    spec = client.get("/openapi.json").json()
    # Every operation has a description.
    for path, operations in spec["paths"].items():
        # Every method of the path.
        for method, operation in operations.items():
            # Non-empty description.
            assert operation.get("description"), f"{method.upper()} {path}"
    # Fields of the prediction request.
    fields = spec["components"]["schemas"]["PredictRequest"]["properties"]
    # Every field is described.
    assert all(f.get("description") for f in fields.values())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


# NDVI against hand-computed values, from JSON lists and from .npy files.
def test_predict_ndvi(client: TestClient) -> None:
    # Request with a no-data value and the values in the response.
    body = {"image": [RED, NIR], "nodata": -1.0, "parameters": {"return_values": True}}
    # Run the model.
    response = client.post("/predict/ndvi_calculator_tiny", json=body)
    # Success.
    assert response.status_code == 200
    # Response document.
    data = response.json()
    # Metadata.
    assert data["task"] == "spectral_index" and data["input_shape"] == [2, 2, 2]
    # Values of the single output (float32 precision).
    values = data["result"]["values"][0]
    # Valid pixels.
    assert values[0] == pytest.approx(NDVI[0], abs=1e-6)
    # Third pixel and the no-data pixel as null.
    assert values[1][0] == pytest.approx(0.8, abs=1e-6) and values[1][1] is None
    # Statistics of the three valid pixels.
    stats = data["result"]["statistics"]["ndvi"]
    # Mean (0.5 + 0 + 0.8) / 3, minimum 0, maximum 0.8.
    assert stats["mean"] == pytest.approx(1.3 / 3, abs=1e-6) and stats["missing"] == 1
    # Range.
    assert stats["min"] == pytest.approx(0.0, abs=1e-6) and stats["max"] == pytest.approx(0.8)
    # The same image as a float32 .npy file.
    encoded = npy_base64(np.array([RED, NIR], dtype=np.float32))
    # Request with the file.
    body = {"image_npy_base64": encoded, "nodata": -1.0}
    # Same statistics.
    again = client.post("/predict/ndvi_calculator_tiny", json=body).json()["result"]
    # Mean of the valid pixels.
    assert again["statistics"]["ndvi"]["mean"] == pytest.approx(1.3 / 3, abs=1e-6)
    # Values are omitted unless requested.
    assert "values" not in again


# Invalid requests give 404, 413 and 422 with clear messages.
def test_predict_errors(client: TestClient) -> None:
    # Unknown model.
    response = client.post("/predict/nothing_tiny", json={"image": [RED, NIR]})
    # Not found.
    assert response.status_code == 404
    # Wrong band count for NDVI.
    response = client.post("/predict/ndvi_calculator_tiny", json={"image": [RED]})
    # Unprocessable with the expected bands.
    assert response.status_code == 422 and "expects 2 bands (red, nir), got 1" in response.text
    # 9 x 9 = 81 pixels exceed the limit of 64.
    big = np.zeros((2, 9, 9)).tolist()
    # Content too large.
    assert client.post("/predict/ndvi_calculator_tiny", json={"image": big}).status_code == 413
    # Bodies over 4 KiB are rejected before parsing.
    padding = {"image": [RED, NIR], "crs": "x" * 5000}
    # Content too large.
    assert client.post("/predict/ndvi_calculator_tiny", json=padding).status_code == 413
    # Pickled .npy files are refused (no code execution).
    pickled = npy_base64(np.array([{"a": 1}], dtype=object), allow_pickle=True)
    # Unprocessable.
    response = client.post("/predict/ndvi_calculator_tiny", json={"image_npy_base64": pickled})
    # The message names the problem.
    assert response.status_code == 422 and "not a valid .npy file" in response.text
    # Both encodings at once.
    both = {"image": [RED, NIR], "image_npy_base64": pickled}
    # Unprocessable.
    assert client.post("/predict/ndvi_calculator_tiny", json=both).status_code == 422
    # Thresholds outside [0, 1] fail request validation.
    bad = {"image": [RED, NIR], "parameters": {"threshold": 2.0}}
    # Unprocessable.
    assert client.post("/predict/ndvi_calculator_tiny", json=bad).status_code == 422
    # Ragged lists.
    ragged = {"image": [[[1.0, 2.0]], [[1.0]]]}
    # Unprocessable.
    assert client.post("/predict/ndvi_calculator_tiny", json=ragged).status_code == 422


# Starter models of other tasks return well-formed summaries.
def test_predict_other_tasks(client: TestClient) -> None:
    # Four-band 4 x 4 image with 10 m pixels.
    image = np.linspace(0, 1, 64).reshape(4, 4, 4).tolist()
    # Water segmentation with georeferencing and the class map.
    body = {"image": image, "transform": [10, 0, 0, 0, -10, 0], "parameters": {"return_mask": True}}
    # Run the starter model.
    result = client.post("/predict/water_surface_detector_tiny", json=body).json()["result"]
    # Every pixel is classified.
    assert sum(result["class_pixels"].values()) == 16 and result["nodata_pixels"] == 0
    # Areas: 16 pixels of 100 square metres.
    assert sum(result["class_areas"].values()) == pytest.approx(1600.0)
    # Fractions sum to one.
    assert sum(result["class_fractions"].values()) == pytest.approx(1.0)
    # Class map of the image size.
    assert np.asarray(result["mask"]).shape == (4, 4)
    # Detection with three bands.
    boxes = client.post("/predict/ship_detector_tiny", json={"image": image[:3]}).json()
    # Count and list agree; pixel coordinates without a transform.
    assert boxes["result"]["count"] == len(boxes["result"]["detections"])
    # No map coordinates.
    assert boxes["result"]["crs"] == "pixel"
    # Earlier detection endpoint.
    legacy = client.post("/detect/ship_detector_tiny", json={"image_data": image[:3]}).json()
    # Earlier layout.
    assert set(legacy) == {"model_id", "count", "detections"}
    # Earlier segmentation endpoint.
    seg = client.post("/segment/water_surface_detector_tiny", json={"image_data": image}).json()
    # Mask shape and classes.
    assert seg["mask_shape"] == [4, 4] and seg["classes"] == ["background", "water"]
    # A segmentation model on the detection endpoint.
    wrong = client.post("/detect/water_surface_detector_tiny", json={"image_data": image})
    # Unprocessable.
    assert wrong.status_code == 422
    # Earlier single-band endpoint with a one-band elevation model.
    single = client.post("/infer/viewshed_analyzer_tiny", json={"data": image[0]}).json()
    # Dense regression summary.
    assert single["success"] and single["result"]["result"]["shape"][1:] == [4, 4]
    # Errors are reported in the body there.
    failed = client.post("/infer/nothing_tiny", json={"data": image[0]}).json()
    # Failure with a message.
    assert failed["success"] is False and "unknown model" in failed["error"]


# ---------------------------------------------------------------------------
# Result summaries
# ---------------------------------------------------------------------------


# Hand-made result records give hand-computed summaries.
def test_summaries() -> None:
    # Service with default limits.
    service = ModelInferenceService(max_values=100)
    # 10 m pixels with the origin at (1000, 2000).
    transform = (10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    # One box from pixel (2, 3) to (6, 5).
    boxes = DetectionResult([Detection((2.0, 3.0, 6.0, 5.0), 0.9, 0, "ship")], model_id="m")
    # Summary with map coordinates and GeoJSON.
    det = service.summarise(boxes, transform, "EPSG:32632", {"return_geojson": True})
    # x: 1000 + 10 * (2, 6); y: 2000 - 10 * (5, 3).
    assert det["detections"][0]["geo_bbox"] == [1020.0, 1950.0, 1060.0, 1970.0]
    # Counts per class.
    assert det["counts_by_class"] == {"ship": 1} and det["crs"] == "EPSG:32632"
    # GeoJSON ring in map coordinates.
    ring = det["geojson"]["features"][0]["geometry"]["coordinates"][0]
    # First corner.
    assert ring[0] == [1020.0, 1950.0]
    # Class map with one no-data pixel.
    mask = np.array([[0, 1], [1, 255]], dtype=np.uint8)
    # Summary with the class map.
    seg = service.summarise(SegmentationResult(mask, ["land", "water"]), transform, None, {})
    # Pixel counts.
    assert seg["class_pixels"] == {"land": 1, "water": 2} and seg["nodata_pixels"] == 1
    # Fractions of the three valid pixels.
    assert seg["class_fractions"] == pytest.approx({"land": 1 / 3, "water": 2 / 3})
    # Areas of 100 square metres per pixel.
    assert seg["class_areas"] == {"land": 100.0, "water": 200.0}
    # Scene values.
    scene = RegressionResult(np.array([4.5], dtype=np.float32), ["yield"], ["t ha-1"])
    # Named values.
    assert service.summarise(scene) == {
        "outputs": ["yield"],  # Output names.
        "units": ["t ha-1"],  # Units.
        "values": {"yield": 4.5},  # Value by name.
    }  # End of the expected summary.
    # Returned arrays respect the value limit (11 x 11 = 121 > 100).
    dense = RegressionResult(np.zeros((1, 11, 11), dtype=np.float32), ["h"])
    # Too large to return.
    with pytest.raises(PayloadTooLargeError):
        # Values requested.
        service.summarise(dense, params={"return_values": True})


# JSON conversion and statistics.
def test_json_helpers() -> None:
    # NaN and infinity become null, NumPy scalars become Python numbers.
    converted = json_safe({"a": np.float32("nan"), "b": [np.int64(3), float("inf")], 1: (2.5,)})
    # Expected document.
    assert converted == {"a": None, "b": [3, None], "1": [2.5]}
    # Statistics of 1, 2, 3 and a NaN.
    stats = value_statistics(np.array([1.0, 2.0, 3.0, np.nan]))
    # Mean 2, population standard deviation sqrt(2/3).
    assert stats["mean"] == 2.0 and stats["std"] == pytest.approx(np.sqrt(2 / 3))
    # Counts.
    assert stats["count"] == 3 and stats["missing"] == 1
    # Empty selections.
    assert value_statistics(np.array([np.nan]))["mean"] is None


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


# API keys protect every route except /health.
def test_api_key() -> None:
    # Application with a key.
    app = create_app(config=ServingConfig(api_key="s3cret"))
    # Client without a default key.
    anonymous = TestClient(app)
    # Health needs no key.
    assert anonymous.get("/health").status_code == 200
    # Missing key.
    assert anonymous.get("/models").status_code == 401
    # Wrong key.
    assert anonymous.get("/models", headers={"X-API-Key": "guess"}).status_code == 403
    # Right key.
    assert anonymous.get("/models", headers={"X-API-Key": "s3cret"}).status_code == 200
    # Disabled authentication accepts everything.
    assert APIKeyAuth(None)(None) is None  # type: ignore[arg-type]


# JSON routes reject other media types with 415, as README.md states.
def test_json_media_type(client: TestClient) -> None:
    # A JSON body sent as plain text.
    body = '{"image": [[[0.1, 0.2]], [[0.5, 0.6]]]}'
    # Route of the prediction.
    url = "/predict/ndvi_calculator_tiny"
    # Plain text is refused before the body is parsed.
    response = client.post(url, content=body, headers={"Content-Type": "text/plain"})
    # Unsupported media type.
    assert response.status_code == 415
    # The same body as JSON is accepted.
    response = client.post(url, content=body, headers={"Content-Type": "application/json"})
    # Success.
    assert response.status_code == 200
    # Structured syntax suffixes such as application/geo+json count as JSON.
    response = client.post(url, content=body, headers={"Content-Type": "application/geo+json"})
    # Success.
    assert response.status_code == 200


# Token bucket: 60 requests per minute with bursts of 2.
def test_rate_limiter() -> None:
    # Fake clock.
    now = [100.0]
    # Limiter with the fake clock.
    limiter = RateLimiter(60, burst=2, clock=lambda: now[0])
    # Two requests of the burst.
    assert limiter.acquire("a") == 0.0 and limiter.acquire("a") == 0.0
    # The third waits one second (one token per second).
    assert limiter.acquire("a") == pytest.approx(1.0)
    # Other clients have their own bucket.
    assert limiter.acquire("b") == 0.0
    # Half a second later half a token is back.
    now[0] += 0.5
    # Wait for the other half.
    assert limiter.acquire("a") == pytest.approx(0.5)
    # One more second refills a token.
    now[0] += 1.0
    # Allowed again.
    assert limiter.acquire("a") == 0.0
    # Rates must be positive.
    with pytest.raises(ValueError):
        # Zero rate.
        RateLimiter(0)
    # Application limited to 2 requests per minute and client.
    limited = TestClient(create_app(config=ServingConfig(rate_limit_per_minute=2)))
    # Two requests pass.
    assert [limited.get("/capabilities/io_zarr").status_code for _ in range(2)] == [200, 200]
    # The third is refused with a waiting time.
    response = limited.get("/capabilities/io_zarr")
    # Too many requests.
    assert response.status_code == 429 and int(response.headers["Retry-After"]) >= 1


# Chunked bodies without Content-Length are counted as they arrive.
def test_request_size_middleware_streaming() -> None:
    # Messages sent by the middleware.
    sent: list[dict[str, Any]] = []

    # Application that reads the whole body, then answers 200.
    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        # Read until the last chunk.
        while (await receive()).get("more_body"):
            # Keep reading.
            pass
        # Answer.
        await send({"type": "http.response.start", "status": 200, "headers": []})

    # Three chunks of 100 bytes.
    chunks = [{"type": "http.request", "body": b"x" * 100, "more_body": i < 2} for i in range(3)]

    # Deliver the chunks in order.
    async def receive() -> dict[str, Any]:
        # Next chunk.
        return chunks.pop(0)

    # Record what the middleware sends.
    async def send(message: dict[str, Any]) -> None:
        # Store it.
        sent.append(message)

    # Limit of 250 bytes: the third chunk exceeds it.
    middleware = RequestSizeLimitMiddleware(app, max_size=250)
    # Run one request without Content-Length.
    asyncio.run(middleware({"type": "http", "headers": []}, receive, send))
    # The middleware answered 413.
    assert sent[0]["status"] == 413


# Media types and CORS.
def test_content_type_and_cors(client: TestClient) -> None:
    # JSON with a charset is accepted.
    validate_content_type("application/json; charset=utf-8")
    # Plain text is not.
    with pytest.raises(HTTPException) as info:
        # Unsupported type.
        validate_content_type("text/plain")
    # 415 Unsupported Media Type.
    assert info.value.status_code == 415
    # Preflight request from a browser.
    headers = {"Origin": "https://example.org", "Access-Control-Request-Method": "POST"}
    # CORS answer.
    response = client.options("/predict/ndvi_calculator_tiny", headers=headers)
    # Wildcard origin without credentials.
    assert response.headers["access-control-allow-origin"] == "*"
    # Credentials are never allowed with a wildcard.
    assert "access-control-allow-credentials" not in response.headers


# =============================================================================
# End of module tests/unit/test_serving.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
