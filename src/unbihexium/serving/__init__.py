# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/serving/__init__.py
# Title       : REST service for the model zoo
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires unbihexium[serving]
#               (FastAPI, Starlette, uvicorn)
# =============================================================================
#
# Abstract
# --------
#   app         create_app(): the FastAPI application
#   inference   ModelInferenceService: validation, model cache and JSON
#               summaries of predictions
#   schemas     request and response models of the OpenAPI document
#   security    body size limit, API key, rate limit
#
# Usage
# -----
#   uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Application factory.
from unbihexium.serving.app import create_app

# Inference service.
from unbihexium.serving.inference import (
    ModelInferenceService,  # Runs the models.
    PayloadTooLargeError,  # Inputs over the limits.
    UnknownModelError,  # Unknown model ids.
)  # End of the inference imports.

# Request and response models.
from unbihexium.serving.schemas import (
    CapabilitiesResponse,  # Capability list.
    DetectionRequest,  # Earlier detection request.
    DetectionResponse,  # Earlier detection response.
    HealthResponse,  # Health check.
    InferenceRequest,  # Earlier inference request.
    InferenceResponse,  # Earlier inference response.
    ModelInfo,  # One model.
    ModelsResponse,  # Model list.
    PredictRequest,  # Prediction request.
    PredictResponse,  # Prediction response.
)  # End of the schema imports.

# Public names of the package.
__all__ = [
    "CapabilitiesResponse",  # Capability list.
    "DetectionRequest",  # Earlier detection request.
    "DetectionResponse",  # Earlier detection response.
    "HealthResponse",  # Health check.
    "InferenceRequest",  # Earlier inference request.
    "InferenceResponse",  # Earlier inference response.
    "ModelInferenceService",  # Runs the models.
    "ModelInfo",  # One model.
    "ModelsResponse",  # Model list.
    "PayloadTooLargeError",  # Inputs over the limits.
    "PredictRequest",  # Prediction request.
    "PredictResponse",  # Prediction response.
    "UnknownModelError",  # Unknown model ids.
    "create_app",  # Application factory.
]  # End of the export list.


# =============================================================================
# End of module src/unbihexium/serving/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
