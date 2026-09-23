# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FastAPI serving module for unbihexium.

This module provides a REST API for model inference and pipeline execution.
"""

from unbihexium.serving.app import create_app
from unbihexium.serving.inference import ModelInferenceService
from unbihexium.serving.schemas import (
    CapabilitiesResponse,
    DetectionRequest,
    DetectionResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ModelsResponse,
)

__all__ = [
    "create_app",
    "ModelInferenceService",
    "HealthResponse",
    "CapabilitiesResponse",
    "ModelsResponse",
    "ModelInfo",
    "InferenceRequest",
    "InferenceResponse",
    "DetectionRequest",
    "DetectionResponse",
]
