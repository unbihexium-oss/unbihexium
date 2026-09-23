# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/__init__.py
# Title       : Model zoo network architectures
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# Public interface of the network architectures: build_model creates any of
# the 520 model zoo models with deterministic starter weights; the network
# classes can also be used directly for custom architectures.
# =============================================================================

# Building blocks.
from unbihexium.ai.models.blocks import (
    ConvNormAct,  # Convolution, group normalisation and ReLU.
    DownStage,  # Strided encoder stage.
    HeadConv,  # Prediction head.
    RangeActivation,  # Sigmoid scaled to a value range.
    ResidualBlock,  # Residual block with an identity shortcut.
    UpStage,  # Decoder stage with a skip connection.
)  # End of the block imports.

# Encoder and decoder.
from unbihexium.ai.models.encoder import (
    Decoder,  # U-Net decoder.
    Encoder,  # Residual encoder.
    stage_channels,  # Channel widths of the encoder levels.
)  # End of the encoder imports.

# Model construction from the catalogue.
from unbihexium.ai.models.factory import (
    BuildConfig,  # Input and output description of a model.
    ZooModel,  # Network with its configuration.
    build_from_config,  # Build a model from a configuration.
    build_model,  # Build a model by name.
)  # End of the factory imports.

# Deterministic initialisation and digests.
from unbihexium.ai.models.init import (
    count_parameters,  # Number of trainable parameters.
    initialize,  # Deterministic weight initialisation.
    seed_for,  # Seed of a model id.
    weights_digest,  # SHA-256 digest of the weights.
)  # End of the initialisation imports.

# Task networks.
from unbihexium.ai.models.networks import (
    DETECTION_STRIDE,  # Output stride of the detector.
    CenterNet,  # Object detector.
    SceneRegressor,  # Scene-level regressor.
    SuperResolutionNet,  # Super-resolution network.
    UNet,  # Dense per-pixel network.
)  # End of the network imports.

# Spectral index modules.
from unbihexium.ai.models.spectral import (
    FORMULA_CHANNELS,  # Channel count of each formula.
    SpectralIndex,  # Formula as a network module.
    safe_divide,  # Division with NaN for zero denominators.
)  # End of the spectral imports.

# Names exported by `from unbihexium.ai.models import *`.
__all__ = [
    "DETECTION_STRIDE",  # Output stride of the detector.
    "FORMULA_CHANNELS",  # Channel count of each formula.
    "BuildConfig",  # Input and output description of a model.
    "CenterNet",  # Object detector.
    "ConvNormAct",  # Convolution, normalisation and activation.
    "Decoder",  # U-Net decoder.
    "DownStage",  # Strided encoder stage.
    "Encoder",  # Residual encoder.
    "HeadConv",  # Prediction head.
    "RangeActivation",  # Sigmoid scaled to a value range.
    "ResidualBlock",  # Residual block.
    "SceneRegressor",  # Scene-level regressor.
    "SpectralIndex",  # Formula as a network module.
    "SuperResolutionNet",  # Super-resolution network.
    "UNet",  # Dense per-pixel network.
    "UpStage",  # Decoder stage.
    "ZooModel",  # Network with its configuration.
    "build_from_config",  # Build from a configuration.
    "build_model",  # Build by name.
    "count_parameters",  # Number of parameters.
    "initialize",  # Deterministic initialisation.
    "safe_divide",  # Division with NaN.
    "seed_for",  # Seed of a model id.
    "stage_channels",  # Encoder channel widths.
    "weights_digest",  # Digest of the weights.
]  # End of the export list.

# =============================================================================
# End of module src/unbihexium/ai/models/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
