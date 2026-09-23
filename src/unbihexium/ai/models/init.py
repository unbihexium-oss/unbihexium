# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/ai/models/init.py
# Title       : Deterministic, platform-independent weight initialisation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch and NumPy
# =============================================================================
#
# Abstract
# --------
# Initialises the weights of a model zoo network so that the same model id
# always yields bit-identical weights on every platform and with every
# PyTorch version. PyTorch's own random initialisation cannot guarantee this,
# because its generators and default schemes change between releases.
#
# Method
# ------
#   seed     the first four bytes of SHA-256("unbihexium:" + model id),
#            read as an unsigned little-endian integer
#   stream   numpy.random.RandomState(seed); the legacy RandomState stream
#            is frozen by NumPy's compatibility policy (NEP 19), so the same
#            seed produces the same numbers in every NumPy release
#   order    modules are visited in the order of model.named_modules()
#   scheme   convolution and linear weights: He (Kaiming) normal with fan-in
#            and gain sqrt(2) (He et al., 2015), or N(0, init_std**2) when the
#            layer defines init_std; biases: zero, or the layer's init_bias;
#            group normalisation: weight one, bias zero
#
# The digest of a model is the SHA-256 of all tensors of its state dict in
# sorted key order, each written as its key, shape and little-endian float32
# bytes. The digest is recorded in the model zoo manifests, so a user can
# verify that a locally built or downloaded model is the published starter
# model.
#
# References
# ----------
#   He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving deep into rectifiers:
#     surpassing human-level performance on ImageNet classification. ICCV.
#   NumPy Enhancement Proposal 19 (2018). Random number generator policy.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# SHA-256 for seeds and digests.
import hashlib

# Square root for the He gain.
import math

# Stable random number stream and float32 conversion.
import numpy as np

# Tensor type and no-grad context.
import torch

# Layer types that are initialised.
from torch import nn

# Prefix of the seed material, so that seeds differ from other projects.
SEED_PREFIX = "unbihexium:"


# Derive the 32-bit seed of a model id.
def seed_for(model_id: str) -> int:
    # Hash the prefixed model id.
    digest = hashlib.sha256(f"{SEED_PREFIX}{model_id}".encode()).digest()
    # Read the first four bytes as an unsigned little-endian integer.
    return int.from_bytes(digest[:4], "little")


# Draw a float32 array of normal values with the given standard deviation.
def _normal(rng: np.random.RandomState, shape: tuple[int, ...], std: float) -> torch.Tensor:
    # Standard normal draws in float64, scaled and converted to float32.
    values = (rng.standard_normal(size=shape) * std).astype(np.float32)
    # Convert to a tensor that owns its memory.
    return torch.from_numpy(values.copy())


# Initialise every layer of a model deterministically from a seed.
@torch.no_grad()
def initialize(model: nn.Module, seed: int) -> nn.Module:
    # One random stream for the whole model.
    rng = np.random.RandomState(seed)
    # Visit modules in their definition order.
    for _, module in model.named_modules():
        # Convolution and fully connected layers.
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Fan-in: input channels times kernel area.
            fan_in = module.weight[0].numel()
            # Layer-specific standard deviation, or He normal.
            std = getattr(module, "init_std", None) or math.sqrt(2.0 / fan_in)
            # Replace the weights with deterministic values.
            module.weight.copy_(_normal(rng, tuple(module.weight.shape), std))
            # Initialise the bias when the layer has one.
            if module.bias is not None:
                # Constant bias: zero or the layer's requested value.
                module.bias.fill_(float(getattr(module, "init_bias", 0.0) or 0.0))
        # Group normalisation starts as the identity transform.
        elif isinstance(module, nn.GroupNorm):
            # Unit scale.
            module.weight.fill_(1.0)
            # Zero shift.
            module.bias.fill_(0.0)
    # Return the model for chaining.
    return model


# Compute the SHA-256 digest of a model's weights.
def weights_digest(model_or_state: nn.Module | dict[str, torch.Tensor]) -> str:
    # Accept both a module and a state dict.
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    # Incremental hash.
    hasher = hashlib.sha256()
    # Visit tensors in sorted key order.
    for key in sorted(state):
        # Tensor on the CPU as float32 in little-endian byte order.
        array = state[key].detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4")
        # Key and shape make the digest sensitive to the architecture.
        hasher.update(f"{key}:{tuple(array.shape)}\n".encode())
        # Raw tensor bytes.
        hasher.update(array.tobytes())
    # Hexadecimal digest.
    return hasher.hexdigest()


# Count the trainable parameters of a model.
def count_parameters(model: nn.Module) -> int:
    # Sum the element counts of parameters that require gradients.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# End of module src/unbihexium/ai/models/init.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
