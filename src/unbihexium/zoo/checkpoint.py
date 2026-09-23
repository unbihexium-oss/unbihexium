# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/checkpoint.py
# Title       : Safe saving and loading of model checkpoints
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch (unbihexium[torch])
# =============================================================================
#
# Abstract
# --------
# A checkpoint is a PyTorch file (model.pt) containing only plain data: the
# format marker, the BuildConfig of the model as a dictionary, the state dict
# of the weights, the weights digest and optional training state. Checkpoints
# are loaded with torch.load(weights_only=True), which refuses to unpickle
# arbitrary Python objects, so loading a checkpoint from an untrusted source
# cannot execute code.
#
# Layout
# ------
#   {
#     "format": "unbihexium-checkpoint",
#     "format_version": 1,
#     "config": BuildConfig.to_dict(),
#     "state_dict": {name: tensor, ...},
#     "weights_digest": "<sha256>",
#     "training": {...}            # optional: epoch, metrics, optimiser state
#   }
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Serialisation.
import torch

# Model configuration and construction.
from unbihexium.ai.models.factory import ZooModel, build_from_config

# Digest of the weights.
from unbihexium.ai.models.init import weights_digest

# Configuration record of a model.
from unbihexium.zoo.config import BuildConfig

# Format marker of Unbihexium checkpoints.
CHECKPOINT_FORMAT = "unbihexium-checkpoint"

# Current version of the checkpoint layout.
CHECKPOINT_VERSION = 1


# Raised when a file is not a valid Unbihexium checkpoint.
class CheckpointError(ValueError):
    # No behaviour beyond ValueError; the class exists for precise handling.
    pass


# Save a model to a checkpoint file and return the weights digest.
def save_checkpoint(
    model: ZooModel,  # Model to save.
    path: str | Path,  # Destination file.
    training: dict[str, Any] | None = None,  # Optional training state.
) -> str:  # Digest of the saved weights.
    # Normalise the path.
    path = Path(path)
    # Create the parent directory if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Weights on the CPU, detached from any autograd graph.
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Digest of the weights.
    digest = weights_digest(state)
    # Assemble the plain-data checkpoint.
    payload = {
        "format": CHECKPOINT_FORMAT,  # Format marker.
        "format_version": CHECKPOINT_VERSION,  # Layout version.
        "config": model.config.to_dict(),  # Model description.
        "state_dict": state,  # Weights.
        "weights_digest": digest,  # Digest of the weights.
        "training": training or {},  # Optional training state.
    }  # End of the checkpoint.
    # Write to a temporary file first so that a crash leaves no partial file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Serialise the checkpoint.
    torch.save(payload, tmp)
    # Atomically move the file into place.
    tmp.replace(path)
    # Return the digest for manifests and logs.
    return digest


# Read a checkpoint file and validate its structure.
def read_checkpoint(path: str | Path) -> dict[str, Any]:
    # weights_only=True refuses arbitrary pickled objects.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    # The payload must be a dictionary with the format marker.
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        # Report the invalid file.
        raise CheckpointError(f"{path} is not an Unbihexium checkpoint")
    # Newer layouts cannot be read by this version.
    if int(payload.get("format_version", 0)) > CHECKPOINT_VERSION:
        # Ask the user to upgrade.
        raise CheckpointError(f"{path} was written by a newer Unbihexium; please upgrade")
    # Return the validated payload.
    return payload


# Load a model from a checkpoint file.
def load_checkpoint(path: str | Path, verify: bool = True) -> ZooModel:
    # Read and validate the file.
    payload = read_checkpoint(path)
    # Rebuild the configuration.
    config = BuildConfig.from_dict(payload["config"])
    # Build the architecture without initialisation; the weights follow.
    model = build_from_config(config, initialise=False)
    # Load the weights; strict loading catches architecture mismatches.
    model.load_state_dict(payload["state_dict"], strict=True)
    # Optionally compare the weights with the recorded digest.
    if verify and payload.get("weights_digest"):
        # Recompute the digest from the loaded weights.
        actual = weights_digest(model)
        # A mismatch means the file was modified or corrupted.
        if actual != payload["weights_digest"]:
            # Report the corrupted checkpoint.
            raise CheckpointError(f"{path}: weights do not match the recorded digest")
    # Loaded models start in evaluation mode.
    return model.eval()


# =============================================================================
# End of module src/unbihexium/zoo/checkpoint.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
