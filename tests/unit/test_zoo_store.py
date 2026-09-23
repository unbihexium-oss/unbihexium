# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_zoo_store.py
# Title       : Tests of checkpoints, ONNX export and the local model store
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, PyTorch and ONNX Runtime
# =============================================================================
#
# Abstract
# --------
# Round-trips models through checkpoints, detects tampered checkpoints,
# exports models to ONNX and compares ONNX Runtime with PyTorch, and checks
# that the local store builds, caches, verifies and clears models in a
# temporary cache directory.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Test framework.
import pytest

# PyTorch is optional for the library; skip these tests without it.
torch = pytest.importorskip("torch")

# Model construction.
from unbihexium.ai.models import build_model  # noqa: E402 - imported after the skip check

# Store functions under test.
from unbihexium.zoo import (  # noqa: E402 - imported after the skip check
    clear_cache,  # Remove cached models.
    ensure_model,  # Cache a model.
    is_model_cached,  # Cache lookup.
    list_cached,  # Cache listing.
    load_model,  # Load a model.
    verify_directory,  # Verify file checksums.
    verify_model,  # Verify a cached model.
)  # End of the store imports.

# Checkpoint functions under test.
from unbihexium.zoo.checkpoint import (  # noqa: E402 - imported after the skip check
    CheckpointError,  # Raised for invalid checkpoints.
    load_checkpoint,  # Read a checkpoint.
    save_checkpoint,  # Write a checkpoint.
)  # End of the checkpoint imports.


# Checkpoints round-trip the weights and configuration exactly.
def test_checkpoint_round_trip(tmp_path: Path) -> None:
    # Customised model, so the configuration differs from the catalogue.
    model = build_model("crop_classifier", "tiny", outputs=["other", "wheat", "maize"])
    # Save it.
    digest = save_checkpoint(model, tmp_path / "model.pt", training={"epoch": 3})
    # Load it again.
    loaded = load_checkpoint(tmp_path / "model.pt")
    # Same weights.
    assert loaded.digest() == digest == model.digest()
    # Same configuration.
    assert loaded.config == model.config


# Modified weights are detected on load.
def test_checkpoint_tampering(tmp_path: Path) -> None:
    # Save a model.
    save_checkpoint(build_model("ndwi_calculator", "tiny"), tmp_path / "ok.pt")
    # Build another model and save it.
    model = build_model("water_surface_detector", "tiny")
    # Save the model.
    save_checkpoint(model, tmp_path / "m.pt")
    # Read the raw checkpoint.
    payload = torch.load(tmp_path / "m.pt", weights_only=True)
    # Change one weight.
    key = next(iter(payload["state_dict"]))
    # Add a small value.
    payload["state_dict"][key] += 1.0
    # Write the tampered file.
    torch.save(payload, tmp_path / "m.pt")
    # Loading fails the digest check.
    with pytest.raises(CheckpointError):
        # Load with verification.
        load_checkpoint(tmp_path / "m.pt")


# Non-checkpoint files are rejected.
def test_not_a_checkpoint(tmp_path: Path) -> None:
    # A plain tensor file.
    torch.save({"weights": torch.zeros(1)}, tmp_path / "x.pt")
    # Loading it fails.
    with pytest.raises(CheckpointError):
        # Load the invalid file.
        load_checkpoint(tmp_path / "x.pt")


# ONNX exports reproduce the PyTorch outputs.
@pytest.mark.parametrize(
    "family", ["ship_detector", "lulc_classifier", "yield_predictor", "ndvi_calculator"]
)
def test_onnx_export(tmp_path: Path, family: str) -> None:
    # ONNX Runtime and onnx are optional.
    pytest.importorskip("onnxruntime")
    # onnx is needed to write metadata.
    pytest.importorskip("onnx")
    # Export function; it verifies against ONNX Runtime.
    from unbihexium.zoo.export import export_onnx, read_onnx_config

    # Export the tiny model.
    path = export_onnx(build_model(family, "tiny"), tmp_path / "model.onnx")
    # The file exists.
    assert path.is_file()
    # The configuration is stored in the metadata.
    assert read_onnx_config(path)["family"] == family


# The store builds, caches, verifies and clears models.
def test_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Nothing is cached yet.
    assert list_cached() == []
    # Build and cache a model.
    directory = ensure_model("ndvi_calculator_tiny")
    # The checkpoint, configuration and checksums exist.
    assert (directory / "model.pt").is_file() and (directory / "config.json").is_file()
    # The checksums verify.
    assert all(verify_directory(directory).values())
    # The model is cached and verifies against the published digest.
    assert is_model_cached("ndvi_calculator_tiny") and verify_model("ndvi_calculator_tiny")
    # Loading by id returns the verified starter model.
    assert load_model("ship_detector_tiny").model_id == "ship_detector_tiny"
    # Loading by path returns the cached model.
    assert load_model(directory / "model.pt").model_id == "ndvi_calculator_tiny"
    # Clearing removes the model.
    assert clear_cache() == 1 and list_cached() == []


# =============================================================================
# End of module tests/unit/test_zoo_store.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
