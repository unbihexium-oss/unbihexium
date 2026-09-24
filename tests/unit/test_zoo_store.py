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
    ModelZooEntry,  # Registry entry.
    VerificationError,  # Raised for digest mismatches.
    clear_cache,  # Remove cached models.
    ensure_model,  # Cache a model.
    get_model,  # Registry lookup.
    is_model_cached,  # Cache lookup.
    list_cached,  # Cache listing.
    load_model,  # Load a model.
    register_model,  # Register a user model.
    unregister_model,  # Remove a user model.
    verify_directory,  # Verify file checksums.
    verify_model,  # Verify a cached model.
    write_sha256_file,  # Write checksums.
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


# Verification checks every file listed in model.sha256.
@pytest.mark.parametrize("name", ["config.json", "model.pt", "model.sha256"])
def test_verify_detects_modified_files(
    tmp_path: Path,  # Temporary directory of the test.
    monkeypatch: pytest.MonkeyPatch,  # Environment patching.
    name: str,  # File that is modified.
) -> None:  # The test returns nothing.
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Build and cache a model.
    directory = ensure_model("ndvi_calculator_tiny")
    # The untouched model verifies.
    assert verify_model("ndvi_calculator_tiny")
    # Append bytes to the file.
    with (directory / name).open("ab") as fh:
        # Extra bytes, and a line that lists a missing file for model.sha256.
        fh.write(b"0" * 64 + b"  model.onnx\n" if name == "model.sha256" else b" ")
    # The modification is detected.
    assert not verify_model("ndvi_calculator_tiny")


# A missing listed file or a missing checksum file fails verification.
def test_verify_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Build and cache a model.
    directory = ensure_model("ndvi_calculator_tiny")
    # Remove the configuration.
    (directory / "config.json").unlink()
    # Verification fails.
    assert not verify_model("ndvi_calculator_tiny")
    # Rebuild the entry.
    ensure_model("ndvi_calculator_tiny")
    # Remove the checksum file.
    (directory / "model.sha256").unlink()
    # Verification fails.
    assert not verify_model("ndvi_calculator_tiny")


# A corrupt checkpoint makes verification fail instead of raising.
def test_verify_corrupt_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Build and cache a model.
    directory = ensure_model("ndvi_calculator_tiny")
    # Replace the checkpoint with bytes that cannot be unpickled.
    (directory / "model.pt").write_bytes(b"not a checkpoint")
    # Checksums that match the corrupt file, so the loader is reached.
    write_sha256_file(directory, ["model.pt", "config.json"])
    # Verification returns False.
    assert verify_model("ndvi_calculator_tiny") is False


# A verified store entry is reused without writing any file.
def test_ensure_model_reuses_verified_entry(
    tmp_path: Path,  # Temporary directory of the test.
    monkeypatch: pytest.MonkeyPatch,  # Environment and attribute patching.
) -> None:  # The test returns nothing.
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Build and cache a model.
    directory = ensure_model("ndvi_calculator_tiny")
    # Any write now fails, as in a read-only store.
    with monkeypatch.context() as patch:
        # Text writes fail.
        patch.setattr(Path, "write_text", lambda *a, **k: pytest.fail("rewrote a file"))
        # Binary writes fail.
        patch.setattr(Path, "write_bytes", lambda *a, **k: pytest.fail("rewrote a file"))
        # Checkpoint writes fail.
        patch.setattr(torch, "save", lambda *a, **k: pytest.fail("rewrote the checkpoint"))
        # The verified entry is returned unchanged.
        assert ensure_model("ndvi_calculator_tiny") == directory
    # A modified configuration is detected and the entry is rebuilt.
    (directory / "config.json").write_text("{}", encoding="utf-8")
    # Rebuild.
    ensure_model("ndvi_calculator_tiny")
    # The rebuilt entry verifies.
    assert verify_model("ndvi_calculator_tiny")


# The ONNX export is used only when model.sha256 vouches for it.
def test_ensure_model_verifies_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ONNX Runtime and onnx are optional.
    pytest.importorskip("onnxruntime")
    # onnx is needed to write metadata.
    pytest.importorskip("onnx")
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path))
    # Build and export a model.
    directory = ensure_model("ndvi_calculator_tiny", onnx=True)
    # The export is listed and verifies.
    assert verify_directory(directory)["model.onnx"]
    # Tamper with the export.
    with (directory / "model.onnx").open("ab") as fh:
        # Extra bytes.
        fh.write(b" ")
    # Verification of the store entry fails.
    assert not verify_model("ndvi_calculator_tiny")
    # ensure_model replaces the tampered export.
    ensure_model("ndvi_calculator_tiny", onnx=True)
    # Every listed file verifies again.
    assert all(verify_directory(directory).values()) and verify_model("ndvi_calculator_tiny")


# Registered local and downloaded checkpoints are checked against their digest.
def test_registered_digest_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a temporary cache root.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path / "cache"))
    # Save a checkpoint of a catalogue model.
    save_checkpoint(build_model("ndvi_calculator", "tiny"), tmp_path / "local.pt")
    # Catalogue entry used as a template.
    base = get_model("ndvi_calculator_tiny")
    # Registered entry with a wrong digest.
    entry = ModelZooEntry(
        model_id="my_local_model",  # User model id.
        spec=base.spec,  # Specification.
        variant=base.variant,  # Variant.
        weights_digest="0" * 64,  # Wrong digest.
        source="local",  # Local checkpoint.
        local_path=str(tmp_path / "local.pt"),  # Path of the checkpoint.
    )  # End of the entry.
    # Register the entry.
    register_model(entry)
    # Remove the registration afterwards.
    try:
        # Loading fails.
        with pytest.raises(VerificationError):
            # Load with verification.
            load_model("my_local_model")
        # Caching fails and leaves no checkpoint.
        with pytest.raises(VerificationError):
            # Cache the model.
            ensure_model("my_local_model")
        # No checkpoint was kept.
        assert not is_model_cached("my_local_model")
        # With the right digest the model loads.
        entry.weights_digest = base.weights_digest
        # Load it.
        assert load_model("my_local_model").digest() == base.weights_digest
    # Clean up the registry.
    finally:
        # Remove the user entry.
        unregister_model("my_local_model")


# clear_cache refuses names that leave the store.
def test_clear_cache_rejects_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Store below the temporary directory.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(tmp_path / "cache"))
    # Directory outside the store.
    outside = tmp_path / "cache" / "x"
    # Create it with a file.
    outside.mkdir(parents=True)
    # Marker file.
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    # Parent references, separators and unknown names are rejected.
    for name in ["../x", "..", "a/b", "no_such_model"]:
        # Each is an error.
        with pytest.raises(ValueError):
            # Try to remove it.
            clear_cache(name)
    # The outside directory is untouched.
    assert (outside / "keep.txt").is_file()
    # A store entry that links outside is removed without touching the target.
    store = tmp_path / "cache" / "models"
    # Create the store.
    store.mkdir()
    # Link to the outside directory.
    (store / "linked").symlink_to(outside, target_is_directory=True)
    # Removing the entry removes the link only.
    assert clear_cache("linked") == 1 and (outside / "keep.txt").is_file()


# =============================================================================
# End of module tests/unit/test_zoo_store.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
