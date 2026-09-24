# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/store.py
# Title       : Local model store: build, cache, download and load models
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyTorch for building models
# =============================================================================
#
# Abstract
# --------
# Every one of the 520 model zoo models can be obtained by any user of the
# library, without large downloads: the starter weights are generated
# locally and deterministically from the model id (see models.init), then
# verified against the published digest. Fine-tuned models registered with
# a download URL or a local path are fetched or read instead, and their
# weights are checked against the registered digest when the entry has one.
#
# ensure_model reuses a cached model only when every file listed in its
# model.sha256 matches and config.json describes the current entry; any
# other directory is rebuilt. Unchanged files are never rewritten, so a
# verified store can be read-only. verify_model checks the same checksums
# and the weights digest and returns False instead of raising.
#
# Cache layout
# ------------
#   $UNBIHEXIUM_CACHE/models/<model_id>/
#       model.pt        checkpoint (see zoo.checkpoint)
#       model.onnx      ONNX export, when requested
#       config.json     BuildConfig and entry metadata
#       model.sha256    sha256sum-compatible checksums of the files
#
# UNBIHEXIUM_CACHE defaults to ~/.cache/unbihexium.
#
# Usage
# -----
#   from unbihexium.zoo import load_model, ensure_model
#   model = load_model("ship_detector_base")          # in memory
#   directory = ensure_model("ship_detector_base", onnx=True)  # on disk
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON configuration files.
import json

# Environment variables.
import os

# Remove cache directories.
import shutil

# Represent file paths.
from pathlib import Path

# Type-only imports that would otherwise require PyTorch.
from typing import TYPE_CHECKING, Any

# Registry lookups.
from unbihexium.zoo.registry import ModelZooEntry, get_model

# File checksums.
from unbihexium.zoo.verify import (
    VerificationError,  # Raised when a checksum does not match.
    compute_sha256,  # SHA-256 of a file.
    read_sha256_file,  # Read a sha256sum file.
    verify_file,  # Verify one file.
    write_sha256_file,  # Write a sha256sum file.
)  # End of the checksum imports.

# Only needed by type checkers; importing it at runtime would require PyTorch.
if TYPE_CHECKING:
    # Model type returned by the loaders.
    from unbihexium.ai.models.factory import ZooModel

# Environment variable selecting the cache root.
CACHE_ENV = "UNBIHEXIUM_CACHE"

# File names inside a model directory.
CHECKPOINT_NAME = "model.pt"
# ONNX export file name.
ONNX_NAME = "model.onnx"
# Configuration file name.
CONFIG_NAME = "config.json"
# Checksum file name.
CHECKSUM_NAME = "model.sha256"

# Maximum size of a downloaded checkpoint, as a guard against wrong URLs.
MAX_DOWNLOAD_BYTES = 4 * 1024**3


# Root directory of the model cache.
def get_cache_dir() -> Path:
    # Environment override or the XDG-style default.
    root = os.environ.get(CACHE_ENV) or str(Path.home() / ".cache" / "unbihexium")
    # Expand "~" and return the models subdirectory.
    return Path(root).expanduser() / "models"


# Reject names that are not a single directory name inside the store.
def _check_name(model_id: str) -> str:
    # Separators, parent references and empty names could leave the store.
    if not model_id or model_id in {".", ".."} or "/" in model_id or "\\" in model_id:
        # Report the invalid name.
        raise ValueError(f"invalid model id {model_id!r}")
    # Return the name unchanged.
    return model_id


# Directory of one model inside the cache.
def model_dir(model_id: str, cache_dir: str | Path | None = None) -> Path:
    # Model ids must name one directory inside the store.
    _check_name(model_id)
    # Use the given cache root or the default one.
    root = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    # One subdirectory per model id.
    return root / model_id


# Return the registry entry of a model id or raise a helpful error.
def _entry(model_id: str) -> ModelZooEntry:
    # Look the model up.
    entry = get_model(model_id)
    # Unknown ids are errors.
    if entry is None:
        # Point the user to the list command.
        raise KeyError(f"unknown model {model_id!r}; see `unbihexium zoo list`")
    # Return the entry.
    return entry


# Check that a model's weights match its published or registered digest.
def _check_digest(model: ZooModel, entry: ModelZooEntry) -> None:
    # Downloaded and local checkpoints are compared even when customised.
    registered = entry.source in ("url", "local")
    # Unpublished digests and customised catalogue models cannot be compared.
    if not entry.weights_digest or (model.config.customised and not registered):
        # Nothing to verify.
        return
    # Compare the digest of the weights with the published value.
    actual = model.digest()
    # A mismatch means a platform or version problem, or a modified file.
    if actual != entry.weights_digest:
        # Report both digests.
        message = f"{entry.model_id}: weights digest {actual} != published {entry.weights_digest}"
        # Raise the error.
        raise VerificationError(message)


# Download a checkpoint from a URL into a file.
def _download(url: str, target: Path) -> None:
    # requests is a core dependency.
    import requests

    # Stream the response to avoid holding the file in memory.
    with requests.get(url, stream=True, timeout=60) as response:
        # Fail on HTTP errors.
        response.raise_for_status()
        # Number of bytes written.
        written = 0
        # Temporary file next to the target.
        tmp = target.with_suffix(".part")
        # Write the body in chunks.
        with tmp.open("wb") as fh:
            # Iterate over 1 MiB chunks.
            for chunk in response.iter_content(chunk_size=1 << 20):
                # Count the bytes.
                written += len(chunk)
                # Guard against unexpectedly large files.
                if written > MAX_DOWNLOAD_BYTES:
                    # Abort the download.
                    raise VerificationError(f"download of {url} exceeds {MAX_DOWNLOAD_BYTES} bytes")
                # Write the chunk.
                fh.write(chunk)
        # Move the completed file into place.
        tmp.replace(target)


# Load a model into memory.
def load_model(name: str | Path, variant: str | None = None, verify: bool = True) -> ZooModel:
    # PyTorch-dependent modules are imported lazily.
    from unbihexium.ai.models.factory import build_model  # Builds catalogue models.
    from unbihexium.zoo.checkpoint import load_checkpoint  # Reads checkpoint files.

    # A path to a checkpoint file is loaded directly.
    if Path(name).suffix == ".pt" and Path(name).is_file():
        # Load and verify the recorded digest.
        return load_checkpoint(name, verify=verify)
    # Otherwise the name is a model id or family.
    model_id = f"{name}_{variant}" if variant else str(name)
    # Registry entry.
    entry = _entry(model_id)
    # Registered checkpoints: a local file or a download in the cache.
    if entry.source in ("local", "url"):
        # Local files are read in place.
        if entry.source == "local" and entry.local_path:
            # Path of the registered file.
            path = Path(entry.local_path)
        # Downloads are fetched and verified into the cache first.
        else:
            # Path of the cached checkpoint.
            path = ensure_model(entry.model_id) / CHECKPOINT_NAME
        # Load the checkpoint.
        model = load_checkpoint(path, verify=verify)
        # Compare with the registered digest.
        if verify:
            # Raises VerificationError on a mismatch.
            _check_digest(model, entry)
        # Return the model.
        return model
    # Catalogue models are built in memory with their starter weights.
    model = build_model(entry.model_id)
    # Compare with the published digest.
    if verify:
        # Raises VerificationError on a mismatch.
        _check_digest(model, entry)
    # Return the model.
    return model


# Make sure a model exists in the cache and return its directory.
def ensure_model(
    model_id: str,  # Model id.
    cache_dir: str | Path | None = None,  # Cache root override.
    onnx: bool = False,  # Also export to ONNX.
    force: bool = False,  # Rebuild or download even if cached.
) -> Path:  # Directory of the cached model.
    # Registry entry.
    entry = _entry(model_id)
    # Target directory.
    directory = model_dir(entry.model_id, cache_dir)
    # Path of the checkpoint.
    checkpoint = directory / CHECKPOINT_NAME
    # Path of the ONNX export.
    onnx_path = directory / ONNX_NAME
    # Files of the cached entry that match model.sha256.
    verified = set() if force else _verified_files(directory)
    # The cached entry is reused only when it describes the current entry.
    current = {CHECKPOINT_NAME, CONFIG_NAME} <= verified and _config_matches(directory, entry)
    # Create the directory.
    directory.mkdir(parents=True, exist_ok=True)
    # Obtain the checkpoint when missing, modified, outdated or forced.
    if not current:
        # Downloaded models come from their URL.
        if entry.source == "url" and entry.download_url:
            # Fetch the checkpoint.
            _download(entry.download_url, checkpoint)
        # Local models are copied from their path.
        elif entry.source == "local" and entry.local_path:
            # Copy the registered file into the cache.
            shutil.copyfile(entry.local_path, checkpoint)
        # Registered entries without a location cannot be obtained.
        elif entry.source in ("url", "local"):
            # Report the incomplete registration.
            raise ValueError(f"{entry.model_id}: source {entry.source!r} without a location")
        # Catalogue models are built.
        else:
            # PyTorch-dependent module imported lazily.
            from unbihexium.zoo.checkpoint import save_checkpoint

            # Build the starter model and check its digest.
            model = load_model(entry.model_id)
            # Write the checkpoint.
            save_checkpoint(model, checkpoint)
        # Registered checkpoints must match the registered digest.
        if entry.source in ("url", "local"):
            # Remove a checkpoint that fails, so that it is not reused.
            try:
                # PyTorch-dependent module imported lazily.
                from unbihexium.zoo.checkpoint import load_checkpoint

                # Load with the recorded digest and compare with the entry.
                _check_digest(load_checkpoint(checkpoint), entry)
            # Any failure leaves no checkpoint behind.
            except Exception:
                # Delete the rejected file.
                checkpoint.unlink(missing_ok=True)
                # Report the original error.
                raise
    # An ONNX file is used only when model.sha256 vouches for it.
    onnx_ok = current and ONNX_NAME in verified
    # Unverified or outdated ONNX files are removed.
    if not onnx_ok and onnx_path.is_file():
        # Delete the file; it is exported again when requested.
        onnx_path.unlink()
    # Export to ONNX when requested and not verified.
    if onnx and not onnx_ok:
        # PyTorch-dependent modules imported lazily.
        from unbihexium.zoo.checkpoint import load_checkpoint  # Reads checkpoint files.
        from unbihexium.zoo.export import export_onnx  # Writes ONNX files.

        # Export the cached checkpoint.
        export_onnx(load_checkpoint(checkpoint), onnx_path)
    # Write the configuration file when it changed.
    _write_config(directory, entry)
    # Write the checksums of the model files when they changed.
    write_sha256_file(directory, [CHECKPOINT_NAME, ONNX_NAME, CONFIG_NAME])
    # Return the directory.
    return directory


# Names of the files of a model directory that match its model.sha256.
def _verified_files(directory: Path) -> set[str]:
    # Path of the checksum file.
    checksums = directory / CHECKSUM_NAME
    # Without a checksum file nothing is verified.
    if not checksums.is_file():
        # Empty set.
        return set()
    # Expected digests; an unreadable file verifies nothing.
    try:
        # Parse the checksum file.
        expected = read_sha256_file(checksums)
    # Undecodable or unreadable files.
    except (OSError, UnicodeDecodeError):
        # Empty set.
        return set()
    # Only files of the store layout count, and all of them must match.
    if not expected or not set(expected) <= {CHECKPOINT_NAME, ONNX_NAME, CONFIG_NAME}:
        # Unknown names mean a foreign or damaged file.
        return set()
    # Every listed file must exist and match.
    if not all(verify_file(directory / n, d) for n, d in expected.items()):
        # One mismatch invalidates the directory.
        return set()
    # Names of the verified files.
    return set(expected)


# Text of config.json for an entry and the checkpoint in a directory.
def _config_text(directory: Path, entry: ModelZooEntry) -> str:
    # Entry metadata.
    data: dict[str, Any] = entry.to_dict()
    # Checksum of the checkpoint file for quick checks.
    data["checkpoint_sha256"] = compute_sha256(directory / CHECKPOINT_NAME)
    # Pretty-printed JSON with a final newline.
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


# Whether config.json of a directory describes the entry and its checkpoint.
def _config_matches(directory: Path, entry: ModelZooEntry) -> bool:
    # Compare the stored text with the expected one.
    try:
        # Exact comparison of the file contents.
        return (directory / CONFIG_NAME).read_text(encoding="utf-8") == _config_text(
            directory,  # Model directory.
            entry,  # Registry entry.
        )  # End of the comparison.
    # Missing or unreadable files do not match.
    except (OSError, UnicodeDecodeError):
        # Not current.
        return False


# Write config.json with the entry metadata and the checkpoint checksum.
def _write_config(directory: Path, entry: ModelZooEntry) -> None:
    # Unchanged files are left alone, so a verified store can be read-only.
    if _config_matches(directory, entry):
        # Nothing to write.
        return
    # Configuration text.
    text = _config_text(directory, entry)
    # Write the file with LF line endings on every platform.
    (directory / CONFIG_NAME).write_text(text, encoding="utf-8", newline="\n")


# Backwards-compatible name: obtain a model and return its checkpoint path.
def download_model(model_id: str, cache_dir: str | Path | None = None, force: bool = False) -> Path:
    # Ensure the model is cached and return the checkpoint.
    return ensure_model(model_id, cache_dir=cache_dir, force=force) / CHECKPOINT_NAME


# Whether a model's checkpoint exists in the cache.
def is_model_cached(model_id: str, cache_dir: str | Path | None = None) -> bool:
    # Check for the checkpoint file.
    return (model_dir(model_id, cache_dir) / CHECKPOINT_NAME).is_file()


# Return the checkpoint path of a cached model, or None.
def get_cached_model_path(model_id: str, cache_dir: str | Path | None = None) -> Path | None:
    # Path of the checkpoint.
    path = model_dir(model_id, cache_dir) / CHECKPOINT_NAME
    # Return it only when it exists.
    return path if path.is_file() else None


# List the model ids present in the cache.
def list_cached(cache_dir: str | Path | None = None) -> list[str]:
    # Cache root.
    root = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    # An absent root means an empty cache.
    if not root.is_dir():
        # Nothing cached.
        return []
    # Directories that contain a checkpoint.
    return sorted(p.name for p in root.iterdir() if (p / CHECKPOINT_NAME).is_file())


# Remove one model or the whole cache; returns the number of removed models.
def clear_cache(model_id: str | None = None, cache_dir: str | Path | None = None) -> int:
    # Cache root.
    root = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    # A single model must be a known model id or an entry of the store.
    if model_id is not None:
        # Rejects separators and parent references.
        _check_name(model_id)
        # Unknown names that are not store entries are errors.
        if get_model(model_id) is None and not (root / model_id).is_dir():
            # Report the unknown name.
            raise ValueError(f"{model_id!r} is neither a model id nor an entry of the store")
    # Models to remove.
    targets = [model_id] if model_id else list_cached(cache_dir)
    # Number removed.
    removed = 0
    # Resolved cache root, for the containment check.
    resolved_root = root.resolve()
    # Remove each model directory.
    for mid in targets:
        # Directory of the model.
        directory = model_dir(mid, cache_dir)
        # A symbolic link is removed without touching its target.
        if directory.is_symlink():
            # Delete the link only.
            directory.unlink()
            # Count the removal.
            removed += 1
        # Only directories directly inside the store are deleted.
        elif directory.is_dir() and directory.resolve().parent == resolved_root:
            # Delete the directory tree.
            shutil.rmtree(directory)
            # Count the removal.
            removed += 1
    # Return the count.
    return removed


# Verify a cached model: file checksums and the published weights digest.
def verify_model(model_id: str, cache_dir: str | Path | None = None) -> bool:
    # Every failure, including unreadable or corrupt files, means "not verified".
    try:
        # PyTorch-dependent module imported lazily.
        from unbihexium.zoo.checkpoint import load_checkpoint

        # Directory of the model; invalid ids raise ValueError.
        directory = model_dir(model_id, cache_dir)
        # Files that match model.sha256; empty when any listed file does not.
        verified = _verified_files(directory)
        # The checkpoint and the configuration must be listed and match.
        if not {CHECKPOINT_NAME, CONFIG_NAME} <= verified:
            # Missing, unlisted or modified files.
            return False
        # Load the checkpoint, which also checks its recorded digest.
        model = load_checkpoint(directory / CHECKPOINT_NAME, verify=True)
        # Compare with the published digest.
        _check_digest(model, _entry(model_id))
    # Corrupt files raise many kinds of errors, for example UnpicklingError.
    except Exception:
        # Report failure.
        return False
    # All checks passed.
    return True


# =============================================================================
# End of module src/unbihexium/zoo/store.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
