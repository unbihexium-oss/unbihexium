# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/evidence.py
# Title       : Evidence and provenance records
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Audit trail of processing runs. An Evidence record identifies one artefact
# (an input file, an output product, a model or a configuration) by its
# SHA-256 digest, so that anyone holding the file can verify that it is the
# artefact that was used. A ProvenanceRecord ties the inputs, outputs and
# models of one pipeline run to its configuration and software environment:
#
#   sha256_file, sha256_bytes    digests of files and byte strings
#   sha256_array                 digest of an array, its dtype and its shape
#   canonical_json               deterministic JSON used for digests
#   capture_environment          interpreter, platform and library versions
#   Evidence                     one artefact and its digest
#   ProvenanceRecord             one run: inputs, outputs, models, config
#
# The record digest is the SHA-256 of the canonical JSON of the record
# without its digest field, so any change of the stored record is detected.
# The model follows the entity-activity-agent structure of W3C PROV: the
# artefacts are entities, the run is the activity, and parent records link
# runs into a derivation chain.
#
# References
# ----------
#   National Institute of Standards and Technology (2015). Secure Hash
#     Standard (SHS). FIPS PUB 180-4.
#   Moreau, L., Missier, P. (eds.) (2013). PROV-DM: the PROV data model.
#     W3C Recommendation, 30 April 2013.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# SHA-256 digests.
import hashlib

# JSON serialisation.
import json

# Interpreter and operating system information.
import platform

# Interpreter version.
import sys

# Unique record identifiers.
import uuid

# Record containers.
from dataclasses import dataclass, field

# Timestamps.
from datetime import datetime, timezone

# Kinds of evidence.
from enum import Enum

# File paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Size of the blocks in which files are hashed.
CHUNK_SIZE = 1 << 20


# Kinds of artefacts recorded as evidence.
class EvidenceType(str, Enum):
    # Data read by a run.
    INPUT = "input"
    # Data written by a run.
    OUTPUT = "output"
    # Data produced and consumed inside a run.
    INTERMEDIATE = "intermediate"
    # Model weights.
    MODEL = "model"
    # Configuration files.
    CONFIG = "config"
    # Log files.
    LOG = "log"


# Current time in UTC as an ISO 8601 string.
def utc_now() -> str:
    # Timezone-aware timestamp with microseconds.
    return datetime.now(timezone.utc).isoformat()


# SHA-256 digest of a byte string as 64 hexadecimal characters.
def sha256_bytes(data: bytes) -> str:
    # Hash in one call.
    return hashlib.sha256(data).hexdigest()


# SHA-256 digest of a file, read in blocks so that large files fit in memory.
def sha256_file(path: str | Path) -> str:
    # Incremental hash.
    digest = hashlib.sha256()
    # Read the file in binary mode.
    with Path(path).open("rb") as handle:
        # Feed every block to the hash.
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            # Update the digest.
            digest.update(block)
    # Hexadecimal digest.
    return digest.hexdigest()


# SHA-256 digest of an array that also covers its dtype and shape.
def sha256_array(array: Any) -> str:
    # C-contiguous copy so that the byte order of the elements is defined.
    values = np.ascontiguousarray(array)
    # Incremental hash.
    digest = hashlib.sha256()
    # The dtype distinguishes arrays with equal bytes but different types.
    digest.update(values.dtype.str.encode())
    # The shape distinguishes arrays with equal bytes but different layouts.
    digest.update(repr(values.shape).encode())
    # The element bytes.
    digest.update(values.tobytes())
    # Hexadecimal digest.
    return digest.hexdigest()


# Deterministic JSON: sorted keys, no whitespace, non-JSON values as strings.
def canonical_json(value: Any) -> str:
    # The same value always gives the same text.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# Versions of the interpreter, the platform and the main libraries.
def capture_environment() -> dict[str, str]:
    # Imported lazily to avoid an import cycle with the package root.
    from unbihexium._version import __version__

    # Environment description.
    environment = {
        "python": sys.version.split()[0],  # Interpreter version.
        "implementation": platform.python_implementation(),  # CPython or other.
        "platform": platform.platform(),  # Operating system and architecture.
        "numpy": np.__version__,  # NumPy version.
        "unbihexium": __version__,  # Library version.
    }  # End of the environment.
    # Optional libraries that influence numerical results.
    for name in ("scipy", "rasterio", "torch", "onnxruntime"):
        # Only libraries that are already imported are reported.
        module = sys.modules.get(name)
        # Record the version when the module is loaded.
        if module is not None:
            # Version attribute of the module.
            environment[name] = str(getattr(module, "__version__", "unknown"))
    # Return the description.
    return environment


# One artefact identified by its SHA-256 digest.
@dataclass
class Evidence:
    # File path, URL or other description of where the artefact lives.
    source: str
    # SHA-256 digest as 64 hexadecimal characters; empty when unknown.
    checksum: str = ""
    # Role of the artefact.
    evidence_type: EvidenceType = EvidenceType.INPUT
    # Free-text description.
    description: str = ""
    # Creation time of the record in ISO 8601.
    timestamp: str = field(default_factory=utc_now)
    # Size of the artefact in bytes, when known.
    size_bytes: int | None = None
    # Additional attributes such as the file name.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Identifier; derived from the type and digest when left empty.
    evidence_id: str = ""

    # Validate the digest and derive the identifier.
    def __post_init__(self) -> None:
        # Accept the type as a plain string.
        self.evidence_type = EvidenceType(self.evidence_type)
        # A digest, when given, must be 64 lowercase hexadecimal characters.
        if self.checksum:
            # Normalise the case.
            self.checksum = self.checksum.lower()
            # Check the length and the alphabet.
            if len(self.checksum) != 64 or any(c not in "0123456789abcdef" for c in self.checksum):
                # Explain the problem.
                raise ValueError(f"checksum must be a SHA-256 hex digest, got {self.checksum!r}")
        # Derive the identifier when none was given.
        if not self.evidence_id:
            # The digest identifies the content; the source is the fallback.
            key = self.checksum or sha256_bytes(self.source.encode())
            # Type prefix and the first 16 hexadecimal characters.
            self.evidence_id = f"{self.evidence_type.value}-{key[:16]}"

    # Digest of a file (kept as a method for callers of earlier releases).
    @staticmethod
    def compute_checksum(path: str | Path) -> str:
        # SHA-256 of the file content.
        return sha256_file(path)

    # Evidence of a file with its digest and size.
    @classmethod
    def from_file(
        cls,  # Evidence class.
        path: str | Path,  # File to record.
        evidence_type: EvidenceType | str = EvidenceType.INPUT,  # Role of the file.
        description: str = "",  # Free-text description.
    ) -> Evidence:  # The record.
        # Path object.
        path = Path(path)
        # The file must exist.
        if not path.is_file():
            # Explain the problem.
            raise FileNotFoundError(f"no such file: {path}")
        # Build the record.
        return cls(
            source=str(path),  # Location.
            checksum=sha256_file(path),  # Content digest.
            evidence_type=EvidenceType(evidence_type),  # Role.
            description=description,  # Description.
            size_bytes=path.stat().st_size,  # Size.
            metadata={"filename": path.name},  # File name.
        )  # End of the record.

    # Evidence of an in-memory array.
    @classmethod
    def from_array(
        cls,  # Evidence class.
        array: Any,  # Array to record.
        source: str,  # Name of the array.
        evidence_type: EvidenceType | str = EvidenceType.INTERMEDIATE,  # Role.
        description: str = "",  # Free-text description.
    ) -> Evidence:  # The record.
        # Array view.
        values = np.asarray(array)
        # Build the record.
        return cls(
            source=source,  # Name.
            checksum=sha256_array(values),  # Digest of dtype, shape and content.
            evidence_type=EvidenceType(evidence_type),  # Role.
            description=description,  # Description.
            size_bytes=int(values.nbytes),  # Size of the elements.
            metadata={"dtype": values.dtype.str, "shape": list(values.shape)},  # Layout.
        )  # End of the record.

    # Whether a file still has the recorded digest.
    def verify(self, path: str | Path | None = None) -> bool:
        # Records without a digest cannot be verified.
        if not self.checksum:
            # Explain the problem.
            raise ValueError(f"evidence {self.evidence_id} has no checksum")
        # File to check: the given path or the recorded source.
        target = Path(path) if path is not None else Path(self.source)
        # Missing files do not match.
        if not target.is_file():
            # Not verified.
            return False
        # Compare the digests.
        return sha256_file(target) == self.checksum

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "evidence_id": self.evidence_id,  # Identifier.
            "evidence_type": self.evidence_type.value,  # Role.
            "source": self.source,  # Location.
            "checksum": self.checksum,  # Digest.
            "description": self.description,  # Description.
            "timestamp": self.timestamp,  # Creation time.
            "size_bytes": self.size_bytes,  # Size.
            "metadata": self.metadata,  # Additional attributes.
        }  # End of the dictionary.

    # Record from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        # Build the record.
        return cls(
            source=data["source"],  # Location.
            checksum=data.get("checksum", ""),  # Digest.
            evidence_type=EvidenceType(data.get("evidence_type", "input")),  # Role.
            description=data.get("description", ""),  # Description.
            timestamp=data.get("timestamp", utc_now()),  # Creation time.
            size_bytes=data.get("size_bytes"),  # Size.
            metadata=dict(data.get("metadata", {})),  # Additional attributes.
            evidence_id=data.get("evidence_id", ""),  # Identifier.
        )  # End of the record.


# Serialise an artefact.
def _artefact_to_json(item: Evidence | str) -> Any:
    # Evidence records become dictionaries, strings stay strings.
    return item.to_dict() if isinstance(item, Evidence) else str(item)


# Deserialise an artefact.
def _artefact_from_json(item: Any) -> Evidence | str:
    # Dictionaries are evidence records, everything else is a string.
    return Evidence.from_dict(item) if isinstance(item, dict) else str(item)


# Provenance of one pipeline run.
@dataclass
class ProvenanceRecord:
    # Identifier of the run.
    run_id: str
    # Identifier of the pipeline.
    pipeline_id: str
    # Identifier of the record.
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Inputs as evidence records or paths.
    inputs: list[Evidence | str] = field(default_factory=list)
    # Outputs as evidence records or paths.
    outputs: list[Evidence | str] = field(default_factory=list)
    # Identifiers of the models used.
    model_ids: list[str] = field(default_factory=list)
    # Evidence of model weight files.
    models: list[Evidence] = field(default_factory=list)
    # Configuration of the run.
    config: dict[str, Any] = field(default_factory=dict)
    # Software environment of the run.
    environment: dict[str, str] = field(default_factory=dict)
    # Records of the runs whose outputs this run consumed.
    parent_records: list[str] = field(default_factory=list)
    # Creation time in ISO 8601.
    created_at: str = field(default_factory=utc_now)

    # Add an input.
    def add_input(self, item: Evidence | str) -> None:
        # Append to the inputs.
        self.inputs.append(item)

    # Add an output.
    def add_output(self, item: Evidence | str) -> None:
        # Append to the outputs.
        self.outputs.append(item)

    # Add a model by identifier or by weight file evidence.
    def add_model(self, model: Evidence | str) -> None:
        # Evidence of a weight file.
        if isinstance(model, Evidence):
            # Keep the evidence and its identifier.
            self.models.append(model)
            # Identifier of the model.
            self.model_ids.append(model.evidence_id)
        # Plain model identifiers.
        else:
            # Keep the identifier.
            self.model_ids.append(model)

    # Record the current software environment.
    def capture_environment(self) -> None:
        # Replace the environment with the current one.
        self.environment = capture_environment()

    # Plain dictionary for JSON output, without the digest.
    def _body(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "record_id": self.record_id,  # Identifier of the record.
            "run_id": self.run_id,  # Identifier of the run.
            "pipeline_id": self.pipeline_id,  # Identifier of the pipeline.
            "created_at": self.created_at,  # Creation time.
            "inputs": [_artefact_to_json(i) for i in self.inputs],  # Inputs.
            "outputs": [_artefact_to_json(o) for o in self.outputs],  # Outputs.
            "model_ids": list(self.model_ids),  # Model identifiers.
            "models": [m.to_dict() for m in self.models],  # Weight evidence.
            "config": self.config,  # Configuration.
            "environment": self.environment,  # Software environment.
            "parent_records": list(self.parent_records),  # Parent runs.
        }  # End of the dictionary.

    # SHA-256 of the canonical JSON of the record.
    def digest(self) -> str:
        # Digest of the body.
        return sha256_bytes(canonical_json(self._body()).encode())

    # Plain dictionary for JSON output, with the digest.
    def to_dict(self) -> dict[str, Any]:
        # Body and digest.
        return {**self._body(), "digest": self.digest()}

    # JSON text of the record, optionally written to a file.
    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        # Serialise.
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        # Write the file when a path is given.
        if path is not None:
            # Output file.
            target = Path(path)
            # Create the parent directory.
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write the text with a final newline.
            target.write_text(text + "\n", encoding="utf-8", newline="\n")
        # Return the text.
        return text

    # Record from a dictionary written by to_dict.
    @classmethod
    def from_dict(cls, data: dict[str, Any], verify: bool = True) -> ProvenanceRecord:
        # Build the record.
        record = cls(
            run_id=data["run_id"],  # Identifier of the run.
            pipeline_id=data["pipeline_id"],  # Identifier of the pipeline.
            record_id=data.get("record_id", str(uuid.uuid4())),  # Identifier.
            inputs=[_artefact_from_json(i) for i in data.get("inputs", [])],  # Inputs.
            outputs=[_artefact_from_json(o) for o in data.get("outputs", [])],  # Outputs.
            model_ids=list(data.get("model_ids", [])),  # Model identifiers.
            models=[Evidence.from_dict(m) for m in data.get("models", [])],  # Weights.
            config=dict(data.get("config", {})),  # Configuration.
            environment=dict(data.get("environment", {})),  # Environment.
            parent_records=list(data.get("parent_records", [])),  # Parent runs.
            created_at=data.get("created_at", utc_now()),  # Creation time.
        )  # End of the record.
        # Stored digest, when present.
        stored = data.get("digest")
        # A stored digest must match the content.
        if verify and stored and stored != record.digest():
            # The record was modified after it was written.
            raise ValueError(f"provenance record {record.record_id} fails its digest check")
        # Return the record.
        return record

    # Record from JSON text or from a JSON file.
    @classmethod
    def from_json(cls, source: str | Path, verify: bool = True) -> ProvenanceRecord:
        # Paths are read; strings that start with a brace are JSON text.
        text = source if isinstance(source, str) and source.lstrip().startswith("{") else None
        # Read the file when a path was given.
        if text is None:
            # File content.
            text = Path(source).read_text(encoding="utf-8")
        # Parse and build.
        return cls.from_dict(json.loads(text), verify=verify)

    # Outputs whose files no longer match their digests.
    def verify_outputs(self) -> list[str]:
        # Sources of failing outputs.
        failed = []
        # Check every output with a digest.
        for item in self.outputs:
            # Only evidence records with digests can be checked.
            if isinstance(item, Evidence) and item.checksum and not item.verify():
                # Record the failure.
                failed.append(item.source)
        # Return the failures.
        return failed


# =============================================================================
# End of module src/unbihexium/core/evidence.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
