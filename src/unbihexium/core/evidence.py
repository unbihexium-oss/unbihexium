"""Evidence and provenance records for audit trails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EvidenceType(str, Enum):
    """Types of evidence records."""

    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"
    MODEL = "model"
    CONFIG = "config"
    LOG = "log"


@dataclass
class Evidence:
    """Evidence record for data provenance."""

    evidence_id: str
    evidence_type: EvidenceType
    source: str
    checksum: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path, evidence_type: EvidenceType) -> Evidence:
        """Create evidence from a file."""
        checksum = cls._compute_checksum(path)
        return cls(
            evidence_id=f"{evidence_type.value}_{path.stem}_{checksum[:8]}",
            evidence_type=evidence_type,
            source=str(path),
            checksum=checksum,
            metadata={"filename": path.name, "size_bytes": path.stat().st_size},
        )

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "source": self.source,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a pipeline execution."""

    record_id: str
    pipeline_id: str
    run_id: str
    created_at: datetime = field(default_factory=datetime.now)
    inputs: list[Evidence] = field(default_factory=list)
    outputs: list[Evidence] = field(default_factory=list)
    models: list[Evidence] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    parent_records: list[str] = field(default_factory=list)

    def add_input(self, evidence: Evidence) -> None:
        self.inputs.append(evidence)

    def add_output(self, evidence: Evidence) -> None:
        self.outputs.append(evidence)

    def add_model(self, evidence: Evidence) -> None:
        self.models.append(evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "inputs": [e.to_dict() for e in self.inputs],
            "outputs": [e.to_dict() for e in self.outputs],
            "models": [e.to_dict() for e in self.models],
            "config": self.config,
            "environment": self.environment,
            "parent_records": self.parent_records,
        }

    def to_json(self, path: Path) -> None:
        """Save provenance record to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> ProvenanceRecord:
        """Load provenance record from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            record_id=data["record_id"],
            pipeline_id=data["pipeline_id"],
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config=data.get("config", {}),
            environment=data.get("environment", {}),
            parent_records=data.get("parent_records", []),
        )
