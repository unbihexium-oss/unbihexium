"""Capability registry for tracking library features."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityDomain(str, Enum):
    """Domains for capabilities."""

    AI = "ai"
    TOURISM = "tourism"
    ANALYSIS = "analysis"
    INDICES = "indices"
    WATER = "water"
    ENVIRONMENT = "environment"
    FORESTRY = "forestry"
    IMAGING = "imaging"
    ASSETS = "assets"
    ENERGY = "energy"
    URBAN = "urban"
    AGRICULTURE = "agriculture"
    RISK = "risk"
    DEFENSE = "defense"
    SAR = "sar"


class CapabilityMaturity(str, Enum):
    """Maturity level of a capability."""

    STABLE = "stable"
    BETA = "beta"
    RESEARCH = "research"
    DEPRECATED = "deprecated"


@dataclass
class Capability:
    """A registered capability of the library."""

    capability_id: str
    name: str
    domain: CapabilityDomain
    description: str = ""
    maturity: CapabilityMaturity = CapabilityMaturity.STABLE
    entry_points: list[str] = field(default_factory=list)
    pipeline_id: str | None = None
    cli_command: str | None = None
    example_path: str | None = None
    test_path: str | None = None
    docs_path: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "domain": self.domain.value,
            "description": self.description,
            "maturity": self.maturity.value,
            "entry_points": self.entry_points,
            "pipeline_id": self.pipeline_id,
            "cli_command": self.cli_command,
        }


class CapabilityRegistry:
    """Registry for library capabilities."""

    _capabilities: dict[str, Capability] = {}

    @classmethod
    def register(cls, capability: Capability) -> None:
        """Register a capability."""
        cls._capabilities[capability.capability_id] = capability

    @classmethod
    def get(cls, capability_id: str) -> Capability | None:
        """Get a capability by ID."""
        return cls._capabilities.get(capability_id)

    @classmethod
    def list_all(cls) -> list[Capability]:
        """List all registered capabilities."""
        return list(cls._capabilities.values())

    @classmethod
    def by_domain(cls, domain: CapabilityDomain) -> list[Capability]:
        """List capabilities by domain."""
        return [c for c in cls._capabilities.values() if c.domain == domain]

    @classmethod
    def by_maturity(cls, maturity: CapabilityMaturity) -> list[Capability]:
        """List capabilities by maturity level."""
        return [c for c in cls._capabilities.values() if c.maturity == maturity]

    @classmethod
    def ids(cls) -> list[str]:
        """List all capability IDs."""
        return list(cls._capabilities.keys())
