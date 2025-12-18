"""Pipeline registry for processing pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from unbihexium.core.pipeline import Pipeline, PipelineConfig


@dataclass
class PipelineEntry:
    """Entry in the pipeline registry."""

    pipeline_id: str
    name: str
    description: str = ""
    config_class: type[PipelineConfig] | None = None
    factory: Callable[..., Pipeline] | None = None
    domains: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
        }


class PipelineRegistry:
    """Registry for processing pipelines."""

    _pipelines: dict[str, PipelineEntry] = {}

    @classmethod
    def register(
        cls,
        pipeline_id: str,
        name: str,
        description: str = "",
        domains: list[str] | None = None,
    ) -> Callable[[Callable[..., Pipeline]], Callable[..., Pipeline]]:
        """Decorator to register a pipeline factory."""

        def decorator(factory: Callable[..., Pipeline]) -> Callable[..., Pipeline]:
            entry = PipelineEntry(
                pipeline_id=pipeline_id,
                name=name,
                description=description,
                factory=factory,
                domains=domains or [],
            )
            cls._pipelines[pipeline_id] = entry
            return factory

        return decorator

    @classmethod
    def get(cls, pipeline_id: str) -> PipelineEntry | None:
        """Get a pipeline entry by ID."""
        return cls._pipelines.get(pipeline_id)

    @classmethod
    def create(cls, pipeline_id: str, **kwargs: Any) -> Pipeline | None:
        """Create a pipeline instance."""
        entry = cls.get(pipeline_id)
        if entry is None or entry.factory is None:
            return None
        return entry.factory(**kwargs)

    @classmethod
    def list_all(cls) -> list[PipelineEntry]:
        """List all registered pipelines."""
        return list(cls._pipelines.values())

    @classmethod
    def ids(cls) -> list[str]:
        """List all pipeline IDs."""
        return list(cls._pipelines.keys())

    @classmethod
    def by_domain(cls, domain: str) -> list[PipelineEntry]:
        """List pipelines by domain."""
        return [p for p in cls._pipelines.values() if domain in p.domains]
