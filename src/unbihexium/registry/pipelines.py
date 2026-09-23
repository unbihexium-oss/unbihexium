# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/registry/pipelines.py
# Title       : Registry of processing pipelines
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# PipelineRegistry maps pipeline ids to factories that build a pipeline
# object. Factories are registered with a decorator:
#
#   @PipelineRegistry.register("ship_detection", "Ship Detection", "...",
#                              ["ai", "maritime"])
#   def create(**params): ...
#
# The task APIs of unbihexium.ai register their pipelines on import (see
# unbihexium.ai.base.register_task_pipeline), and the command line runs
# them with `unbihexium pipeline run <id>`. The registry does not import
# the pipeline classes, so it has no dependency on unbihexium.core.
#
# Lookups: get (None when unknown), require (KeyError listing the known
# ids), create, list_all, ids, by_domain, search and unregister.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Callables.
from collections.abc import Callable

# Records.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any

# Factory that builds a pipeline from keyword parameters.
Factory = Callable[..., Any]


# Registry entry of one pipeline.
@dataclass
class PipelineEntry:
    # Unique id, for example "ship_detection".
    pipeline_id: str
    # Human-readable name.
    name: str
    # What the pipeline does.
    description: str = ""
    # Configuration class of the pipeline, if it has one.
    config_class: type | None = None
    # Function that builds the pipeline.
    factory: Factory | None = None
    # Capability domains of the pipeline.
    domains: list[str] = field(default_factory=list)
    # Free-form metadata.
    tags: dict[str, str] = field(default_factory=dict)

    # JSON-serialisable description.
    def to_dict(self) -> dict[str, Any]:
        # Public fields; the factory is not serialisable.
        return {
            "pipeline_id": self.pipeline_id,  # Id.
            "name": self.name,  # Name.
            "description": self.description,  # Description.
            "domains": list(self.domains),  # Domains.
            "tags": dict(self.tags),  # Metadata.
        }  # End of the dictionary.


# Class-level registry of pipeline factories.
class PipelineRegistry:
    # Entries by id, shared by the whole process.
    _pipelines: dict[str, PipelineEntry] = {}

    # Decorator that registers a pipeline factory under an id.
    @classmethod
    def register(
        cls,  # The registry.
        pipeline_id: str,  # Unique id.
        name: str,  # Human-readable name.
        description: str = "",  # What the pipeline does.
        domains: list[str] | None = None,  # Capability domains.
        tags: dict[str, str] | None = None,  # Free-form metadata.
    ) -> Callable[[Factory], Factory]:  # Decorator returning the factory unchanged.
        # Ids must be non-empty identifiers.
        if not pipeline_id or not pipeline_id.replace("_", "").replace("-", "").isalnum():
            # Explain the accepted form.
            raise ValueError(f"invalid pipeline id {pipeline_id!r}")

        # Record the factory; a later registration replaces an earlier one.
        def decorator(factory: Factory) -> Factory:
            # Registry entry.
            cls._pipelines[pipeline_id] = PipelineEntry(
                pipeline_id=pipeline_id,  # Id.
                name=name,  # Name.
                description=description,  # Description.
                factory=factory,  # Factory.
                domains=list(domains or []),  # Domains.
                tags=dict(tags or {}),  # Metadata.
            )  # End of the entry.
            # Return the factory so that it stays usable.
            return factory

        # Return the decorator.
        return decorator

    # Entry of an id, None when unknown.
    @classmethod
    def get(cls, pipeline_id: str) -> PipelineEntry | None:
        # Dictionary lookup.
        return cls._pipelines.get(pipeline_id)

    # Entry of an id; unknown ids raise a KeyError listing the known ones.
    @classmethod
    def require(cls, pipeline_id: str) -> PipelineEntry:
        # Look the id up.
        entry = cls.get(pipeline_id)
        # Unknown id.
        if entry is None:
            # Explain the known ids.
            raise KeyError(f"unknown pipeline {pipeline_id!r}; known: {', '.join(cls.ids())}")
        # Return the entry.
        return entry

    # Build a pipeline; None when the id is unknown or has no factory.
    @classmethod
    def create(cls, pipeline_id: str, **kwargs: Any) -> Any:
        # Entry of the id.
        entry = cls.get(pipeline_id)
        # Nothing to build.
        if entry is None or entry.factory is None:
            # No pipeline.
            return None
        # Call the factory with the parameters.
        return entry.factory(**kwargs)

    # Every entry, sorted by id.
    @classmethod
    def list_all(cls) -> list[PipelineEntry]:
        # Sorted for stable listings.
        return [cls._pipelines[k] for k in sorted(cls._pipelines)]

    # Alias of list_all kept for earlier releases.
    @classmethod
    def list_pipelines(cls) -> list[PipelineEntry]:
        # Same as list_all.
        return cls.list_all()

    # Every id, sorted.
    @classmethod
    def ids(cls) -> list[str]:
        # Sorted ids.
        return sorted(cls._pipelines)

    # Entries of a domain.
    @classmethod
    def by_domain(cls, domain: str) -> list[PipelineEntry]:
        # Accept enumeration members as well as strings.
        name = str(getattr(domain, "value", domain))
        # Entries that list the domain.
        return [p for p in cls.list_all() if name in p.domains]

    # Entries whose id, name or description contains the text (case-insensitive).
    @classmethod
    def search(cls, text: str) -> list[PipelineEntry]:
        # Lower-case query.
        query = text.lower()
        # Searchable text of each entry.
        return [
            p  # Matching entry.
            for p in cls.list_all()  # Every entry.
            if query in f"{p.pipeline_id} {p.name} {p.description}".lower()  # Text match.
        ]  # End of the matches.

    # Remove an entry; returns whether it existed.
    @classmethod
    def unregister(cls, pipeline_id: str) -> bool:
        # Remove it if present.
        return cls._pipelines.pop(pipeline_id, None) is not None


# =============================================================================
# End of module src/unbihexium/registry/pipelines.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
