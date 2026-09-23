# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/registry/__init__.py
# Title       : Registries of capabilities, models and pipelines
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML
# =============================================================================
#
# Abstract
# --------
#   CapabilityRegistry   what the library can do (library algorithms and one
#                        capability per model zoo family)
#   ModelRegistry        flat view of the model zoo with input validation
#   PipelineRegistry     pipeline factories used by the command line
#
# get_capability, list_capabilities and register_capability are shortcuts
# for the capability registry.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Capability registry.
from unbihexium.registry.capabilities import (
    Capability,  # One capability.
    CapabilityDomain,  # Domains.
    CapabilityMaturity,  # Maturity levels.
    CapabilityRegistry,  # Registry.
)  # End of the capability imports.

# Model registry.
from unbihexium.registry.models import (
    ModelEntry,  # Flat model description.
    ModelRegistry,  # Registry.
)  # End of the model imports.

# Pipeline registry.
from unbihexium.registry.pipelines import (
    PipelineEntry,  # One pipeline.
    PipelineRegistry,  # Registry.
)  # End of the pipeline imports.


# Capability of an id, None when unknown.
def get_capability(capability_id: str) -> Capability | None:
    # Registry lookup.
    return CapabilityRegistry.get(capability_id)


# Every capability, optionally of one domain.
def list_capabilities(domain: CapabilityDomain | str | None = None) -> list[Capability]:
    # Filtered or complete listing.
    return CapabilityRegistry.by_domain(domain) if domain else CapabilityRegistry.list_all()


# Register a capability.
def register_capability(capability: Capability, replace: bool = False) -> Capability:
    # Registry insertion.
    return CapabilityRegistry.register(capability, replace=replace)


# Public names of the package.
__all__ = [
    "Capability",  # One capability.
    "CapabilityDomain",  # Domains.
    "CapabilityMaturity",  # Maturity levels.
    "CapabilityRegistry",  # Capability registry.
    "ModelEntry",  # Flat model description.
    "ModelRegistry",  # Model registry.
    "PipelineEntry",  # One pipeline.
    "PipelineRegistry",  # Pipeline registry.
    "get_capability",  # Capability lookup.
    "list_capabilities",  # Capability listing.
    "register_capability",  # Capability registration.
]  # End of the export list.


# =============================================================================
# End of module src/unbihexium/registry/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
