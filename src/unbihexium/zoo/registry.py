# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/registry.py
# Title       : Registry of model zoo entries
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library and PyYAML
# =============================================================================
#
# Abstract
# --------
# A ModelZooEntry describes one model (a family in one size variant) of the
# model zoo: its specification from the catalogue, the hyperparameters of
# its variant, the expected digest of its starter weights and where its
# files can be obtained. The registry contains the 520 catalogue models and
# any models registered by the user, for example fine-tuned checkpoints
# shared inside a project.
#
# Expected digests are read from digests.json, which
# `python -m unbihexium.zoo.sync` generates next to catalog.yaml. An entry
# without a digest can still be built, but it cannot be verified against a
# published value.
#
# Usage
# -----
#   from unbihexium.zoo import get_model, list_models
#   entry = get_model("ship_detector_base")
#   detectors = list_models(task="detection", variant="tiny")
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON parsing of the digest table.
import json

# Mutable records.
from dataclasses import dataclass, field

# Cache the digest table.
from functools import lru_cache

# Locate packaged data files.
from importlib import resources

# Type of loosely structured values.
from typing import Any

# Catalogue types and lookups.
from unbihexium.zoo.catalog import (
    MODEL_LICENSE,  # Licence of the models.
    CatalogError,  # Raised for unknown families.
    ModelSpec,  # Catalogue entry of a family.
    Task,  # Task enumeration.
    Variant,  # Size variant enumeration.
    VariantSpec,  # Variant hyperparameters.
    all_model_ids,  # Every model id of the zoo.
    get_spec,  # Look up a family.
    get_variant,  # Look up a variant.
    parse_model_id,  # Split a model id.
)  # End of the catalogue imports.

# File with the expected weight digests of the catalogue models.
DIGESTS_FILE = "digests.json"


# Description of one model of the zoo.
@dataclass
class ModelZooEntry:
    # Model id, for example "ship_detector_base".
    model_id: str
    # Catalogue specification of the family.
    spec: ModelSpec
    # Hyperparameters of the variant.
    variant: VariantSpec
    # Expected SHA-256 digest of the starter weights, if published.
    weights_digest: str = ""
    # Number of trainable parameters, if known.
    num_parameters: int = 0
    # Where the files come from: "build" (generated locally), "url" or "local".
    source: str = "build"
    # Download URL of a checkpoint, for source "url".
    download_url: str | None = None
    # Path of a local checkpoint, for source "local".
    local_path: str | None = None
    # Version of the entry.
    version: str = "2.0.0"
    # Licence of the model.
    license: str = MODEL_LICENSE
    # Free-form metadata.
    tags: dict[str, str] = field(default_factory=dict)

    # Model family.
    @property
    def family(self) -> str:
        # Read it from the specification.
        return self.spec.family

    # Task of the model.
    @property
    def task(self) -> Task:
        # Read it from the specification.
        return self.spec.task

    # Capability domain of the model.
    @property
    def domain(self) -> str:
        # Read it from the specification.
        return self.spec.domain

    # Human-readable name including the variant.
    @property
    def name(self) -> str:
        # Family name and variant.
        return f"{self.spec.name} ({self.variant.variant.value})"

    # Whether the model needs training before its predictions are meaningful.
    @property
    def requires_training(self) -> bool:
        # Everything except the spectral index formulas.
        return self.spec.task.is_trainable

    # Serialise to plain Python types.
    def to_dict(self) -> dict[str, Any]:
        # Specification plus entry fields.
        return {
            "model_id": self.model_id,  # Model id.
            "variant": self.variant.variant.value,  # Size variant.
            "tile_size": self.variant.tile_size,  # Recommended tile size.
            "weights_digest": self.weights_digest,  # Expected digest.
            "num_parameters": self.num_parameters,  # Parameter count.
            "source": self.source,  # Origin of the files.
            "download_url": self.download_url,  # Download URL.
            "version": self.version,  # Entry version.
            "requires_training": self.requires_training,  # Starter model flag.
            **self.spec.to_dict(),  # Catalogue fields.
        }  # End of the dictionary.


# Read the published digests; an absent file yields an empty table.
@lru_cache(maxsize=1)
def _published() -> dict[str, dict[str, Any]]:
    # The digest file is generated; it may be missing in a development tree.
    path = resources.files("unbihexium.zoo").joinpath(DIGESTS_FILE)
    # Return an empty table when the file does not exist.
    if not path.is_file():
        # No published digests.
        return {}
    # Parse the JSON table.
    return json.loads(path.read_text(encoding="utf-8")).get("models", {})


# Models registered at runtime by the user.
_custom: dict[str, ModelZooEntry] = {}


# Create the entry of a catalogue model.
def _catalogue_entry(model_id: str) -> ModelZooEntry:
    # Split the model id.
    family, variant = parse_model_id(model_id)
    # Published digest and parameter count, if any.
    published = _published().get(model_id, {})
    # Build the entry.
    return ModelZooEntry(
        model_id=model_id,  # Model id.
        spec=get_spec(family),  # Catalogue specification.
        variant=get_variant(variant),  # Variant hyperparameters.
        weights_digest=str(published.get("weights_digest", "")),  # Expected digest.
        num_parameters=int(published.get("num_parameters", 0)),  # Parameter count.
    )  # End of the entry.


# Register a user model, for example a fine-tuned checkpoint.
def register_model(entry: ModelZooEntry) -> None:
    # Store the entry under its model id, replacing an earlier one.
    _custom[entry.model_id] = entry


# Remove a user model from the registry.
def unregister_model(model_id: str) -> bool:
    # Remove and report whether the model was registered.
    return _custom.pop(model_id, None) is not None


# Return the entry of a model id, or None when it is unknown.
def get_model(model_id: str) -> ModelZooEntry | None:
    # User registrations take precedence over the catalogue.
    if model_id in _custom:
        # Return the user entry.
        return _custom[model_id]
    # A bare family name refers to its base variant.
    family, variant = parse_model_id(model_id)
    # Look the family up in the catalogue.
    try:
        # Build the catalogue entry of the family and variant.
        return _catalogue_entry(f"{family}_{variant.value}")
    # Unknown names yield None.
    except CatalogError:
        # The model does not exist.
        return None


# List entries, optionally filtered by task, domain and variant.
def list_models(
    task: Task | str | None = None,  # Keep only this task.
    domain: str | None = None,  # Keep only this domain.
    variant: Variant | str | None = None,  # Keep only this variant.
) -> list[ModelZooEntry]:  # Matching entries, catalogue first.
    # All catalogue entries followed by user entries.
    entries = [_catalogue_entry(mid) for mid in all_model_ids()] + list(_custom.values())
    # Filter by task.
    if task is not None:
        # Normalise the task.
        wanted_task = Task(task)
        # Keep matching entries.
        entries = [e for e in entries if e.task is wanted_task]
    # Filter by domain.
    if domain is not None:
        # Keep matching entries.
        entries = [e for e in entries if e.domain == domain]
    # Filter by variant.
    if variant is not None:
        # Normalise the variant.
        wanted_variant = Variant(variant)
        # Keep matching entries.
        entries = [e for e in entries if e.variant.variant is wanted_variant]
    # Return the filtered list.
    return entries


# =============================================================================
# End of module src/unbihexium/zoo/registry.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
