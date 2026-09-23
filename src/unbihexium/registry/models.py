# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/registry/models.py
# Title       : Registry view of the model zoo
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML; loading a model
#               needs PyTorch
# =============================================================================
#
# Abstract
# --------
# ModelRegistry exposes the models of the zoo (unbihexium.zoo, 520
# catalogue models plus user-registered ones) as flat ModelEntry records
# with the fields that services and user interfaces need: task, domain,
# variant, input channels, outputs, digest and licence. It adds
#
#   check_input   validation of an input shape against the model, with a
#                 message that names the expected bands
#   load          loading with an in-process cache of opened models
#
# The zoo remains the single source of truth; the registry keeps no copy of
# the catalogue. Entries registered with ModelRegistry.register are
# descriptions only (for example of external models) and are listed after
# the zoo models.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Records.
from dataclasses import dataclass, field

# Type of loosely structured values.
from typing import Any


# Flat description of one model.
@dataclass
class ModelEntry:
    # Model id, for example "ship_detector_base".
    model_id: str
    # Human-readable name.
    name: str = ""
    # Task, for example "detection".
    task: str = ""
    # Model family.
    family: str = ""
    # Capability domain.
    domain: str = ""
    # Size variant.
    variant: str = ""
    # Names of the input channels, in order.
    channels: list[str] = field(default_factory=list)
    # Class names, target names or output bands.
    outputs: list[str] = field(default_factory=list)
    # Units of regression outputs.
    units: list[str] = field(default_factory=list)
    # Expected SHA-256 digest of the weights.
    sha256: str = ""
    # Download URL of the weights, if any.
    download_url: str | None = None
    # Number of trainable parameters, if known.
    num_parameters: int = 0
    # Licence of the model.
    license: str = "MPL-2.0"
    # Origin of the files: build, url, local or external.
    source: str = "build"
    # Whether the model must be trained before its predictions are useful.
    requires_training: bool = True
    # Recommended tile size.
    tile_size: int = 256
    # Free-form metadata.
    tags: dict[str, str] = field(default_factory=dict)

    # Number of input channels.
    @property
    def in_channels(self) -> int:
        # One per channel name.
        return len(self.channels)

    # Record from a model zoo entry.
    @classmethod
    def from_zoo(cls, entry: Any) -> ModelEntry:
        # Copy the fields of the zoo entry.
        return cls(
            model_id=entry.model_id,  # Id.
            name=entry.name,  # Name.
            task=entry.task.value,  # Task.
            family=entry.family,  # Family.
            domain=entry.domain,  # Domain.
            variant=entry.variant.variant.value,  # Variant.
            channels=list(entry.spec.channel_names),  # Input channels.
            outputs=list(entry.spec.outputs),  # Outputs.
            units=list(entry.spec.units),  # Units.
            sha256=entry.weights_digest,  # Digest.
            download_url=entry.download_url,  # URL.
            num_parameters=entry.num_parameters,  # Parameters.
            license=entry.license,  # Licence.
            source=entry.source,  # Origin.
            requires_training=entry.requires_training,  # Starter model flag.
            tile_size=entry.variant.tile_size,  # Tile size.
            tags=dict(entry.tags),  # Metadata.
        )  # End of the record.

    # JSON-serialisable description.
    def to_dict(self) -> dict[str, Any]:
        # Every field plus the channel count.
        return {
            "model_id": self.model_id,  # Id.
            "name": self.name,  # Name.
            "task": self.task,  # Task.
            "family": self.family,  # Family.
            "domain": self.domain,  # Domain.
            "variant": self.variant,  # Variant.
            "in_channels": self.in_channels,  # Channel count.
            "channels": list(self.channels),  # Channel names.
            "outputs": list(self.outputs),  # Outputs.
            "units": list(self.units),  # Units.
            "sha256": self.sha256,  # Digest.
            "download_url": self.download_url,  # URL.
            "num_parameters": self.num_parameters,  # Parameters.
            "license": self.license,  # Licence.
            "source": self.source,  # Origin.
            "requires_training": self.requires_training,  # Starter model flag.
            "tile_size": self.tile_size,  # Tile size.
        }  # End of the dictionary.


# Registry of the models of the zoo and of user descriptions.
class ModelRegistry:
    # User-registered descriptions by id.
    _models: dict[str, ModelEntry] = {}
    # Opened models by (id, variant).
    _loaded: dict[tuple[str, str | None], Any] = {}

    # Register a description; duplicates of zoo ids are rejected.
    @classmethod
    def register(cls, entry: ModelEntry, replace: bool = False) -> ModelEntry:
        # Imported lazily to keep the registry import light.
        from unbihexium.zoo import get_model

        # Zoo ids cannot be shadowed.
        if get_model(entry.model_id) is not None and entry.model_id not in cls._models:
            # Explain the conflict.
            raise ValueError(f"{entry.model_id!r} is a model zoo id")
        # Existing user entries need replace.
        if entry.model_id in cls._models and not replace:
            # Explain the conflict.
            raise ValueError(f"model {entry.model_id!r} is already registered")
        # Store the description.
        cls._models[entry.model_id] = entry
        # Return it.
        return entry

    # Remove a user description; returns whether it existed.
    @classmethod
    def unregister(cls, model_id: str) -> bool:
        # Remove it if present.
        return cls._models.pop(model_id, None) is not None

    # Description of a model id, None when unknown.
    @classmethod
    def get(cls, model_id: str) -> ModelEntry | None:
        # User descriptions first.
        if model_id in cls._models:
            # Return it.
            return cls._models[model_id]
        # Imported lazily.
        from unbihexium.zoo import get_model

        # Zoo entry; family names resolve to the base variant.
        entry = get_model(model_id)
        # Convert it.
        return ModelEntry.from_zoo(entry) if entry is not None else None

    # Description of a model id; unknown ids raise KeyError.
    @classmethod
    def require(cls, model_id: str) -> ModelEntry:
        # Look the id up.
        entry = cls.get(model_id)
        # Unknown id.
        if entry is None:
            # Explain the problem.
            raise KeyError(f"unknown model {model_id!r}; see `unbihexium zoo list`")
        # Return it.
        return entry

    # Every description, filtered by task, domain and variant.
    @classmethod
    def list_all(
        cls,  # The registry.
        task: str | None = None,  # Keep only this task.
        domain: str | None = None,  # Keep only this domain.
        variant: str | None = None,  # Keep only this variant.
    ) -> list[ModelEntry]:  # Zoo models first, then user descriptions.
        # Imported lazily.
        from unbihexium.zoo import list_models

        # Zoo models, filtered by the zoo.
        entries = [ModelEntry.from_zoo(e) for e in list_models(task, domain, variant)]
        # Filter values as strings (enumeration members give their value).
        filters = {"task": task, "domain": domain, "variant": variant}
        # Only the given filters, as strings.
        wanted = {k: str(getattr(v, "value", v)) for k, v in filters.items() if v is not None}
        # User descriptions with the same filters.
        for entry in cls._models.values():
            # Every given filter must match.
            if all(getattr(entry, key) == value for key, value in wanted.items()):
                # Keep it.
                entries.append(entry)
        # Return the list.
        return entries

    # Every model id.
    @classmethod
    def ids(cls) -> list[str]:
        # Ids of the listing.
        return [e.model_id for e in cls.list_all()]

    # Validate an input shape (bands, H, W) or (H, W) against a model.
    @classmethod
    def check_input(cls, model_id: str, shape: tuple[int, ...]) -> ModelEntry:
        # Description of the model.
        entry = cls.require(model_id)
        # Only images are accepted.
        if len(shape) not in (2, 3):
            # Explain the expected layout.
            raise ValueError(f"input must be (bands, height, width), got shape {tuple(shape)}")
        # Band count; 2-D inputs have one band.
        bands = shape[0] if len(shape) == 3 else 1
        # The band count must match the model.
        if entry.channels and bands != entry.in_channels:
            # Name the expected bands.
            raise ValueError(
                f"{entry.model_id} expects {entry.in_channels} bands "
                f"({', '.join(entry.channels)}), got {bands}"
            )  # End of the error.
        # Spatial size must be positive.
        if min(shape[-2:]) < 1:
            # Explain the problem.
            raise ValueError(f"input has an empty spatial size: {tuple(shape)}")
        # Return the description.
        return entry

    # Open a model, reusing an earlier opened instance.
    @classmethod
    def load(cls, model_id: str, variant: str | None = None, verify: bool = True) -> Any:
        # Cache key.
        key = (model_id, variant)
        # Reuse an opened model.
        if key in cls._loaded:
            # Return it.
            return cls._loaded[key]
        # Imported lazily because loading needs PyTorch.
        from unbihexium.zoo import load_model

        # Build or read the model.
        model = load_model(model_id, variant=variant, verify=verify)
        # Remember it.
        cls._loaded[key] = model
        # Return it.
        return model

    # Forget every opened model.
    @classmethod
    def clear_cache(cls) -> None:
        # Empty the cache.
        cls._loaded.clear()


# =============================================================================
# End of module src/unbihexium/registry/models.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
