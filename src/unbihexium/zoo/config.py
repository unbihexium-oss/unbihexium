# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/config.py
# Title       : Effective input and output configuration of a model
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, no optional dependencies
# =============================================================================
#
# Abstract
# --------
# BuildConfig describes a concrete model: its family, variant and task, the
# names of its input channels and outputs, units, value range, upscaling
# factor, spectral index formula and tile size. Checkpoints, ONNX metadata,
# config.json files in the model store, the inference code and the training
# code all exchange this record. It lives in the torch-free zoo package so
# that ONNX Runtime inference works without PyTorch.
#
# The extra dictionary carries metadata added after construction, most
# importantly the per-band normalisation statistics recorded by training
# ("normalization": {"mean": [...], "std": [...]}) and the training summary.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Immutable configuration records and copies with replaced fields.
from dataclasses import dataclass, field, replace

# Type of loosely structured values.
from typing import Any

# Catalogue types and lookups.
from unbihexium.zoo.catalog import ModelSpec, Task, Variant, get_variant

# Output stride of the CenterNet detectors in ai/models/networks.py.
DETECTION_STRIDE = 4


# Effective configuration of a built model.
@dataclass(frozen=True)
class BuildConfig:
    # Model id: family and variant.
    model_id: str
    # Model family.
    family: str
    # Size variant.
    variant: Variant
    # Task of the model.
    task: Task
    # Names of the input channels, in order.
    channel_names: tuple[str, ...]
    # Class, target or output band names.
    outputs: tuple[str, ...]
    # Units of regression targets.
    units: tuple[str, ...] = ()
    # Bounds of regression targets.
    value_range: tuple[float, float] | None = None
    # Upscaling factor of super-resolution models.
    scale: int = 1
    # Formula of spectral index models.
    formula: str | None = None
    # Recommended square tile size.
    tile_size: int = 256
    # True when channel_names or outputs differ from the catalogue.
    customised: bool = False
    # Additional metadata, for example the training history.
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    # Number of input channels.
    @property
    def in_channels(self) -> int:
        # One channel per name.
        return len(self.channel_names)

    # Number of network outputs (classes, targets or bands).
    @property
    def out_channels(self) -> int:
        # One output per name.
        return len(self.outputs)

    # Spatial stride between the input and the output grid.
    @property
    def output_stride(self) -> int:
        # Detectors predict on a grid four times coarser than the input.
        return DETECTION_STRIDE if self.task is Task.DETECTION else 1

    # Input sizes that are multiples of this value avoid padding in the network.
    @property
    def size_multiple(self) -> int:
        # Every encoder level halves the resolution once.
        return 2 ** get_variant(self.variant).depth

    # Per-band normalisation statistics recorded by training, if any.
    @property
    def normalization(self) -> dict[str, list[float]] | None:
        # Stored under a fixed key of the extra metadata.
        value = self.extra.get("normalization")
        # Return a copy so that callers cannot change the configuration.
        return dict(value) if value else None

    # Return a copy with additional metadata merged into extra.
    def with_extra(self, **values: Any) -> BuildConfig:
        # Merge the old and the new metadata.
        merged = {**self.extra, **values}
        # Frozen records are copied with replace.
        return replace(self, extra=merged)

    # Serialise to plain Python types for checkpoints and JSON files.
    def to_dict(self) -> dict[str, Any]:
        # Plain dictionary.
        return {
            "model_id": self.model_id,  # Model id.
            "family": self.family,  # Model family.
            "variant": self.variant.value,  # Size variant.
            "task": self.task.value,  # Task.
            "channel_names": list(self.channel_names),  # Input channels.
            "outputs": list(self.outputs),  # Outputs.
            "units": list(self.units),  # Target units.
            "value_range": list(self.value_range) if self.value_range else None,  # Bounds.
            "scale": self.scale,  # Upscaling factor.
            "formula": self.formula,  # Index formula.
            "tile_size": self.tile_size,  # Tile size.
            "customised": self.customised,  # Deviates from the catalogue.
            "extra": dict(self.extra),  # Additional metadata.
        }  # End of the dictionary.

    # Rebuild a configuration from its dictionary form.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildConfig:
        # Convert every field back to its Python type.
        return cls(
            model_id=str(data["model_id"]),  # Model id.
            family=str(data["family"]),  # Model family.
            variant=Variant(data["variant"]),  # Size variant.
            task=Task(data["task"]),  # Task.
            channel_names=tuple(data["channel_names"]),  # Input channels.
            outputs=tuple(data["outputs"]),  # Outputs.
            units=tuple(data.get("units") or ()),  # Target units.
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,  # Bounds.
            scale=int(data.get("scale", 1)),  # Upscaling factor.
            formula=data.get("formula"),  # Index formula.
            tile_size=int(data.get("tile_size", 256)),  # Tile size.
            customised=bool(data.get("customised", False)),  # Deviation flag.
            extra=dict(data.get("extra") or {}),  # Additional metadata.
        )  # End of the configuration.

    # Derive the configuration of a catalogue entry and variant.
    @classmethod
    def from_spec(
        cls,  # The class itself.
        spec: ModelSpec,  # Catalogue entry.
        variant: Variant | str,  # Size variant.
        channel_names: tuple[str, ...] | list[str] | None = None,  # Input override.
        outputs: tuple[str, ...] | list[str] | None = None,  # Output override.
    ) -> BuildConfig:  # The derived configuration.
        # Normalise the variant.
        variant = Variant(variant)
        # Use the catalogue channels unless overridden.
        names = tuple(channel_names) if channel_names else spec.channel_names
        # Use the catalogue outputs unless overridden.
        outs = tuple(outputs) if outputs else spec.outputs
        # Units only apply when the outputs are unchanged.
        units = spec.units if outs == spec.outputs else ()
        # Build the configuration.
        return cls(
            model_id=spec.model_id(variant),  # Model id.
            family=spec.family,  # Model family.
            variant=variant,  # Size variant.
            task=spec.task,  # Task.
            channel_names=names,  # Input channels.
            outputs=outs,  # Outputs.
            units=units,  # Target units.
            value_range=spec.value_range,  # Bounds.
            scale=spec.scale,  # Upscaling factor.
            formula=spec.formula,  # Index formula.
            tile_size=get_variant(variant).tile_size,  # Tile size.
            customised=(names != spec.channel_names or outs != spec.outputs),  # Deviation flag.
        )  # End of the configuration.


# =============================================================================
# End of module src/unbihexium/zoo/config.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
