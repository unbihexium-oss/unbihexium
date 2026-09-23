# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/catalog.py
# Title       : Model zoo catalogue: model families, tasks and size variants
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML
# =============================================================================
#
# Abstract
# --------
# Loads and validates catalog.yaml, the single source of truth for the 130
# model families of the model zoo, and describes the four size variants
# (tiny, base, large and mega) in which every family is available. The
# catalogue is pure data: it can be read without PyTorch, for example by the
# command line interface to list models or by documentation generators.
#
# Each family is a ModelSpec: task, input bands, number of acquisitions,
# outputs and their units. A ModelSpec combined with a VariantSpec fully
# determines the architecture of a model (see unbihexium.ai.models.factory).
#
# Terminology
# -----------
#   family   a model family such as "ship_detector"
#   variant  a size variant: tiny, base, large or mega
#   model id family and variant joined by an underscore, "ship_detector_base"
#
# Usage
# -----
#   from unbihexium.zoo.catalog import get_spec, list_specs, parse_model_id
#   spec = get_spec("ship_detector")
#   family, variant = parse_model_id("ship_detector_base")
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Immutable records for specifications.
from dataclasses import dataclass, field

# Enumerations of tasks and variant names.
from enum import Enum

# Cache the parsed catalogue so that the YAML file is read once.
from functools import lru_cache

# Locate the packaged catalog.yaml file.
from importlib import resources

# Type of loosely structured YAML values.
from typing import Any

# Parse catalog.yaml.
import yaml

# Name of the catalogue file inside this package.
CATALOG_FILE = "catalog.yaml"

# Licence of every model family in the zoo.
MODEL_LICENSE = "MPL-2.0"


# Tasks supported by the model zoo. The value is the name used in YAML files.
class Task(str, Enum):
    # Object detection with bounding boxes (CenterNet architecture).
    DETECTION = "detection"
    # Per-pixel classification (U-Net architecture).
    SEGMENTATION = "segmentation"
    # Per-pixel classification of change between two acquisitions.
    CHANGE_DETECTION = "change_detection"
    # Per-pixel regression of a continuous variable.
    DENSE_REGRESSION = "dense_regression"
    # One regression vector per image chip.
    SCENE_REGRESSION = "scene_regression"
    # Image-to-image translation at the input resolution.
    ENHANCEMENT = "enhancement"
    # Image-to-image translation to a higher resolution.
    SUPER_RESOLUTION = "super_resolution"
    # Exact spectral index formula; no training needed.
    SPECTRAL_INDEX = "spectral_index"

    # Tasks whose outputs are class labels.
    @property
    def is_classification(self) -> bool:
        # Detection, segmentation and change detection predict classes.
        return self in (Task.DETECTION, Task.SEGMENTATION, Task.CHANGE_DETECTION)

    # Tasks whose outputs are continuous values.
    @property
    def is_regression(self) -> bool:
        # Both regression tasks predict continuous targets.
        return self in (Task.DENSE_REGRESSION, Task.SCENE_REGRESSION)

    # Tasks whose output is a raster with one value per input pixel.
    @property
    def is_dense(self) -> bool:
        # Every task except detection and scene regression is dense.
        return self not in (Task.DETECTION, Task.SCENE_REGRESSION)

    # Tasks with trainable weights.
    @property
    def is_trainable(self) -> bool:
        # Spectral index modules implement fixed formulas.
        return self is not Task.SPECTRAL_INDEX


# The four size variants of every model family.
class Variant(str, Enum):
    # Smallest variant for edge devices and quick experiments.
    TINY = "tiny"
    # Default variant with a balance of accuracy and speed.
    BASE = "base"
    # Larger variant for higher accuracy on GPUs.
    LARGE = "large"
    # Largest variant for maximum capacity.
    MEGA = "mega"


# Architecture hyperparameters of a size variant.
@dataclass(frozen=True)
class VariantSpec:
    # Variant this specification belongs to.
    variant: Variant
    # Number of feature channels of the first encoder stage.
    base_channels: int
    # Number of down-sampling stages of the encoder.
    depth: int
    # Number of residual blocks per encoder stage.
    blocks_per_stage: int
    # Recommended square tile size in pixels for training and inference.
    tile_size: int
    # Channel width of the regression and detection heads.
    head_channels: int


# Hyperparameters of the four variants. Channel widths double at each stage
# and are capped at 8 times base_channels to keep the mega variant tractable.
VARIANTS: dict[Variant, VariantSpec] = {
    # tiny: 16 channels, 3 stages, 1 block per stage.
    Variant.TINY: VariantSpec(Variant.TINY, 16, 3, 1, 256, 32),
    # base: 32 channels, 4 stages, 1 block per stage.
    Variant.BASE: VariantSpec(Variant.BASE, 32, 4, 1, 256, 64),
    # large: 48 channels, 4 stages, 2 blocks per stage.
    Variant.LARGE: VariantSpec(Variant.LARGE, 48, 4, 2, 512, 96),
    # mega: 64 channels, 5 stages, 2 blocks per stage.
    Variant.MEGA: VariantSpec(Variant.MEGA, 64, 5, 2, 512, 128),
}  # End of the variant table.


# Specification of one model family, as read from catalog.yaml.
@dataclass(frozen=True)
class ModelSpec:
    # Family identifier, for example "ship_detector".
    family: str
    # Human-readable name.
    name: str
    # Task of the family.
    task: Task
    # Capability domain, for example "agriculture".
    domain: str
    # What the model does once it is trained.
    description: str
    # Names of the input bands of one acquisition.
    bands: tuple[str, ...]
    # Number of acquisitions stacked on the channel axis.
    dates: int
    # Class names, target names or output band names.
    outputs: tuple[str, ...]
    # Units of regression targets, one per output (empty for classes).
    units: tuple[str, ...] = ()
    # Optional (minimum, maximum) of regression targets.
    value_range: tuple[float, float] | None = None
    # Upscaling factor of super-resolution models.
    scale: int = 1
    # Formula identifier of spectral index models.
    formula: str | None = None
    # Reference data needed for training.
    labels: str = ""
    # Suitable input data sources.
    sources: tuple[str, ...] = field(default_factory=tuple)

    # Total number of input channels: bands times acquisitions.
    @property
    def in_channels(self) -> int:
        # Acquisitions are stacked on the channel axis.
        return len(self.bands) * self.dates

    # Names of all input channels in the order the model expects them.
    @property
    def channel_names(self) -> tuple[str, ...]:
        # A single acquisition uses the band names unchanged.
        if self.dates == 1:
            # Return the band names as they are.
            return self.bands
        # Multiple acquisitions get a _t1, _t2, ... suffix per date.
        return tuple(f"{band}_t{d}" for d in range(1, self.dates + 1) for band in self.bands)

    # Number of output channels of the network.
    @property
    def out_channels(self) -> int:
        # One output channel per class, target or output band.
        return len(self.outputs)

    # True when a regression output is bounded to [0, 1] and uses a sigmoid.
    @property
    def sigmoid_output(self) -> bool:
        # Only regression tasks have value ranges.
        return self.task.is_regression and self.value_range == (0.0, 1.0)

    # Build the model identifier of a variant of this family.
    def model_id(self, variant: Variant | str) -> str:
        # Normalise the variant to its enumeration member.
        variant = Variant(variant)
        # Join family and variant with an underscore.
        return f"{self.family}_{variant.value}"

    # Serialise the specification to plain Python types.
    def to_dict(self) -> dict[str, Any]:
        # Plain dictionary suitable for JSON and YAML output.
        return {
            "family": self.family,  # Family identifier.
            "name": self.name,  # Human-readable name.
            "task": self.task.value,  # Task name.
            "domain": self.domain,  # Capability domain.
            "description": self.description,  # Purpose of the model.
            "bands": list(self.bands),  # Bands of one acquisition.
            "dates": self.dates,  # Number of acquisitions.
            "in_channels": self.in_channels,  # Total input channels.
            "channel_names": list(self.channel_names),  # All input channels.
            "outputs": list(self.outputs),  # Classes, targets or bands.
            "units": list(self.units),  # Units of regression targets.
            "value_range": list(self.value_range) if self.value_range else None,  # Bounds.
            "scale": self.scale,  # Upscaling factor.
            "formula": self.formula,  # Spectral index formula.
            "labels": self.labels,  # Training reference data.
            "sources": list(self.sources),  # Suitable input data.
            "license": MODEL_LICENSE,  # Licence of the model.
        }  # End of the dictionary.


# Raised when catalog.yaml or a lookup is invalid.
class CatalogError(ValueError):
    # No behaviour beyond ValueError; the class exists for precise handling.
    pass


# Resolve a band set name or an explicit band list to a tuple of band names.
def _resolve_bands(value: Any, band_sets: dict[str, list[str]], family: str) -> tuple[str, ...]:
    # A string refers to a named band set.
    if isinstance(value, str):
        # Unknown band set names are catalogue errors.
        if value not in band_sets:
            # Name the family and the missing band set.
            raise CatalogError(f"{family}: unknown band set {value!r}")
        # Return the bands of the named set.
        return tuple(band_sets[value])
    # A list gives the bands explicitly.
    if isinstance(value, list) and value and all(isinstance(b, str) for b in value):
        # Return the explicit bands.
        return tuple(value)
    # Anything else is malformed.
    raise CatalogError(f"{family}: bands must be a band set name or a list of names")


# Convert one YAML model entry to a validated ModelSpec.
def _parse_entry(entry: dict[str, Any], band_sets: dict[str, list[str]]) -> ModelSpec:
    # The family identifier is needed for every error message.
    family = str(entry.get("id", "")).strip()
    # Identifiers are lowercase snake case.
    if not family or not family.replace("_", "").isalnum() or family != family.lower():
        # Reject empty or malformed identifiers.
        raise CatalogError(f"invalid model family id {family!r}")
    # Convert the task name, rejecting unknown tasks.
    try:
        # Look up the task enumeration member.
        task = Task(entry["task"])
    # A missing or unknown task is a catalogue error.
    except (KeyError, ValueError) as exc:
        # Report the family and the problem.
        raise CatalogError(f"{family}: missing or unknown task") from exc
    # Outputs are required and must be non-empty.
    outputs = tuple(str(o) for o in entry.get("outputs") or ())
    # Every model must produce at least one output.
    if not outputs:
        # Report the missing outputs.
        raise CatalogError(f"{family}: outputs must not be empty")
    # Units are optional but must match the outputs when present.
    units = tuple(str(u) for u in entry.get("units") or ())
    # A units list of the wrong length is ambiguous.
    if units and len(units) != len(outputs):
        # Report the mismatch.
        raise CatalogError(f"{family}: units must have one entry per output")
    # The optional value range is a pair of numbers.
    raw_range = entry.get("range")
    # Convert the range to floats when present.
    value_range = (float(raw_range[0]), float(raw_range[1])) if raw_range else None
    # A range must be increasing.
    if value_range and value_range[0] >= value_range[1]:
        # Report the invalid range.
        raise CatalogError(f"{family}: range minimum must be below the maximum")
    # Number of acquisitions; defaults to one.
    dates = int(entry.get("dates", 1))
    # At least one acquisition is needed.
    if dates < 1:
        # Report the invalid number of dates.
        raise CatalogError(f"{family}: dates must be at least 1")
    # Spectral index models must name the formula they implement.
    if task is Task.SPECTRAL_INDEX and not entry.get("formula"):
        # Spectral index models must name their formula.
        raise CatalogError(f"{family}: spectral_index models need a formula")
    # Super-resolution models need an integer scale of at least 2.
    scale = int(entry.get("scale", 1))
    # Validate the scale for super-resolution models only.
    if task is Task.SUPER_RESOLUTION and scale < 2:
        # Report the invalid scale.
        raise CatalogError(f"{family}: super_resolution models need scale >= 2")
    # Build the immutable specification.
    return ModelSpec(
        family=family,  # Family identifier.
        name=str(entry.get("name", family)),  # Human-readable name.
        task=task,  # Task enumeration member.
        domain=str(entry.get("domain", "ai")),  # Capability domain.
        description=str(entry.get("description", "")),  # Purpose.
        bands=_resolve_bands(entry.get("bands"), band_sets, family),  # Input bands.
        dates=dates,  # Number of acquisitions.
        outputs=outputs,  # Classes, targets or bands.
        units=units,  # Target units.
        value_range=value_range,  # Target bounds.
        scale=scale,  # Upscaling factor.
        formula=entry.get("formula"),  # Index formula.
        labels=str(entry.get("labels", "")),  # Training data.
        sources=tuple(str(s) for s in entry.get("sources") or ()),  # Input data.
    )  # End of the specification.


# Read and validate the catalogue; the result is cached.
@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, ModelSpec]:
    # Read catalog.yaml from the installed package.
    text = resources.files("unbihexium.zoo").joinpath(CATALOG_FILE).read_text(encoding="utf-8")
    # Parse the YAML document.
    data = yaml.safe_load(text)
    # Named band sets referenced by the entries.
    band_sets = data.get("band_sets", {})
    # Specifications keyed by family identifier, in catalogue order.
    specs: dict[str, ModelSpec] = {}
    # Parse every model entry.
    for entry in data.get("models", []):
        # Convert and validate the entry.
        spec = _parse_entry(entry, band_sets)
        # Family identifiers must be unique.
        if spec.family in specs:
            # Report the duplicate.
            raise CatalogError(f"duplicate model family {spec.family!r}")
        # Store the specification.
        specs[spec.family] = spec
    # Return the validated catalogue.
    return specs


# Version of the catalogue format and content.
def catalog_version() -> str:
    # Read the version field of catalog.yaml.
    text = resources.files("unbihexium.zoo").joinpath(CATALOG_FILE).read_text(encoding="utf-8")
    # Return it as a string.
    return str(yaml.safe_load(text).get("version", "0"))


# Return the specifications of all model families, optionally filtered.
def list_specs(task: Task | str | None = None, domain: str | None = None) -> list[ModelSpec]:
    # Start from all families in catalogue order.
    specs = list(_load_catalog().values())
    # Filter by task when requested.
    if task is not None:
        # Normalise the task to its enumeration member.
        task = Task(task)
        # Keep the families of that task.
        specs = [s for s in specs if s.task is task]
    # Filter by domain when requested.
    if domain is not None:
        # Keep the families of that domain.
        specs = [s for s in specs if s.domain == domain]
    # Return the filtered list.
    return specs


# Split a model id such as "ship_detector_base" into family and variant.
def parse_model_id(model_id: str) -> tuple[str, Variant]:
    # The variant is the text after the last underscore.
    family, _, suffix = model_id.rpartition("_")
    # A known variant suffix separates family and variant.
    if family and suffix in {v.value for v in Variant}:
        # Return the family and the variant.
        return family, Variant(suffix)
    # Without a suffix the id names a family; the base variant is implied.
    return model_id, Variant.BASE


# Return the specification of a model family or model id.
def get_spec(name: str) -> ModelSpec:
    # Accept both family names and model ids.
    family, _ = parse_model_id(name)
    # Look the family up in the catalogue.
    specs = _load_catalog()
    # Unknown names are errors with a helpful message.
    if family not in specs:
        # Report the unknown name.
        raise CatalogError(f"unknown model {name!r}; see `unbihexium zoo list`")
    # Return the specification.
    return specs[family]


# Return the hyperparameters of a size variant.
def get_variant(variant: Variant | str) -> VariantSpec:
    # Normalise and look up the variant.
    return VARIANTS[Variant(variant)]


# Return every model id of the zoo: families times variants.
def all_model_ids() -> list[str]:
    # One id per family and variant, variants in size order.
    return [spec.model_id(v) for spec in _load_catalog().values() for v in Variant]


# =============================================================================
# End of module src/unbihexium/zoo/catalog.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
