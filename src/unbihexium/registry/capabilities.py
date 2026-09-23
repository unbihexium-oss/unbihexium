# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/registry/capabilities.py
# Title       : Registry of the capabilities of the library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML (model catalogue)
# =============================================================================
#
# Abstract
# --------
# A capability is something the library can do, described with its domain,
# maturity and the entry points that implement it. The registry holds two
# kinds of built-in capabilities, loaded on first access:
#
#   model capabilities     one per model family of the model zoo catalogue
#                          (unbihexium.zoo), with the four size variants as
#                          its models, the task and the input bands; they
#                          run through unbihexium.ai.predict. Families with
#                          a registered pipeline name it. Spectral index
#                          families compute a published formula and are
#                          stable; the other families are starter models
#                          that must be trained, so they are beta.
#   library capabilities   algorithms and formats implemented directly in
#                          the library: raster and vector input and output,
#                          spectral indices, SAR processing, terrain,
#                          geostatistics, spatial analysis, preprocessing,
#                          metrics, the model zoo, training and serving
#
# Users may register their own capabilities. Ids are unique; registering an
# existing id raises ValueError unless replace=True.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Records.
from dataclasses import dataclass, field

# Enumerations.
from enum import Enum

# Type of loosely structured values.
from typing import Any


# Capability domains; the model catalogue uses the same names.
class CapabilityDomain(str, Enum):
    # Generic AI models.
    AI = "ai"
    # Tourism and destination analysis.
    TOURISM = "tourism"
    # Spatial analysis.
    ANALYSIS = "analysis"
    # Spectral indices.
    INDICES = "indices"
    # Water and floods.
    WATER = "water"
    # Environment monitoring.
    ENVIRONMENT = "environment"
    # Forestry.
    FORESTRY = "forestry"
    # Image processing.
    IMAGING = "imaging"
    # Asset management.
    ASSETS = "assets"
    # Energy infrastructure.
    ENERGY = "energy"
    # Urban planning.
    URBAN = "urban"
    # Agriculture.
    AGRICULTURE = "agriculture"
    # Risk and insurance.
    RISK = "risk"
    # Defence and security (neutral monitoring).
    DEFENSE = "defense"
    # Synthetic aperture radar.
    SAR = "sar"
    # Data input and output.
    IO = "io"


# Maturity of a capability.
class CapabilityMaturity(str, Enum):
    # Tested and ready for production use.
    STABLE = "stable"
    # Complete but needs training or wider validation.
    BETA = "beta"
    # Experimental.
    RESEARCH = "research"
    # Scheduled for removal.
    DEPRECATED = "deprecated"


# Size variants of every catalogue family.
VARIANTS = ("tiny", "base", "large", "mega")

# Pipelines registered by unbihexium.ai, by model family.
PIPELINES_BY_FAMILY = {
    "ship_detector": "ship_detection",  # Ship detection pipeline.
    "building_detector": "building_detection",  # Building detection pipeline.
    "change_detector": "change_detection",  # Change detection pipeline.
    "water_surface_detector": "water_detection",  # Water detection pipeline.
    "super_resolution": "super_resolution",  # Super-resolution pipeline.
}  # End of the pipeline mapping.


# One capability of the library.
@dataclass
class Capability:
    # Unique id.
    capability_id: str
    # Human-readable name.
    name: str
    # Domain of the capability.
    domain: CapabilityDomain
    # What the capability does.
    description: str = ""
    # Maturity level.
    maturity: CapabilityMaturity = CapabilityMaturity.STABLE
    # Modules or functions that implement it, as import paths.
    entry_points: list[str] = field(default_factory=list)
    # Registered pipeline that runs it, if any.
    pipeline_id: str | None = None
    # Command line invocation, if any.
    cli_command: str | None = None
    # Example script, relative to the repository.
    example_path: str | None = None
    # Tests, relative to the repository.
    test_path: str | None = None
    # Documentation page, relative to the repository.
    docs_path: str | None = None
    # Model zoo family, for model capabilities.
    model_family: str | None = None
    # Model task, for model capabilities.
    task: str | None = None
    # Input bands of one acquisition, for model capabilities.
    bands: list[str] = field(default_factory=list)
    # Free-form metadata.
    tags: dict[str, str] = field(default_factory=dict)

    # Model ids of the capability: the size variants of its family.
    @property
    def models(self) -> list[str]:
        # No models for library capabilities.
        if self.model_family is None:
            # Empty list.
            return []
        # One id per variant.
        return [f"{self.model_family}_{v}" for v in VARIANTS]

    # JSON-serialisable description.
    def to_dict(self) -> dict[str, Any]:
        # Every field with enumerations as their values.
        return {
            "capability_id": self.capability_id,  # Id.
            "name": self.name,  # Name.
            "domain": self.domain.value,  # Domain.
            "description": self.description,  # Description.
            "maturity": self.maturity.value,  # Maturity.
            "entry_points": list(self.entry_points),  # Implementations.
            "pipeline_id": self.pipeline_id,  # Pipeline.
            "cli_command": self.cli_command,  # Command line.
            "docs_path": self.docs_path,  # Documentation.
            "model_family": self.model_family,  # Model family.
            "models": self.models,  # Model ids.
            "task": self.task,  # Task.
            "bands": list(self.bands),  # Input bands.
            "tags": dict(self.tags),  # Metadata.
        }  # End of the dictionary.


# Library capabilities: (id, name, domain, description, entry points).
LIBRARY_CAPABILITIES: tuple[tuple[str, str, CapabilityDomain, str, tuple[str, ...]], ...] = (
    (  # Row of the io_geotiff capability.
        "io_geotiff",  # Id.
        "GeoTIFF and COG input and output",  # Name.
        CapabilityDomain.IO,  # Domain.
        "Read and write GeoTIFF with windows, overviews and the COG layout.",  # Description.
        ("unbihexium.io.geotiff",),  # Entry points.
    ),
    (  # Row of the io_zarr capability.
        "io_zarr",  # Id.
        "Zarr input and output",  # Name.
        CapabilityDomain.IO,  # Domain.
        "Chunked, compressed arrays and georeferenced rasters in Zarr v2 and v3.",  # Description.
        ("unbihexium.io.zarr_io",),  # Entry points.
    ),
    (  # Row of the io_geojson capability.
        "io_geojson",  # Id.
        "GeoJSON input and output",  # Name.
        CapabilityDomain.IO,  # Domain.
        "RFC 7946 validation, bounds, ring orientation and reprojection.",  # Description.
        ("unbihexium.io.geojson",),  # Entry points.
    ),
    (  # Row of the io_geoparquet capability.
        "io_geoparquet",  # Id.
        "GeoParquet input and output",  # Name.
        CapabilityDomain.IO,  # Domain.
        "Vector data in GeoParquet with its geo metadata.",  # Description.
        ("unbihexium.io.parquet",),  # Entry points.
    ),
    (  # Row of the io_stac capability.
        "io_stac",  # Id.
        "STAC catalogues and search",  # Name.
        CapabilityDomain.IO,  # Domain.
        "Parse, filter and search STAC items, collections and APIs.",  # Description.
        ("unbihexium.io.stac",),  # Entry points.
    ),
    (  # Row of the spectral_indices capability.
        "spectral_indices",  # Id.
        "Spectral indices",  # Name.
        CapabilityDomain.INDICES,  # Domain.
        "Vegetation, water, burn and moisture indices from reflectance bands.",  # Description.
        ("unbihexium.indices",),  # Entry points.
    ),
    (  # Row of the sar_processing capability.
        "sar_processing",  # Id.
        "SAR processing",  # Name.
        CapabilityDomain.SAR,  # Domain.
        "Calibration, speckle filtering, interferometry and polarimetry.",  # Description.
        ("unbihexium.sar",),  # Entry points.
    ),
    (  # Row of the terrain_analysis capability.
        "terrain_analysis",  # Id.
        "Terrain analysis",  # Name.
        CapabilityDomain.ANALYSIS,  # Domain.
        "Slope, aspect, hillshade, curvature and wetness from elevation models.",  # Description.
        ("unbihexium.terrain",),  # Entry points.
    ),
    (  # Row of the geostatistics capability.
        "geostatistics",  # Id.
        "Geostatistics",  # Name.
        CapabilityDomain.ANALYSIS,  # Domain.
        "Variograms, kriging and spatial autocorrelation.",  # Description.
        ("unbihexium.geostat",),  # Entry points.
    ),
    (  # Row of the spatial_analysis capability.
        "spatial_analysis",  # Id.
        "Spatial analysis",  # Name.
        CapabilityDomain.ANALYSIS,  # Domain.
        "Zonal statistics, suitability analysis and network analysis.",  # Description.
        ("unbihexium.analysis",),  # Entry points.
    ),
    (  # Row of the image_preprocessing capability.
        "image_preprocessing",  # Id.
        "Image preprocessing",  # Name.
        CapabilityDomain.IMAGING,  # Domain.
        "Radiometry, masks, resampling, pansharpening and enhancement.",  # Description.
        ("unbihexium.preprocessing",),  # Entry points.
    ),
    (  # Row of the prediction_postprocessing capability.
        "prediction_postprocessing",  # Id.
        "Prediction postprocessing",  # Name.
        CapabilityDomain.IMAGING,  # Domain.
        "Cleaning and vectorising of class maps and detections.",  # Description.
        ("unbihexium.postprocessing",),  # Entry points.
    ),
    (  # Row of the accuracy_metrics capability.
        "accuracy_metrics",  # Id.
        "Accuracy metrics",  # Name.
        CapabilityDomain.ANALYSIS,  # Domain.
        "Classification, detection and regression accuracy measures.",  # Description.
        ("unbihexium.metrics", "unbihexium.ai.evaluation"),  # Entry points.
    ),
    (  # Row of the visualization capability.
        "visualization",  # Id.
        "Visualisation",  # Name.
        CapabilityDomain.IMAGING,  # Domain.
        "Colour composites, stretches and maps of rasters and results.",  # Description.
        ("unbihexium.visualization",),  # Entry points.
    ),
    (  # Row of the model_zoo capability.
        "model_zoo",  # Id.
        "Model zoo",  # Name.
        CapabilityDomain.AI,  # Domain.
        "Catalogue of 130 model families in four sizes with a verified store.",  # Description.
        ("unbihexium.zoo",),  # Entry points.
    ),
    (  # Row of the model_training capability.
        "model_training",  # Id.
        "Model training",  # Name.
        CapabilityDomain.AI,  # Domain.
        "Training, fine-tuning and evaluation of model zoo models.",  # Description.
        ("unbihexium.ai.training", "unbihexium.ai.evaluation"),  # Entry points.
    ),
    (  # Row of the model_serving capability.
        "model_serving",  # Id.
        "REST model serving",  # Name.
        CapabilityDomain.AI,  # Domain.
        "FastAPI service that runs any model zoo model on posted images.",  # Description.
        ("unbihexium.serving",),  # Entry points.
    ),
)  # End of the library capabilities.


# Capabilities of the model catalogue, one per family.
def catalogue_capabilities() -> list[Capability]:
    # Imported here so that the registry does not load the catalogue on import.
    from unbihexium.zoo.catalog import Task, list_specs

    # One capability per family.
    found = []
    # Visit every family.
    for spec in list_specs():
        # Formula models need no training.
        trained = spec.task is Task.SPECTRAL_INDEX
        # Formulas are stable; starter models are beta until trained.
        maturity = CapabilityMaturity.STABLE if trained else CapabilityMaturity.BETA
        # Capability of the family.
        found.append(
            Capability(  # Record of the family.
                capability_id=spec.family,  # The family id.
                name=spec.name,  # Name.
                domain=CapabilityDomain(spec.domain),  # Domain.
                description=spec.description,  # Description.
                maturity=maturity,  # Maturity.
                entry_points=["unbihexium.ai.predict.predict"],  # Generic runner.
                pipeline_id=PIPELINES_BY_FAMILY.get(spec.family),  # Pipeline, if any.
                cli_command=f"unbihexium predict {spec.family}_base INPUT OUTPUT",  # Command.
                docs_path="docs/model_zoo/model_catalog.md",  # Documentation.
                model_family=spec.family,  # Family.
                task=spec.task.value,  # Task.
                bands=list(spec.bands),  # Input bands.
                tags={"requires_training": str(not trained).lower()},  # Starter model flag.
            )  # End of the capability.
        )  # End of the append.
    # Return the capabilities.
    return found


# Library capabilities as records.
def library_capabilities() -> list[Capability]:
    # One record per table row.
    return [
        Capability(  # Record of the row.
            capability_id=cid,  # Id.
            name=name,  # Name.
            domain=domain,  # Domain.
            description=description,  # Description.
            entry_points=list(entry_points),  # Implementations.
        )  # End of the capability.
        for cid, name, domain, description, entry_points in LIBRARY_CAPABILITIES  # Rows.
    ]  # End of the list.


# Class-level registry of capabilities.
class CapabilityRegistry:
    # Capabilities by id, shared by the whole process.
    _capabilities: dict[str, Capability] = {}
    # Whether the built-in capabilities were loaded.
    _loaded: bool = False

    # Load the built-in capabilities once.
    @classmethod
    def _ensure_loaded(cls) -> None:
        # Already loaded.
        if cls._loaded:
            # Nothing to do.
            return
        # Mark first so that register() does not recurse.
        cls._loaded = True
        # Library capabilities, then one per model family.
        for capability in library_capabilities() + catalogue_capabilities():
            # User capabilities registered earlier keep their place.
            cls._capabilities.setdefault(capability.capability_id, capability)

    # Register a capability; duplicates raise unless replace is set.
    @classmethod
    def register(cls, capability: Capability, replace: bool = False) -> Capability:
        # Built-ins first, so that duplicates are detected.
        cls._ensure_loaded()
        # Existing id.
        if capability.capability_id in cls._capabilities and not replace:
            # Explain the conflict.
            raise ValueError(f"capability {capability.capability_id!r} is already registered")
        # Store it.
        cls._capabilities[capability.capability_id] = capability
        # Return it, which allows use in expressions.
        return capability

    # Remove a capability; returns whether it existed.
    @classmethod
    def unregister(cls, capability_id: str) -> bool:
        # Built-ins first.
        cls._ensure_loaded()
        # Remove it if present.
        return cls._capabilities.pop(capability_id, None) is not None

    # Restore the built-in capabilities only.
    @classmethod
    def reset(cls) -> None:
        # Forget every capability.
        cls._capabilities = {}
        # Reload on next access.
        cls._loaded = False

    # Capability of an id, None when unknown.
    @classmethod
    def get(cls, capability_id: str) -> Capability | None:
        # Built-ins first.
        cls._ensure_loaded()
        # Dictionary lookup.
        return cls._capabilities.get(capability_id)

    # Capability of an id; unknown ids raise KeyError.
    @classmethod
    def require(cls, capability_id: str) -> Capability:
        # Look the id up.
        capability = cls.get(capability_id)
        # Unknown id.
        if capability is None:
            # Explain the problem.
            raise KeyError(f"unknown capability {capability_id!r}")
        # Return it.
        return capability

    # Every capability, sorted by id.
    @classmethod
    def list_all(cls) -> list[Capability]:
        # Built-ins first.
        cls._ensure_loaded()
        # Sorted for stable listings.
        return [cls._capabilities[k] for k in sorted(cls._capabilities)]

    # Alias of list_all kept for earlier releases.
    @classmethod
    def list_capabilities(cls) -> list[Capability]:
        # Same as list_all.
        return cls.list_all()

    # Every id, sorted.
    @classmethod
    def ids(cls) -> list[str]:
        # Ids of the sorted listing.
        return [c.capability_id for c in cls.list_all()]

    # Capabilities of a domain, given as a member or its value.
    @classmethod
    def by_domain(cls, domain: CapabilityDomain | str) -> list[Capability]:
        # Enumeration member; unknown names raise ValueError.
        wanted = CapabilityDomain(domain)
        # Matching capabilities.
        return [c for c in cls.list_all() if c.domain is wanted]

    # Capabilities of a maturity level.
    @classmethod
    def by_maturity(cls, maturity: CapabilityMaturity | str) -> list[Capability]:
        # Enumeration member.
        wanted = CapabilityMaturity(maturity)
        # Matching capabilities.
        return [c for c in cls.list_all() if c.maturity is wanted]

    # Capabilities of a model task, for example "detection".
    @classmethod
    def by_task(cls, task: str) -> list[Capability]:
        # Accept enumeration members as well as strings.
        name = str(getattr(task, "value", task))
        # Matching capabilities.
        return [c for c in cls.list_all() if c.task == name]

    # Capability that provides a model id, None when no family matches.
    @classmethod
    def for_model(cls, model_id: str) -> Capability | None:
        # Imported here to keep the registry import light.
        from unbihexium.zoo.catalog import parse_model_id

        # Family of the model id.
        family, _ = parse_model_id(model_id)
        # Capability of the family.
        capability = cls.get(family)
        # Only model capabilities provide models.
        return capability if capability and capability.model_family == family else None

    # Capabilities whose id, name or description contains the text.
    @classmethod
    def search(cls, text: str) -> list[Capability]:
        # Lower-case query.
        query = text.lower()
        # Matching capabilities.
        return [
            c  # Matching capability.
            for c in cls.list_all()  # Every capability.
            if query in f"{c.capability_id} {c.name} {c.description}".lower()  # Text match.
        ]  # End of the matches.

    # Number of capabilities per domain.
    @classmethod
    def domain_counts(cls) -> dict[str, int]:
        # Counts by domain value.
        counts: dict[str, int] = {}
        # Visit every capability.
        for c in cls.list_all():
            # Increment the count.
            counts[c.domain.value] = counts.get(c.domain.value, 0) + 1
        # Return the counts.
        return counts


# =============================================================================
# End of module src/unbihexium/registry/capabilities.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
