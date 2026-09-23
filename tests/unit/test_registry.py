# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_registry.py
# Title       : Tests of the capability, model and pipeline registries
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest; no PyTorch needed
# =============================================================================
#
# Abstract
# --------
# The capability registry holds one capability per model zoo family (130)
# and the 17 library capabilities, with unique ids, valid domains, existing
# entry-point packages and model ids that resolve in the zoo. The model
# registry exposes the 520 zoo models and validates input shapes against the
# catalogue band sets. The pipeline registry keeps the decorator API used by
# unbihexium.ai.base and returns entries sorted by id.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Locate packages without importing them.
import importlib.util

# Test framework.
import pytest

# Registries under test.
from unbihexium.registry import (
    Capability,  # Capability record.
    CapabilityDomain,  # Domains.
    CapabilityMaturity,  # Maturity levels.
    CapabilityRegistry,  # Capability registry.
    ModelEntry,  # Model record.
    ModelRegistry,  # Model registry.
    PipelineRegistry,  # Pipeline registry.
    get_capability,  # Shortcut.
    list_capabilities,  # Shortcut.
    register_capability,  # Shortcut.
)  # End of the registry imports.

# Capability tables.
from unbihexium.registry.capabilities import LIBRARY_CAPABILITIES, PIPELINES_BY_FAMILY

# Model zoo catalogue.
from unbihexium.zoo import get_model, list_specs


# Whether an import path names a module, or a function of a module, that exists.
def importable(path: str) -> bool:
    # Modules and packages.
    try:
        # Found without importing.
        if importlib.util.find_spec(path) is not None:
            # A module.
            return True
    # The parent is a module, so the path names an attribute.
    except ModuleNotFoundError:
        # Try the parent below.
        pass
    # Module of a function.
    return importlib.util.find_spec(path.rsplit(".", 1)[0]) is not None


# The registry is populated from the catalogue and the library table.
def test_builtin_capabilities() -> None:
    # Every identifier.
    ids = CapabilityRegistry.ids()
    # 130 model families plus the library capabilities, all unique.
    assert len(ids) == len(set(ids)) == len(list_specs()) + len(LIBRARY_CAPABILITIES)
    # Ids are sorted.
    assert ids == sorted(ids)
    # Every catalogue family is a capability with its four variants.
    for spec in list_specs():
        # Capability of the family.
        capability = CapabilityRegistry.require(spec.family)
        # Task and bands from the catalogue.
        assert capability.task == spec.task.value and capability.bands == list(spec.bands)
        # Every model id resolves in the zoo.
        assert all(get_model(m) is not None for m in capability.models)
    # Every entry point names an existing package or module.
    for capability in CapabilityRegistry.list_all():
        # Library capabilities point to modules.
        for entry in capability.entry_points:
            # The module (or the module of the function) exists.
            assert importable(entry), entry


# Maturity follows the training state of the models.
def test_capability_maturity_and_domains() -> None:
    # The NDVI formula needs no training.
    ndvi = CapabilityRegistry.require("ndvi_calculator")
    # Stable and flagged as trained.
    assert ndvi.maturity is CapabilityMaturity.STABLE and ndvi.tags["requires_training"] == "false"
    # Starter detectors are beta until trained.
    ship = CapabilityRegistry.require("ship_detector")
    # Beta with a pipeline and a command.
    assert ship.maturity is CapabilityMaturity.BETA and ship.pipeline_id == "ship_detection"
    # Command line of the family.
    assert ship.cli_command == "unbihexium predict ship_detector_base INPUT OUTPUT"
    # Four variants in size order.
    assert ship.models == [f"ship_detector_{v}" for v in ("tiny", "base", "large", "mega")]
    # Domain filters accept values and members.
    assert CapabilityRegistry.by_domain("io") == CapabilityRegistry.by_domain(CapabilityDomain.IO)
    # Five input and output formats.
    assert [c.capability_id for c in list_capabilities("io")] == [
        "io_geojson",  # GeoJSON.
        "io_geoparquet",  # GeoParquet.
        "io_geotiff",  # GeoTIFF.
        "io_stac",  # STAC.
        "io_zarr",  # Zarr.
    ]  # End of the expected ids.
    # Domain counts add up to the total.
    assert sum(CapabilityRegistry.domain_counts().values()) == len(CapabilityRegistry.ids())
    # Unknown domains are errors.
    with pytest.raises(ValueError):
        # No such domain.
        CapabilityRegistry.by_domain("astrology")
    # The seven spectral index families.
    indices = CapabilityRegistry.by_task("spectral_index")
    # Every one of them is stable.
    assert len(indices) == 7 and all(c.maturity is CapabilityMaturity.STABLE for c in indices)


# Lookups by model id, text search and serialisation.
def test_capability_lookups() -> None:
    # A variant id maps to its family.
    assert CapabilityRegistry.for_model("water_surface_detector_mega").capability_id == (
        "water_surface_detector"
    )  # End of the comparison.
    # Library capabilities provide no models.
    assert CapabilityRegistry.for_model("io_geotiff") is None
    # Search in names and descriptions, case-insensitive.
    assert "sar_processing" in [c.capability_id for c in CapabilityRegistry.search("SPECKLE")]
    # Dictionary form.
    data = get_capability("io_zarr").to_dict()
    # Enumerations become strings.
    assert data["domain"] == "io" and data["maturity"] == "stable" and data["models"] == []
    # Every pipeline named by a family exists after the task APIs are imported.
    import unbihexium.ai

    # Registered pipeline ids.
    pipelines = set(PipelineRegistry.ids())
    # Each mapping target is registered.
    assert set(PIPELINES_BY_FAMILY.values()) <= pipelines


# Registration, duplicates, replacement and removal.
def test_capability_registration() -> None:
    # A user capability.
    custom = Capability("test_custom_capability", "Custom", CapabilityDomain.ANALYSIS)
    # Register it.
    try:
        # Registration returns the record.
        assert register_capability(custom) is custom
        # It is listed.
        assert get_capability("test_custom_capability") is custom
        # Duplicates are errors.
        with pytest.raises(ValueError, match="already registered"):
            # Same id again.
            register_capability(Capability("test_custom_capability", "X", CapabilityDomain.AI))
        # Replacement is explicit.
        replaced = Capability("test_custom_capability", "Y", CapabilityDomain.AI)
        # Replace it.
        register_capability(replaced, replace=True)
        # The new record is stored.
        assert CapabilityRegistry.require("test_custom_capability").name == "Y"
    # Always clean up the shared registry.
    finally:
        # Remove it.
        assert CapabilityRegistry.unregister("test_custom_capability")
    # Unknown ids.
    with pytest.raises(KeyError):
        # Removed capability.
        CapabilityRegistry.require("test_custom_capability")
    # Built-in ids cannot be registered twice.
    with pytest.raises(ValueError):
        # Collides with a zoo family.
        register_capability(Capability("ship_detector", "Ship", CapabilityDomain.AI))


# The model registry mirrors the zoo and validates inputs.
def test_model_registry() -> None:
    # Every zoo model.
    assert len(ModelRegistry.ids()) == 520
    # Filters of the zoo.
    tiny_indices = ModelRegistry.list_all(task="spectral_index", variant="tiny")
    # Seven spectral index families.
    assert len(tiny_indices) == 7 and all(e.variant == "tiny" for e in tiny_indices)
    # NDVI takes red and near infrared.
    ndvi = ModelRegistry.require("ndvi_calculator_tiny")
    # Channels and outputs.
    assert ndvi.channels == ["red", "nir"] and ndvi.outputs == ["ndvi"] and ndvi.in_channels == 2
    # Family names resolve to the base variant.
    assert ModelRegistry.require("ndvi_calculator").model_id == "ndvi_calculator_base"
    # Change detectors take two stacked dates.
    change = ModelRegistry.require("change_detector_tiny")
    # Twice the bands of one date.
    assert change.in_channels == 2 * len(get_model("change_detector_tiny").spec.bands)
    # Matching shapes pass.
    assert ModelRegistry.check_input("ndvi_calculator_tiny", (2, 8, 8)) is not None
    # Wrong band counts name the expected bands.
    with pytest.raises(ValueError, match=r"expects 2 bands \(red, nir\), got 3"):
        # Three bands.
        ModelRegistry.check_input("ndvi_calculator_tiny", (3, 8, 8))
    # Single-band models accept 2-D shapes.
    single = next(e for e in ModelRegistry.list_all() if e.in_channels == 1)
    # A 2-D input has one band.
    assert ModelRegistry.check_input(single.model_id, (8, 8)).in_channels == 1
    # Unknown models.
    with pytest.raises(KeyError):
        # No such model.
        ModelRegistry.require("does_not_exist_tiny")


# User model descriptions.
def test_model_registration() -> None:
    # External model description.
    entry = ModelEntry("test_external_model", task="detection", source="external")
    # Register it.
    try:
        # Stored and listed after the zoo models.
        ModelRegistry.register(entry)
        # Lookup.
        assert ModelRegistry.get("test_external_model") is entry
        # Task filters include it.
        assert entry in ModelRegistry.list_all(task="detection")
        # Duplicates need replace.
        with pytest.raises(ValueError):
            # Same id.
            ModelRegistry.register(entry)
    # Clean up.
    finally:
        # Remove it.
        assert ModelRegistry.unregister("test_external_model")
    # Zoo ids cannot be shadowed.
    with pytest.raises(ValueError, match="model zoo id"):
        # Collides with the catalogue.
        ModelRegistry.register(ModelEntry("ship_detector_tiny"))


# The pipeline decorator API used by unbihexium.ai.base.
def test_pipeline_registry() -> None:
    # Register a factory.
    try:
        # Decorated factory.
        @PipelineRegistry.register("test_double", "Double", "Doubles a value", ["analysis"])
        def create(**kwargs: int) -> int:
            # Twice the value parameter.
            return 2 * kwargs.get("value", 0)

        # The decorator returns the factory unchanged.
        assert create(value=2) == 4
        # Creation passes the parameters.
        assert PipelineRegistry.create("test_double", value=21) == 42
        # Entry fields.
        assert PipelineRegistry.require("test_double").to_dict()["domains"] == ["analysis"]
        # Domain filter.
        assert "test_double" in [p.pipeline_id for p in PipelineRegistry.by_domain("analysis")]
        # Text search.
        assert [p.pipeline_id for p in PipelineRegistry.search("doubles")] == ["test_double"]
        # Alias of earlier releases.
        assert PipelineRegistry.list_pipelines() == PipelineRegistry.list_all()
    # Clean up.
    finally:
        # Remove it.
        PipelineRegistry.unregister("test_double")
    # Unknown ids give None from create and get.
    assert PipelineRegistry.create("test_double") is None and PipelineRegistry.get("x") is None
    # Invalid ids are rejected at registration.
    with pytest.raises(ValueError):
        # Spaces are not allowed.
        PipelineRegistry.register("bad id", "Bad")


# =============================================================================
# End of module tests/unit/test_registry.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
