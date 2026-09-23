# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_zoo_catalog.py
# Title       : Tests of the model zoo catalogue and registry
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest
# =============================================================================
#
# Abstract
# --------
# Checks that catalog.yaml describes exactly 130 valid model families in
# four variants, that model ids parse correctly, that every family has a
# consistent input and output layout, and that the registry exposes all 520
# models with published digests. These tests do not need PyTorch.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Count tasks.
from collections import Counter

# Test framework.
import pytest

# Catalogue and registry under test.
from unbihexium.zoo import (
    CatalogError,  # Raised for unknown names.
    Task,  # Task enumeration.
    Variant,  # Variant enumeration.
    all_model_ids,  # Every model id.
    get_model,  # Registry lookup.
    get_spec,  # Catalogue lookup.
    get_variant,  # Variant lookup.
    list_models,  # Registry listing.
    list_specs,  # Catalogue listing.
    parse_model_id,  # Model id parser.
)  # End of the imports.

# Expected number of families per task.
EXPECTED_TASKS = {
    "detection": 19,  # Object detectors.
    "segmentation": 26,  # Semantic segmentation.
    "change_detection": 6,  # Change detection.
    "dense_regression": 49,  # Per-pixel regression.
    "scene_regression": 11,  # Chip-level regression.
    "enhancement": 11,  # Image-to-image.
    "super_resolution": 1,  # Super-resolution.
    "spectral_index": 7,  # Spectral indices.
}  # End of the expected counts.


# The catalogue has 130 families with the expected task distribution.
def test_catalog_size_and_tasks() -> None:
    # All families.
    specs = list_specs()
    # 130 families.
    assert len(specs) == 130
    # Task distribution.
    assert Counter(s.task.value for s in specs) == EXPECTED_TASKS
    # 520 model ids, all unique.
    assert len(set(all_model_ids())) == 520


# Every family has a consistent input and output layout.
@pytest.mark.parametrize("spec", list_specs(), ids=lambda s: s.family)
def test_spec_is_consistent(spec) -> None:
    # At least one input channel.
    assert spec.in_channels >= 1
    # One channel name per input channel.
    assert len(spec.channel_names) == spec.in_channels
    # Channel names are unique.
    assert len(set(spec.channel_names)) == spec.in_channels
    # At least one output.
    assert spec.out_channels >= 1
    # Units, when given, match the outputs.
    assert not spec.units or len(spec.units) == len(spec.outputs)
    # Descriptions and label requirements are documented.
    assert spec.description and spec.labels and spec.sources
    # Segmentation needs at least two classes.
    if spec.task in (Task.SEGMENTATION, Task.CHANGE_DETECTION):
        # Background or no-change plus at least one class.
        assert spec.out_channels >= 2
    # Change detection stacks two acquisitions.
    if spec.task is Task.CHANGE_DETECTION:
        # Two dates.
        assert spec.dates == 2
    # Spectral indices name their formula.
    if spec.task is Task.SPECTRAL_INDEX:
        # Formula present and single output.
        assert spec.formula and spec.out_channels == 1


# Model ids split into family and variant.
def test_parse_model_id() -> None:
    # Explicit variant suffix.
    assert parse_model_id("ship_detector_mega") == ("ship_detector", Variant.MEGA)
    # A family without suffix implies the base variant.
    assert parse_model_id("super_resolution") == ("super_resolution", Variant.BASE)
    # Families whose name ends like a word are not split wrongly.
    assert parse_model_id("zonal_statistics_tiny") == ("zonal_statistics", Variant.TINY)


# Unknown names raise CatalogError.
def test_unknown_family() -> None:
    # The lookup fails with a helpful error.
    with pytest.raises(CatalogError):
        # Unknown family.
        get_spec("no_such_model")


# Variants grow monotonically in capacity.
def test_variants_grow() -> None:
    # Hyperparameters in size order.
    specs = [get_variant(v) for v in Variant]
    # Base widths increase.
    assert [s.base_channels for s in specs] == sorted(s.base_channels for s in specs)
    # Depths never decrease.
    assert [s.depth for s in specs] == sorted(s.depth for s in specs)


# The registry exposes all catalogue models with digests.
def test_registry_entries() -> None:
    # All models.
    entries = list_models()
    # 520 catalogue models.
    assert len(entries) == 520
    # Filtering by task and variant.
    detectors = list_models(task="detection", variant="tiny")
    # One tiny model per detection family.
    assert len(detectors) == EXPECTED_TASKS["detection"]
    # Lookup of a model id.
    entry = get_model("ship_detector_base")
    # The entry exists and describes the right model.
    assert entry is not None and entry.family == "ship_detector"
    # Published digests exist for every catalogue model.
    assert all(len(e.weights_digest) == 64 for e in entries)
    # Unknown models yield None.
    assert get_model("no_such_model") is None
    # Starter models need training; spectral indices do not.
    assert entry.requires_training and not get_model("ndvi_calculator_tiny").requires_training


# =============================================================================
# End of module tests/unit/test_zoo_catalog.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
