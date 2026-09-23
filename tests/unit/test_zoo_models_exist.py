# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests to verify all models exist in inventory and have required files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def get_inventory_path() -> Path:
    """Get path to inventory.yaml."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "model_zoo" / "inventory.yaml"


def get_manifests_dir() -> Path:
    """Get path to manifests directory."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "model_zoo" / "manifests"


def get_cards_dir() -> Path:
    """Get path to model cards directory."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "model_zoo" / "cards"


def load_inventory() -> dict:
    """Load inventory.yaml."""
    inventory_path = get_inventory_path()
    if not inventory_path.exists():
        pytest.skip("inventory.yaml not found")
    with open(inventory_path) as f:
        return yaml.safe_load(f)


class TestModelInventory:
    """Tests for model inventory completeness."""

    def test_inventory_exists(self) -> None:
        """Test that inventory.yaml exists."""
        assert get_inventory_path().exists()

    def test_inventory_has_models(self) -> None:
        """Test that inventory contains models."""
        inventory = load_inventory()
        assert "models" in inventory
        assert len(inventory["models"]) > 0

    def test_model_ids_valid(self) -> None:
        """Test that all model IDs follow naming convention."""
        inventory = load_inventory()
        for model in inventory["models"]:
            model_id = model["model_id"]
            assert model_id.startswith("ubx-"), f"Invalid prefix: {model_id}"
            assert "-" in model_id, f"Missing version separator: {model_id}"

    def test_required_fields(self) -> None:
        """Test that all models have required fields."""
        inventory = load_inventory()
        required_fields = ["model_id", "name", "task", "architecture", "license"]

        for model in inventory["models"]:
            for field in required_fields:
                assert field in model, f"Missing {field} in {model.get('model_id', 'unknown')}"


class TestModelManifests:
    """Tests for model manifest files."""

    def test_manifests_directory_exists(self) -> None:
        """Test that manifests directory exists."""
        manifests_dir = get_manifests_dir()
        if not manifests_dir.exists():
            pytest.skip("manifests directory not yet created")
        assert manifests_dir.is_dir()


class TestModelCards:
    """Tests for model card files."""

    def test_cards_directory_exists(self) -> None:
        """Test that cards directory exists."""
        cards_dir = get_cards_dir()
        if not cards_dir.exists():
            pytest.skip("cards directory not yet created")
        assert cards_dir.is_dir()


class TestCapabilityMapping:
    """Tests for capability to model mapping."""

    def test_mapping_file_exists(self) -> None:
        """Test that capability mapping file exists."""
        repo_root = Path(__file__).parent.parent.parent
        mapping_path = repo_root / "model_zoo" / "capability_to_models.yaml"
        assert mapping_path.exists()

    def test_mapping_has_entries(self) -> None:
        """Test that mapping contains entries."""
        repo_root = Path(__file__).parent.parent.parent
        mapping_path = repo_root / "model_zoo" / "capability_to_models.yaml"

        with open(mapping_path) as f:
            mapping = yaml.safe_load(f)

        assert "mappings" in mapping
        assert len(mapping["mappings"]) > 0

    def test_all_capabilities_have_models(self) -> None:
        """Test that all capabilities map to valid model IDs."""
        repo_root = Path(__file__).parent.parent.parent
        mapping_path = repo_root / "model_zoo" / "capability_to_models.yaml"
        inventory_path = repo_root / "model_zoo" / "inventory.yaml"

        with open(mapping_path) as f:
            mapping = yaml.safe_load(f)

        with open(inventory_path) as f:
            inventory = yaml.safe_load(f)

        model_ids = {m["model_id"] for m in inventory["models"]}

        for cap_id, cap_info in mapping["mappings"].items():
            model_id = cap_info["primary_model_id"]
            assert model_id in model_ids, f"Capability {cap_id} references unknown model: {model_id}"
