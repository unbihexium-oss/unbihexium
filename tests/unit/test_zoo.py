"""Tests for model zoo."""

from __future__ import annotations

import pytest


class TestModelZoo:
    """Tests for model zoo functionality."""

    def test_list_models(self) -> None:
        from unbihexium.zoo import list_models

        models = list_models()
        assert len(models) >= 5  # Built-in tiny models

    def test_get_model(self) -> None:
        from unbihexium.zoo import get_model

        model = get_model("ship_detector_tiny")
        assert model is not None
        assert model.model_id == "ship_detector_tiny"

    def test_get_nonexistent_model(self) -> None:
        from unbihexium.zoo import get_model

        model = get_model("nonexistent_model")
        assert model is None

    def test_filter_by_task(self) -> None:
        from unbihexium.zoo import list_models

        detection_models = list_models(task="detection")
        assert all(m.config.task.value == "detection" for m in detection_models)


class TestModelDownloader:
    """Tests for model downloading."""

    def test_download_creates_placeholder(self, tmp_path) -> None:
        from unbihexium.zoo import download_model

        path = download_model("ship_detector_tiny", cache_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".pt"
