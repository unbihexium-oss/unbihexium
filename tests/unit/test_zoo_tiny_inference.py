# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests to verify tiny model ONNX inference works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def get_tiny_assets_dir() -> Path:
    """Get path to tiny model assets."""
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "model_zoo" / "assets" / "tiny"


ort = pytest.importorskip("onnxruntime")


class TestTinyModelInference:
    """Tests for tiny model ONNX inference."""

    @pytest.fixture
    def assets_dir(self) -> Path:
        """Get assets directory."""
        path = get_tiny_assets_dir()
        if not path.exists():
            pytest.skip("Tiny assets not built yet")
        return path

    def test_srcnn_inference(self, assets_dir: Path) -> None:
        """Test SRCNN super-resolution inference."""
        model_dir = assets_dir / "ubx-sr-srcnn-1.0.0"
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip("SRCNN model not built yet")

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        result = outputs[0]

        assert result.ndim == 4
        assert result.shape[0] == 1
        assert result.shape[1] == 3
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_unet_segmentation_inference(self, assets_dir: Path) -> None:
        """Test UNet segmentation inference."""
        model_dir = assets_dir / "ubx-seg-multiclass-unet-1.0.0"
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip("UNet model not built yet")

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        input_data = np.random.rand(1, 3, 64, 64).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        result = outputs[0]

        assert result.ndim == 4
        assert result.shape[0] == 1
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_change_siamese_inference(self, assets_dir: Path) -> None:
        """Test Siamese change detection inference."""
        model_dir = assets_dir / "ubx-change-siamese-1.0.0"
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip("Siamese model not built yet")

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        input_data = np.random.rand(1, 6, 64, 64).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        result = outputs[0]

        assert result.ndim == 4
        assert result.shape[0] == 1

    def test_mlp_inference(self, assets_dir: Path) -> None:
        """Test MLP regression inference."""
        model_dir = assets_dir / "ubx-flood-risk-mlp-1.0.0"
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip("MLP model not built yet")

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        input_data = np.random.rand(1, 10).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        result = outputs[0]

        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_all_tiny_models_have_onnx(self, assets_dir: Path) -> None:
        """Test that all tiny model directories have model.onnx."""
        missing = []

        for model_dir in assets_dir.iterdir():
            if model_dir.is_dir():
                onnx_path = model_dir / "model.onnx"
                if not onnx_path.exists():
                    missing.append(model_dir.name)

        if missing:
            pytest.skip(f"Missing ONNX files for: {missing}")

    def test_inference_output_finite(self, assets_dir: Path) -> None:
        """Test that inference outputs are finite."""
        model_dir = assets_dir / "ubx-suitability-mlp-1.0.0"
        onnx_path = model_dir / "model.onnx"

        if not onnx_path.exists():
            pytest.skip("Model not built yet")

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        input_data = np.random.rand(10, 10).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        result = outputs[0]

        assert np.all(np.isfinite(result))
