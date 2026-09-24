# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_cli.py
# Title       : Tests of the command line interface
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and click; the model
#               commands also need PyTorch
# =============================================================================
#
# Abstract
# --------
# Invokes the `unbihexium` command with click's test runner: version, help,
# the model zoo commands, predict on a GeoTIFF, train on synthetic data,
# evaluate, pipelines and spectral indices. Model commands use the tiny
# variants and a temporary cache directory.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON output.
import json

# Represent file paths.
from pathlib import Path

# Arrays.
import numpy as np

# Test framework.
import pytest

# GeoTIFF writing.
import rasterio

# Command line test runner.
from click.testing import CliRunner

# Affine transform type.
from rasterio.transform import Affine

# Root command group under test.
from unbihexium.cli import cli


# Test runner.
@pytest.fixture
def runner() -> CliRunner:
    # Default runner.
    return CliRunner()


# Temporary model cache.
@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Cache root inside the test directory.
    root = tmp_path / "cache"
    # Point the store at it.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(root))
    # Return the root.
    return root


# Write a georeferenced GeoTIFF.
def geotiff(path: Path, bands: int, size: int = 40) -> Path:
    # Seeded random data.
    data = np.random.default_rng(0).random((bands, size, size), dtype=np.float32)
    # Open the file for writing.
    with rasterio.open(
        path,  # File.
        "w",  # Write mode.
        driver="GTiff",  # Format.
        height=size,  # Rows.
        width=size,  # Columns.
        count=bands,  # Bands.
        dtype="float32",  # Data type.
        crs="EPSG:32635",  # UTM zone 35N.
        transform=Affine(10, 0, 500000, 0, -10, 7000000),  # 10 m grid.
    ) as dst:  # Dataset handle.
        # Write the bands.
        dst.write(data)
    # Return the path.
    return path


# Version and help.
def test_version_and_help(runner: CliRunner) -> None:
    # Version option.
    result = runner.invoke(cli, ["--version"])
    # Success.
    assert result.exit_code == 0 and "unbihexium" in result.output
    # Help text.
    result = runner.invoke(cli, ["--help"])
    # Success with the description and the commands.
    assert result.exit_code == 0 and "Unbihexium" in result.output and "train" in result.output


# Model zoo listing and information.
def test_zoo_list_and_info(runner: CliRunner) -> None:
    # JSON listing of the tiny detectors.
    args = ["zoo", "list", "--task", "detection", "--variant", "tiny", "--json"]
    # Run the command.
    result = runner.invoke(cli, args)
    # Success.
    assert result.exit_code == 0
    # Nineteen detectors.
    assert len(json.loads(result.output)) == 19
    # Information about one model.
    result = runner.invoke(cli, ["zoo", "info", "ship_detector_tiny"])
    # Success with the digest.
    assert result.exit_code == 0 and "weights_digest" in result.output
    # Unknown models fail.
    assert runner.invoke(cli, ["zoo", "info", "no_such_model"]).exit_code == 1


# Build, locate, verify and clear a cached model.
def test_zoo_store(runner: CliRunner) -> None:
    # PyTorch builds the model.
    pytest.importorskip("torch")
    # Build the NDVI model.
    assert runner.invoke(cli, ["zoo", "build", "ndvi_calculator_tiny"]).exit_code == 0
    # Its checkpoint path.
    result = runner.invoke(cli, ["zoo", "where", "ndvi_calculator_tiny"])
    # Path of model.pt.
    assert result.exit_code == 0 and result.output.strip().endswith("model.pt")
    # Verification succeeds.
    assert runner.invoke(cli, ["zoo", "verify", "ndvi_calculator_tiny"]).exit_code == 0
    # Clear everything without asking.
    result = runner.invoke(cli, ["zoo", "clear", "--yes"])
    # One model removed.
    assert result.exit_code == 0 and "Removed 1" in result.output


# Predict writes GeoJSON for detectors and GeoTIFF for segmenters.
def test_predict(runner: CliRunner, tmp_path: Path) -> None:
    # PyTorch runs the models.
    pytest.importorskip("torch")
    # Three-band image.
    image = geotiff(tmp_path / "rgb.tif", 3)
    # Detection output.
    out = tmp_path / "ships.geojson"
    # Run the detector.
    result = runner.invoke(cli, ["predict", "ship_detector_tiny", str(image), str(out)])
    # Success and a feature collection.
    assert result.exit_code == 0 and json.loads(out.read_text())["type"] == "FeatureCollection"
    # Change detection with two dates.
    out = tmp_path / "change.tif"
    # Run the change detector.
    args = ["predict", "change_detector_tiny", str(image), str(out), "--second", str(image)]
    # Success.
    assert runner.invoke(cli, args).exit_code == 0
    # The change map keeps the grid.
    with rasterio.open(out) as src:
        # Same size and transform.
        assert src.shape == (40, 40) and src.transform.a == 10
    # Wrong band count fails with exit status 1.
    args = ["predict", "lulc_classifier_tiny", str(image), str(tmp_path / "x.tif")]
    # Run the command.
    result = runner.invoke(cli, args)
    # Failure with the band message.
    assert result.exit_code == 1 and "expects 10 bands" in result.output


# Train on synthetic data and evaluate is reachable.
def test_train_synthetic(runner: CliRunner, tmp_path: Path) -> None:
    # PyTorch trains the model.
    pytest.importorskip("torch")
    # Output directory.
    out = tmp_path / "run"
    # Short synthetic training.
    args = [
        "train",  # Command.
        "yield_predictor_tiny",  # Model.
        "--synthetic",  # Synthetic data.
        "8",  # Samples.
        "--epochs",  # Epochs.
        "1",  # One epoch.
        "--chip-size",  # Chip size.
        "32",  # Pixels.
        "--device",  # Device.
        "cpu",  # CPU.
        "--output",  # Output directory.
        str(out),  # Path.
    ]  # End of the arguments.
    # Run the command.
    result = runner.invoke(cli, args)
    # Success.
    assert result.exit_code == 0, result.output
    # Best and last checkpoints and the history.
    assert (
        (out / "best.pt").is_file()  # Best checkpoint.
        and (out / "last.pt").is_file()  # Last checkpoint.
        and (out / "history.json").is_file()  # History file.
    )
    # Training needs data.
    assert runner.invoke(cli, ["train", "yield_predictor_tiny"]).exit_code == 1


# Spectral index command writes a georeferenced index.
def test_index(runner: CliRunner, tmp_path: Path) -> None:
    # Thirteen-band Sentinel-2 image.
    image = geotiff(tmp_path / "s2.tif", 13)
    # Output file.
    out = tmp_path / "ndvi.tif"
    # Compute NDVI from bands 4 and 8.
    result = runner.invoke(cli, ["index", "ndvi", "-i", str(image), "-o", str(out)])
    # Success.
    assert result.exit_code == 0
    # Read the input bands and the index.
    with rasterio.open(image) as src, rasterio.open(out) as dst:
        # Red and near infrared.
        red, nir = src.read(4).astype(float), src.read(8).astype(float)
        # Index values.
        ndvi = dst.read(1)
    # The formula.
    assert np.allclose(ndvi, (nir - red) / (nir + red), atol=1e-5)
    # Bands beyond the raster fail.
    result = runner.invoke(cli, ["index", "ndvi", "-i", str(image), "-o", str(out), "--nir", "20"])
    # Failure.
    assert result.exit_code == 1


# Pipelines are listed after the AI package registers them.
def test_pipeline_list(runner: CliRunner) -> None:
    # List the pipelines.
    result = runner.invoke(cli, ["pipeline", "list"])
    # The task pipelines are registered.
    assert result.exit_code == 0 and "ship_detection" in result.output


# `serve` starts uvicorn with the configured and the given address.
def test_serve(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The serving extra is optional.
    uvicorn = pytest.importorskip("uvicorn")
    # Settings cache of the library.
    from unbihexium.config import reset_settings

    # Arguments of the uvicorn.run calls.
    calls: list[dict[str, object]] = []
    # Record the call instead of starting a server.
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    # Configuration file with a port and a log level.
    config = tmp_path / "serve.yaml"
    # Serving section and top-level level.
    config.write_text("log_level: INFO\nserving:\n  port: 9100\n", encoding="utf-8")
    # `serve --config` sets UNBIHEXIUM_CONFIG; register it so that it is restored.
    monkeypatch.setenv("UNBIHEXIUM_CONFIG", str(config))
    # Forget cached settings.
    reset_settings()
    # Values from the file.
    result = runner.invoke(cli, ["serve", "--config", str(config)])
    # One call on the configured port and the default host.
    assert result.exit_code == 0, result.output
    # Port and host of the configuration.
    assert calls[-1]["port"] == 9100 and calls[-1]["host"] == "127.0.0.1"
    # Log level of the configuration.
    assert calls[-1]["log_level"] == "info"
    # Command line options win.
    result = runner.invoke(cli, ["serve", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers"])
    # Second call with the given address.
    assert calls[-1]["host"] == "0.0.0.0" and calls[-1]["port"] == 8001
    # Forwarded headers are trusted on request.
    assert calls[-1]["proxy_headers"] is True
    # Leave no settings behind for other tests.
    reset_settings()


# =============================================================================
# End of module tests/unit/test_cli.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
