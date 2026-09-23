# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/e2e/test_workflows.py
# Title       : End-to-end workflows through the command line and REST API
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, click, rasterio and
#               PyTorch; the REST test also needs FastAPI and httpx
# =============================================================================
#
# Abstract
# --------
# Runs complete user workflows the way a user would:
#
#   - ship detection: train a tiny detector on synthetic data with
#     `unbihexium train`, then detect ships in a GeoTIFF with
#     `unbihexium predict` and read the GeoJSON result
#   - burn severity: compute NBR before and after a fire with
#     `unbihexium index`, then classify the difference with the index
#     library
#   - urban change: detect changes between two dates with
#     `unbihexium predict --second`
#   - REST: start the FastAPI application in process and request NDVI
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON output of the commands.
import json

# Numbers in the JSON response.
import re

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

# Root command group.
from unbihexium.cli import cli

# PyTorch is needed by the model commands.
torch = pytest.importorskip("torch")

# 10 m grid in UTM zone 35N.
TRANSFORM = Affine(10, 0, 500000, 0, -10, 7000000)


# Write a float32 GeoTIFF on the 10 m grid.
def write_geotiff(path: Path, data: np.ndarray) -> Path:
    # Open the file for writing.
    with rasterio.open(
        path,  # File.
        "w",  # Write mode.
        driver="GTiff",  # Format.
        height=data.shape[1],  # Rows.
        width=data.shape[2],  # Columns.
        count=data.shape[0],  # Bands.
        dtype="float32",  # Data type.
        crs="EPSG:32635",  # UTM zone 35N.
        transform=TRANSFORM,  # 10 m grid.
    ) as dst:  # Dataset handle.
        # Write the bands.
        dst.write(data.astype(np.float32))
    # Return the path.
    return path


# Train a detector, then detect ships in a GeoTIFF.
def test_ship_detection_workflow(tmp_path: Path, isolated_cache: Path) -> None:
    # Command runner.
    runner = CliRunner()
    # Output directory of the training run.
    run_dir = tmp_path / "run"
    # Short synthetic training of the tiny detector.
    args = [
        "train",  # Command.
        "ship_detector_tiny",  # Model.
        "--synthetic",  # Synthetic data.
        "16",  # Samples.
        "--epochs",  # Epochs.
        "1",  # One epoch.
        "--chip-size",  # Chip size.
        "64",  # Pixels.
        "--device",  # Device.
        "cpu",  # CPU.
        "--output",  # Output directory.
        str(run_dir),  # Path.
    ]  # End of the arguments.
    # Run the training.
    result = runner.invoke(cli, args)
    # Success.
    assert result.exit_code == 0, result.output
    # Scene to analyse.
    scene = write_geotiff(tmp_path / "scene.tif", np.random.default_rng(0).random((3, 96, 96)))
    # Detection output.
    out = tmp_path / "ships.geojson"
    # Detect with the trained checkpoint and a low threshold.
    args = ["predict", str(run_dir / "best.pt"), str(scene), str(out), "--threshold", "0.05"]
    # Run the prediction.
    result = runner.invoke(cli, args)
    # Success.
    assert result.exit_code == 0, result.output
    # Parse the GeoJSON.
    collection = json.loads(out.read_text())
    # Feature collection in the scene CRS.
    assert collection["type"] == "FeatureCollection" and collection["crs"] == "EPSG:32635"
    # Every box lies inside the scene footprint.
    for feature in collection["features"]:
        # Polygon corners.
        ring = np.asarray(feature["geometry"]["coordinates"][0])
        # Inside x from 500000 to 500960 and y from 6999040 to 7000000.
        assert ring[:, 0].min() >= 500000 - 1e-6 and ring[:, 1].max() <= 7000000 + 1e-6


# Burn severity from NBR before and after a fire.
def test_burn_severity_workflow(tmp_path: Path) -> None:
    # Command runner.
    runner = CliRunner()
    # Thirteen-band scenes; NBR uses band 8 (NIR) and band 13 (SWIR 2.2 um).
    before = np.full((13, 8, 8), 0.1)
    # Healthy vegetation: high NIR, low SWIR.
    before[7], before[12] = 0.5, 0.1
    # After the fire: lower NIR, higher SWIR.
    after = before.copy()
    # Burnt pixels.
    after[7], after[12] = 0.2, 0.3
    # NBR files.
    outputs = []
    # Compute NBR for both dates.
    for name, data in (("before", before), ("after", after)):
        # Scene file.
        scene = write_geotiff(tmp_path / f"{name}.tif", data)
        # Output file.
        out = tmp_path / f"nbr_{name}.tif"
        # Run the index command.
        result = runner.invoke(cli, ["index", "nbr", "-i", str(scene), "-o", str(out)])
        # Success.
        assert result.exit_code == 0, result.output
        # Keep the output.
        outputs.append(out)
    # Read both indices.
    nbr = [rasterio.open(p).read(1).astype(np.float64) for p in outputs]
    # Difference NBR, pre minus post.
    dnbr = nbr[0] - nbr[1]
    # (0.5 - 0.1) / 0.6 - (0.2 - 0.3) / 0.5 = 0.6667 + 0.2.
    np.testing.assert_allclose(dnbr, 0.4 / 0.6 + 0.2, rtol=1e-5)
    # Severity classes of Key and Benson (2006).
    from unbihexium.core.index import classify_burn_severity

    # dNBR 0.867 is high severity.
    classes = classify_burn_severity(dnbr)
    # Every pixel has the same class, which is not unburned.
    assert np.unique(classes).size == 1 and int(classes.flat[0]) > 1


# Detect changes between two dates of a city.
def test_urban_change_detection_workflow(tmp_path: Path, isolated_cache: Path) -> None:
    # Command runner.
    runner = CliRunner()
    # Seeded generator.
    rng = np.random.default_rng(1)
    # First date.
    first = write_geotiff(tmp_path / "t1.tif", rng.random((3, 48, 48)))
    # Second date.
    second = write_geotiff(tmp_path / "t2.tif", rng.random((3, 48, 48)))
    # Change map.
    out = tmp_path / "change.tif"
    # Run the change detector on both dates.
    args = ["predict", "change_detector_tiny", str(first), str(out), "--second", str(second)]
    # Run the command.
    result = runner.invoke(cli, args)
    # Success.
    assert result.exit_code == 0, result.output
    # Read the change map.
    with rasterio.open(out) as src:
        # Class labels.
        labels = src.read(1)
        # Grid of the inputs.
        assert src.transform == TRANSFORM
    # Labels are no change (0), change (1) or no data (255).
    assert set(np.unique(labels)) <= {0, 1, 255}


# Request NDVI from the REST service.
def test_rest_ndvi_workflow() -> None:
    # FastAPI is optional.
    pytest.importorskip("fastapi")
    # httpx drives the test client.
    pytest.importorskip("httpx")
    # Test client and application factory.
    from fastapi.testclient import TestClient  # In-process client.

    from unbihexium.serving import create_app  # Application factory.

    # Client of the application.
    client = TestClient(create_app())
    # Health check.
    assert client.get("/health").status_code == 200
    # Red 0.1 and near infrared 0.5 on a 2 x 2 image.
    image = [[[0.1, 0.1], [0.1, 0.1]], [[0.5, 0.5], [0.5, 0.5]]]
    # Request the exact NDVI model.
    response = client.post("/predict/ndvi_calculator_tiny", json={"image": image})
    # Success.
    assert response.status_code == 200, response.text
    # Means reported anywhere in the response document.
    means = [float(m) for m in re.findall(r'"mean":\s*([0-9.eE+-]+)', response.text)]
    # The mean NDVI is (0.5 - 0.1) / (0.5 + 0.1).
    assert means and means[0] == pytest.approx(0.4 / 0.6, rel=1e-6)


# =============================================================================
# End of module tests/e2e/test_workflows.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
