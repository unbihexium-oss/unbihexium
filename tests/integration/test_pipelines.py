# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/integration/test_pipelines.py
# Title       : Integration tests across packages
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, rasterio and PyTorch
# =============================================================================
#
# Abstract
# --------
# Exercises several packages together on files in a temporary directory:
#
#   - registered pipelines read GeoTIFFs and run the task APIs
#   - the model store builds, verifies and reloads a model, and the loaded
#     model gives the same prediction as a freshly built one
#   - a GeoTIFF is read, an index computed, and the result written as a
#     georeferenced GeoTIFF that keeps the grid of the input
#   - training on a dataset folder produces a checkpoint whose ONNX export
#     gives the same prediction as PyTorch
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Represent file paths.
from pathlib import Path

# Arrays.
import numpy as np

# Test framework.
import pytest

# GeoTIFF writing.
import rasterio

# Affine transform type.
from rasterio.transform import Affine

# PyTorch is needed to build and run models.
torch = pytest.importorskip("torch")

# Tiled inference.
from unbihexium.ai.inference import Predictor  # noqa: E402 - imported after the skip check

# Model construction.
from unbihexium.ai.models import build_model  # noqa: E402 - imported after the skip check

# Raster container.
from unbihexium.core.raster import Raster  # noqa: E402 - imported after the skip check

# Pipeline registry.
from unbihexium.registry.pipelines import PipelineRegistry  # noqa: E402

# Model store.
from unbihexium.zoo import ensure_model, load_model, verify_model  # noqa: E402

# 10 m grid in UTM zone 35N used by the test files.
TRANSFORM = Affine(10, 0, 500000, 0, -10, 7000000)


# Write a seeded float32 GeoTIFF with the given number of bands.
def write_geotiff(path: Path, bands: int, size: int = 64, seed: int = 0) -> Path:
    # Seeded reflectances.
    data = np.random.default_rng(seed).random((bands, size, size), dtype=np.float32)
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
        transform=TRANSFORM,  # 10 m grid.
    ) as dst:  # Dataset handle.
        # Write the bands.
        dst.write(data)
    # Return the path.
    return path


# Registered pipelines run the task APIs on files.
def test_registered_pipelines(tmp_path: Path) -> None:
    # The AI package registers its pipelines on import.
    import unbihexium.ai  # Registers the task pipelines.

    # Water detection needs four bands.
    image = write_geotiff(tmp_path / "scene.tif", 4)
    # Pipeline with the tiny model.
    pipeline = PipelineRegistry.create("water_detection", variant="tiny")
    # The pipeline exists.
    assert pipeline is not None
    # Run it on the file.
    run = pipeline.run({"input": str(image)})
    # The run completed.
    assert run.status.value == "completed"
    # The class map keeps the size of the input.
    assert pipeline.last_result.mask.shape == (64, 64)
    # Change detection reads two files.
    change = PipelineRegistry.create("change_detection", variant="tiny")
    # First date.
    first = write_geotiff(tmp_path / "t1.tif", 3, seed=1)
    # Second date.
    second = write_geotiff(tmp_path / "t2.tif", 3, seed=2)
    # Run on both dates.
    change.run({"input1": str(first), "input2": str(second)})
    # The change map keeps the grid of the inputs.
    assert change.last_result.transform[:6] == tuple(TRANSFORM)[:6]


# The store builds, verifies and reloads a model with identical weights.
def test_store_round_trip(isolated_cache: Path) -> None:
    # Build and cache the tiny detector.
    directory = ensure_model("ship_detector_tiny")
    # The files verify against the published digest.
    assert verify_model("ship_detector_tiny")
    # Model loaded from the cached checkpoint.
    cached = load_model(directory / "model.pt")
    # Freshly built model.
    fresh = build_model("ship_detector_tiny")
    # Fixed input.
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    # Both models give the same output.
    with torch.no_grad():
        # Compare the outputs exactly.
        assert torch.equal(cached(x), fresh(x))


# Read a GeoTIFF, compute NDVI with the exact model and keep the grid.
def test_read_process_write_geotiff(tmp_path: Path) -> None:
    # Two-band image: red and near infrared.
    image = write_geotiff(tmp_path / "rn.tif", 2)
    # Read it.
    raster = Raster.from_file(image)
    # NDVI through the spectral index model.
    ndvi = Predictor("ndvi_calculator").dense(raster.data)
    # Reference formula.
    red, nir = raster.data[0].astype(np.float64), raster.data[1].astype(np.float64)
    # Same values.
    np.testing.assert_allclose(ndvi[0], (nir - red) / (nir + red), rtol=1e-5)
    # Write the index on the input grid.
    out = Raster.from_array(ndvi, crs=raster.metadata.crs, transform=raster.metadata.transform)
    # Written file.
    path = out.to_file(tmp_path / "ndvi.tif")
    # Read the header back.
    with rasterio.open(path) as src:
        # Grid and CRS of the input.
        assert src.transform == TRANSFORM and src.crs.to_epsg() == 32635


# Train on a folder, export to ONNX and compare the backends.
def test_train_export_predict(tmp_path: Path) -> None:
    # ONNX Runtime and onnx are optional.
    pytest.importorskip("onnxruntime")
    # onnx writes the export metadata.
    pytest.importorskip("onnx")
    # Training, synthetic data and export.
    from unbihexium.ai.data import SyntheticDataset  # Synthetic samples.
    from unbihexium.ai.training import TrainConfig, train  # Training.
    from unbihexium.zoo.export import export_onnx  # ONNX export.
    from unbihexium.zoo.store import load_model as load  # Checkpoint loading.

    # Dataset root.
    root = tmp_path / "data"
    # Synthetic water samples.
    source = SyntheticDataset(build_model("water_surface_detector_tiny").config, 4, 64)
    # Train split directories.
    (root / "train" / "images").mkdir(parents=True)
    # Label directory.
    (root / "train" / "labels").mkdir(parents=True)
    # Write the samples as NumPy files.
    for i in range(len(source)):
        # Sample.
        sample = source[i]
        # Image.
        np.save(root / "train" / "images" / f"s{i}.npy", sample.image)
        # Mask.
        np.save(root / "train" / "labels" / f"s{i}.npy", sample.mask)
    # Short training run on the CPU.
    config = TrainConfig(
        epochs=1,  # One epoch.
        batch_size=2,  # Small batches.
        chip_size=64,  # Chip size.
        device="cpu",  # CPU.
        output_dir=str(tmp_path / "run"),  # Output directory.
        verbose=False,  # Quiet.
    )  # End of the configuration.
    # Train.
    result = train("water_surface_detector_tiny", root, config)
    # Export the best checkpoint.
    onnx_path = export_onnx(load(result.best_checkpoint), tmp_path / "water.onnx")
    # Input image.
    image = source[0].image
    # PyTorch prediction with the stored normalisation.
    expected = Predictor(result.best_checkpoint).dense(image)
    # ONNX Runtime prediction.
    actual = Predictor(onnx_path).dense(image)
    # Same probabilities.
    np.testing.assert_allclose(actual, expected, atol=1e-4)


# =============================================================================
# End of module tests/integration/test_pipelines.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
