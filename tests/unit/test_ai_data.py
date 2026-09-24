# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_ai_data.py
# Title       : Tests of dataset folders and synthetic datasets
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and rasterio; no
#               PyTorch needed
# =============================================================================
#
# Abstract
# --------
# Writes small datasets in the documented folder layout to a temporary
# directory (NumPy and GeoTIFF images, class masks, pixel and GeoJSON boxes,
# JSON and CSV scene targets, super-resolution targets) and checks that
# FolderDataset reads them, applies dataset.yaml settings, reads windows and
# reports layout errors. Also checks that SyntheticDataset produces valid
# samples for every trainable task.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# JSON label files.
import json

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

# Datasets under test.
from unbihexium.ai.data import DatasetError, FolderDataset, SyntheticDataset, raster_shape

# Configurations of catalogue models.
from unbihexium.zoo import BuildConfig, get_spec, list_specs


# Configuration of a catalogue family in the tiny variant.
def config_of(family: str) -> BuildConfig:
    # Catalogue configuration.
    return BuildConfig.from_spec(get_spec(family), "tiny")


# Write a float32 GeoTIFF on a 10 m UTM grid.
def write_tif(path: Path, data: np.ndarray) -> None:
    # Create the parent directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file for writing.
    with rasterio.open(
        path,  # File.
        "w",  # Write mode.
        driver="GTiff",  # Format.
        height=data.shape[1],  # Rows.
        width=data.shape[2],  # Columns.
        count=data.shape[0],  # Bands.
        dtype=str(data.dtype),  # Data type.
        crs="EPSG:32635",  # UTM zone 35N.
        transform=Affine(10, 0, 500000, 0, -10, 7000000),  # Pixel size and upper-left corner.
    ) as dst:  # Dataset handle.
        # Write the bands.
        dst.write(data)


# Segmentation data from NumPy files, with a label map.
def test_segmentation_folder(tmp_path: Path) -> None:
    # Water detector: four bands, two classes.
    config = config_of("water_surface_detector")
    # Image directory.
    (tmp_path / "train" / "images").mkdir(parents=True)
    # Label directory.
    (tmp_path / "train" / "labels").mkdir(parents=True)
    # Image with six bands; dataset.yaml selects four of them.
    np.save(tmp_path / "train" / "images" / "a.npy", np.random.rand(6, 20, 30).astype(np.float32))
    # Raw mask values 0, 10 and 99.
    np.save(tmp_path / "train" / "labels" / "a.npy", np.array([[0, 10, 99] * 10] * 20))
    # Settings: bands, label map and scale.
    settings = "band_indices: [0, 1, 2, 5]\nlabel_map: {0: 0, 10: 1}\nscale: 2.0\n"
    # Write dataset.yaml.
    (tmp_path / "dataset.yaml").write_text(settings, encoding="utf-8")
    # Open the split.
    ds = FolderDataset(tmp_path, "train", config)
    # One sample.
    assert len(ds) == 1 and ds.shape(0) == (4, 20, 30)
    # Load it.
    sample = ds[0]
    # Selected bands.
    assert sample.image.shape == (4, 20, 30)
    # Unmapped values are ignored.
    assert sorted(np.unique(sample.mask).tolist()) == [0, 1, 255]
    # Windowed reading.
    window = ds.load(0, (5, 3, 10, 12))
    # Window size.
    assert window.image.shape == (4, 10, 12) and window.mask.shape == (10, 12)


# Detection data from a GeoTIFF with GeoJSON boxes in map coordinates.
def test_detection_geojson(tmp_path: Path) -> None:
    # Ship detector: three bands.
    config = config_of("ship_detector")
    # GeoTIFF image.
    image = np.random.rand(3, 40, 50).astype(np.float32)
    # Write it.
    write_tif(tmp_path / "train" / "images" / "s1.tif", image)
    # Box from pixel (5, 4) to (15, 10): x from 500050 to 500150, y from 6999960 to 6999900.
    ring = [
        [500050, 6999960],  # Upper-left corner.
        [500150, 6999960],  # Upper-right corner.
        [500150, 6999900],  # Lower-right corner.
        [500050, 6999900],  # Lower-left corner.
        [500050, 6999960],  # Closing corner.
    ]
    # GeoJSON label file with the class name.
    collection = {
        "type": "FeatureCollection",  # Object type.
        "features": [  # Feature list.
            {  # Feature of the box.
                "type": "Feature",  # Object type.
                "properties": {"class": "ship"},  # Class name.
                "geometry": {"type": "Polygon", "coordinates": [ring]},  # Box polygon.
            }  # End of the feature.
        ],  # End of the features.
    }  # End of the collection.
    # Write the label file.
    (tmp_path / "train" / "labels").mkdir(parents=True)
    # JSON text.
    text = json.dumps(collection)
    # Write it.
    (tmp_path / "train" / "labels" / "s1.geojson").write_text(text, encoding="utf-8")
    # Load the sample.
    sample = FolderDataset(tmp_path, "train", config)[0]
    # Box in pixel coordinates.
    assert np.allclose(sample.boxes, [[5, 4, 15, 10]])
    # Class index of "ship".
    assert sample.labels.tolist() == [0]
    # Header shape of the GeoTIFF.
    assert raster_shape(tmp_path / "train" / "images" / "s1.tif") == (3, 40, 50)


# Detection windows shift and drop boxes.
def test_detection_window(tmp_path: Path) -> None:
    # Ship detector.
    config = config_of("ship_detector")
    # Image.
    (tmp_path / "train" / "images").mkdir(parents=True)
    # Random image.
    np.save(tmp_path / "train" / "images" / "a.npy", np.random.rand(3, 64, 64).astype(np.float32))
    # Labels.
    (tmp_path / "train" / "labels").mkdir(parents=True)
    # Two pixel boxes.
    labels = {"boxes": [[2, 2, 10, 10], [40, 40, 50, 50]], "labels": [0, "ship"]}
    # Write the label file.
    (tmp_path / "train" / "labels" / "a.json").write_text(json.dumps(labels), encoding="utf-8")
    # Window that contains only the second box.
    sample = FolderDataset(tmp_path, "train", config).load(0, (32, 32, 32, 32))
    # The second box in window coordinates.
    assert sample.boxes.tolist() == [[8, 8, 18, 18]]


# Scene targets from a CSV table or JSON files.
def test_scene_targets(tmp_path: Path) -> None:
    # Yield predictor: ten bands, one output named "yield".
    config = config_of("yield_predictor")
    # Two images.
    (tmp_path / "train" / "images").mkdir(parents=True)
    # Save the images.
    for name in ("f1", "f2"):
        # Random field image.
        np.save(tmp_path / "train" / "images" / f"{name}.npy", np.random.rand(10, 16, 16))
    # CSV table for f1.
    (tmp_path / "train" / "targets.csv").write_text("id,yield\nf1,5.5\n", encoding="utf-8")
    # JSON file for f2.
    (tmp_path / "train" / "labels").mkdir(parents=True)
    # Values by name.
    values = json.dumps({"values": {"yield": 7.0}})
    # Write them.
    (tmp_path / "train" / "labels" / "f2.json").write_text(values, encoding="utf-8")
    # Open the split.
    ds = FolderDataset(tmp_path, "train", config)
    # Value from the table.
    assert ds[0].vector.tolist() == [5.5]
    # Value from the JSON file.
    assert ds[1].vector.tolist() == [7.0]


# Super-resolution targets live on a finer grid.
def test_super_resolution_window(tmp_path: Path) -> None:
    # Catalogue super-resolution model with scale 4.
    config = config_of("super_resolution")
    # Low-resolution image.
    (tmp_path / "train" / "images").mkdir(parents=True)
    # Save it.
    np.save(tmp_path / "train" / "images" / "a.npy", np.random.rand(3, 16, 16))
    # High-resolution target.
    (tmp_path / "train" / "labels").mkdir(parents=True)
    # Save it.
    np.save(tmp_path / "train" / "labels" / "a.npy", np.random.rand(3, 64, 64))
    # Window of 8 x 8 low-resolution pixels.
    sample = FolderDataset(tmp_path, "train", config).load(0, (4, 4, 8, 8))
    # Target window of 32 x 32 pixels.
    assert sample.values.shape == (3, 32, 32) and sample.scale == 4


# Layout problems are reported clearly.
def test_layout_errors(tmp_path: Path) -> None:
    # Ship detector.
    config = config_of("ship_detector")
    # Missing split.
    with pytest.raises(DatasetError, match="does not exist"):
        # Open a missing split.
        FolderDataset(tmp_path, "train", config)
    # Image with the wrong band count and no label.
    (tmp_path / "train" / "images").mkdir(parents=True)
    # Five bands instead of three.
    np.save(tmp_path / "train" / "images" / "a.npy", np.zeros((5, 8, 8)))
    # Open the split.
    ds = FolderDataset(tmp_path, "train", config)
    # Band count error with a hint.
    with pytest.raises(DatasetError, match="band_indices"):
        # Load the sample.
        ds[0]


# Synthetic samples have the right layout for every trainable family.
@pytest.mark.parametrize(
    "spec", [s for s in list_specs() if s.task.is_trainable], ids=lambda s: s.family
)
def test_synthetic_samples(spec) -> None:
    # Tiny configuration.
    config = BuildConfig.from_spec(spec, "tiny")
    # Two samples of 32 x 32 pixels.
    ds = SyntheticDataset(config, length=2, size=32)
    # First sample.
    sample = ds[0]
    # Input bands.
    assert sample.image.shape == (config.in_channels, 32, 32)
    # Finite values.
    assert np.isfinite(sample.image).all()
    # Deterministic generation.
    assert np.array_equal(ds[0].image, sample.image)
    # Target present for the task.
    assert any(t is not None for t in (sample.mask, sample.values, sample.boxes, sample.vector))


# Spectral index models cannot be trained.
def test_synthetic_rejects_formulas() -> None:
    # NDVI is a fixed formula.
    with pytest.raises(ValueError, match="not trainable"):
        # Create the dataset.
        SyntheticDataset(config_of("ndvi_calculator"))


# =============================================================================
# End of module tests/unit/test_ai_data.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
