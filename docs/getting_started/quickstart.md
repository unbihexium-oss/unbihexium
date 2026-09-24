<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/getting_started/quickstart.md
Title       : Quick Start
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Quick Start

| Field | Value |
| --- | --- |
| Document | UBX-DOC-GS-QUICKSTART |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document is a guided first session with Unbihexium. Starting from an empty directory, it creates a small georeferenced test scene, reads and writes GeoTIFF and Cloud Optimized GeoTIFF files, computes spectral indices, converts a thresholded index into polygons, builds and runs a tiny model of the model zoo, trains it briefly on synthetic data, and repeats the same steps with the `unbihexium` command. It is written for new users who have completed [installation.md](installation.md) and want to see the main parts of the library working together. Every code block was executed in the order given against the main branch; outputs shown are the outputs of that run. The document does not explain every option; the complete interfaces are in [docs/reference/api.md](../reference/api.md) and [docs/reference/cli.md](../reference/cli.md).

## Contents

- [1. Before you start](#1-before-you-start)
- [2. Checking the installation](#2-checking-the-installation)
- [3. Creating a georeferenced test scene](#3-creating-a-georeferenced-test-scene)
- [4. Computing spectral indices](#4-computing-spectral-indices)
- [5. Working with the Raster class](#5-working-with-the-raster-class)
- [6. From a raster to polygons](#6-from-a-raster-to-polygons)
- [7. Building and running a tiny model](#7-building-and-running-a-tiny-model)
- [8. Training briefly and predicting again](#8-training-briefly-and-predicting-again)
- [9. An exact spectral index model](#9-an-exact-spectral-index-model)
- [10. The same session on the command line](#10-the-same-session-on-the-command-line)
- [11. Next steps](#11-next-steps)
- [References](#references)

## 1. Before you start

### 1.1 Requirements

Sections 2 to 6 need only the core installation. Sections 7 to 10 build and train models and therefore need the `torch` extra; Section 10.4 also uses the `onnx` extra. Because the main branch differs from the release 1.0.1 on PyPI (see [installation.md](installation.md#3-choosing-an-installation-method)), install from a clone:

```bash
python -m pip install -e ".[torch,onnx]"
```

### 1.2 Working directory and model store

Run the session in an empty directory. Every file is created there, except the models built with `unbihexium zoo build`, which are written to the model store in the directory named by the environment variable `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`). To keep the session self-contained, point it to the working directory:

```bash
mkdir unbihexium-quickstart && cd unbihexium-quickstart
export UNBIHEXIUM_CACHE="$PWD/cache"      # Windows PowerShell: $env:UNBIHEXIUM_CACHE = "$PWD\cache"
```

The Python blocks are separate scripts; each can be saved to a file and run with `python`, or pasted into any interactive Python session started in the working directory.

## 2. Checking the installation

```python
import unbihexium

print(unbihexium.__version__)
```

```text
1.0.1
```

Importing `unbihexium` loads only the version. Each subpackage (`unbihexium.io`, `unbihexium.indices` and so on) is imported when it is needed, and importing `unbihexium.ai` does not import PyTorch.

## 3. Creating a georeferenced test scene

The session uses a synthetic 128 x 128 pixel scene with four bands (blue, green, red, near infrared) of surface reflectance: bare soil on the left, vegetation on the right half and a water body in the lower-left corner. It is georeferenced in UTM zone 35N (EPSG:32635) with 10 m pixels.

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import geotiff_info, read_geotiff, write_geotiff

# A synthetic scene: bare soil, vegetation on the right half, water in one corner.
rows, cols = 128, 128
y, x = np.mgrid[0:rows, 0:cols]
vegetation = x >= cols // 2
water = (y >= 96) & (x < 48)

# Surface reflectance of the bands blue, green, red and near infrared.
blue = np.where(water, 0.06, np.where(vegetation, 0.03, 0.08))
green = np.where(water, 0.08, np.where(vegetation, 0.06, 0.10))
red = np.where(water, 0.04, np.where(vegetation, 0.04, 0.14))
nir = np.where(water, 0.02, np.where(vegetation, 0.45, 0.20))
noise = np.random.default_rng(42).normal(0.0, 0.005, size=(4, rows, cols))
stack = (np.stack([blue, green, red, nir]) + noise).clip(0, 1).astype("float32")

# Georeference: UTM zone 35N, upper-left corner at (500000, 6700000), 10 m pixels.
transform = from_origin(500000.0, 6700000.0, 10.0, 10.0)
write_geotiff(stack, "scene.tif", crs="EPSG:32635", transform=transform,
              descriptions=["blue", "green", "red", "nir"])

info = geotiff_info("scene.tif")
print(info["crs"], info["width"], info["height"], info["count"], info["dtype"])

data, meta = read_geotiff("scene.tif")
print(data.shape, data.dtype, meta["crs"])
```

```text
EPSG:32635 128 128 4 float32
(4, 128, 128) float32 EPSG:32635
```

`write_geotiff` takes the array first and the path second, writes a tiled, DEFLATE-compressed GeoTIFF and returns the path. `read_geotiff` returns a `(bands, rows, cols)` array (float32 by default) and a metadata dictionary with, among others, the keys `crs`, `transform`, `bounds`, `nodata`, `width`, `height`, `count`, `descriptions` and `is_cog`. `geotiff_info` reads the metadata without reading pixels.

## 4. Computing spectral indices

The Normalized Difference Vegetation Index [1] and the Normalized Difference Water Index of McFeeters [2] are normalised differences of two bands:

$$
\mathrm{NDVI} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{red}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{red}}}, \qquad
\mathrm{NDWI} = \frac{\rho_{\mathrm{green}} - \rho_{\mathrm{NIR}}}{\rho_{\mathrm{green}} + \rho_{\mathrm{NIR}}}
$$

```python
import numpy as np

from unbihexium.indices import compute_index, ndvi, ndwi
from unbihexium.io import is_cog, read_geotiff, write_geotiff

data, meta = read_geotiff("scene.tif")
blue, green, red, nir = data

vi = ndvi(nir=nir, red=red)
wi = ndwi(green=green, nir=nir)
by_name = compute_index("NDVI", nir=nir, red=red)
print(round(float(np.nanmean(vi)), 3), round(float(np.nanmax(wi)), 3), np.allclose(vi, by_name))

path = write_geotiff(vi.astype("float32"), "ndvi.tif", crs=meta["crs"],
                     transform=meta["transform"], cog=True)
print(path, is_cog(path))
```

```text
0.459 0.909 True
ndvi.tif True
```

The index functions accept arrays of any shape, compute in float64 and return NaN where the denominator is zero or a value is not finite. `compute_index` evaluates any index of `unbihexium.indices.INDEX_FUNCTIONS` by name. `cog=True` writes the Cloud Optimized GeoTIFF layout with the GDAL COG driver, which adds internal overviews when the raster is larger than one block; `is_cog` checks the layout.

## 5. Working with the Raster class

`unbihexium.core.Raster` keeps an array together with its CRS, affine transform and no-data value, and offers statistics, windows, resampling, reprojection and clipping.

```python
from unbihexium.core import Raster

raster = Raster.from_file("ndvi.tif")
print(raster.shape, raster.crs, raster.resolution)
stats = raster.statistics()[0]
print({key: round(stats[key], 3) for key in ("min", "mean", "max")})
```

```text
(1, 128, 128) EPSG:32635 (10.0, 10.0)
{'min': -0.824, 'mean': 0.459, 'max': 0.908}
```

## 6. From a raster to polygons

A threshold on NDVI gives a vegetation mask. `unbihexium.postprocessing` removes small objects and converts the mask into polygons in map coordinates.

```python
from unbihexium.io import read_geotiff
from unbihexium.postprocessing import polygons_to_geodataframe, remove_small_objects, threshold

values, meta = read_geotiff("ndvi.tif")
mask = threshold(values[0], threshold=0.5)                   # 1 where NDVI > 0.5
mask = remove_small_objects(mask.astype(bool), min_size=20)  # drop isolated pixels
polygons = polygons_to_geodataframe(mask.astype("uint8"), transform=meta["transform"],
                                    crs=meta["crs"])
print(len(polygons), round(float(polygons.area.sum())), polygons.crs.to_string())
polygons.to_file("vegetation.geojson", driver="GeoJSON")
```

```text
1 819200 EPSG:32635
```

The vegetated right half of the scene (64 x 128 pixels of 100 square metres each) becomes one polygon of 819,200 square metres, which is written as GeoJSON.

## 7. Building and running a tiny model

### 7.1 Starter models

The model zoo contains 520 models: 130 families, each in the variants tiny, base, large and mega. Apart from the 28 models of the 7 spectral index families, which compute exact formulas, every model is an untrained starter model: a complete, trainable architecture with deterministic initial weights that are generated locally from the model id and verified against a published SHA-256 digest. The predictions of a starter model are not meaningful until it is trained on labelled data. The example below uses `water_surface_detector_tiny`, a U-Net segmentation model whose input bands are exactly those of the test scene.

### 7.2 Building the model

```python
from unbihexium.zoo import get_model, load_model

entry = get_model("water_surface_detector_tiny")
print(entry.spec.bands, entry.spec.outputs, entry.requires_training)

model = load_model("water_surface_detector_tiny")  # built locally, verified by digest
print(model.model_id, model.task.value, model.num_parameters())
```

```text
('blue', 'green', 'red', 'nir') ('background', 'water') True
water_surface_detector_tiny segmentation 733090
```

`get_model` reads the catalogue and works without PyTorch; `load_model` builds the network in memory with PyTorch. `unbihexium zoo build` (Section 10) writes the same model to the model store.

### 7.3 Running the model

The task APIs of `unbihexium.ai` open a model, check the number of bands, tile the image, run the network and return a result object with georeferencing.

```python
from unbihexium.ai import WaterDetector

detector = WaterDetector(variant="tiny")  # untrained starter weights
result = detector.predict("scene.tif")
print(result.model_id, result.mask.shape, result.classes)
print(sorted(result.class_fractions()))
```

```text
water_surface_detector_tiny (128, 128) ['background', 'water']
['background', 'water']
```

The mask has the shape and grid of the input, but because the weights are untrained its content is arbitrary; this step only shows that the pipeline from file to result works.

## 8. Training briefly and predicting again

`unbihexium.ai.training.train` trains a zoo model on a dataset folder or, for a setup check, on generated synthetic samples. The run below uses 16 synthetic 64 x 64 chips for two epochs, which took about 38 seconds including interpreter start (measured on a 4 vCPU Linux container with CPython 3.13, CPU only).

```python
from unbihexium.ai import WaterDetector
from unbihexium.ai.training import TrainConfig, train

config = TrainConfig(epochs=2, batch_size=4, chip_size=64, output_dir="runs", verbose=False)
result = train("water_surface_detector_tiny", synthetic=16, config=config)
print(result.best_checkpoint)
print(sorted(result.best_metrics))

trained = WaterDetector(weights=result.best_checkpoint)
water = trained.predict("scene.tif")
print(water.model_id, water.mask.shape)
```

```text
runs/water_surface_detector_tiny/best.pt
['accuracy', 'f1_per_class', 'iou_per_class', 'kappa', 'loss', 'mf1', 'miou', 'pixels', 'precision_per_class', 'recall_per_class']
water_surface_detector_tiny (128, 128)
```

The run directory contains `best.pt`, `last.pt` and `history.json`. Checkpoints hold plain data only and are loaded with `torch.load(weights_only=True)`. Synthetic data only proves that training works; a useful model needs labelled data of your sensor and area, organised as described in [docs/model_zoo/training.md](../model_zoo/training.md).

## 9. An exact spectral index model

The spectral index families of the zoo (for example `ndvi_calculator`) compute their formula exactly and need no training. They take their bands in catalogue order, here red and near infrared:

```python
from unbihexium.ai import NDVICalculator
from unbihexium.io import read_geotiff, write_geotiff

data, meta = read_geotiff("scene.tif")
write_geotiff(data[[2, 3]], "red_nir.tif", crs=meta["crs"], transform=meta["transform"])

result = NDVICalculator(variant="tiny").predict("red_nir.tif")
print(result.names, round(result.summary()["ndvi"]["mean"], 3))
```

```text
['ndvi'] 0.459
```

The mean equals the value computed with `unbihexium.indices.ndvi` in Section 4.

## 10. The same session on the command line

### 10.1 Information and spectral indices

The commands below run in the same working directory. `unbihexium index` takes 1-based band numbers of the input file; the defaults follow the 13-band Sentinel-2 Level-1C order, so they are set explicitly for the four-band test scene.

```bash
unbihexium --version
unbihexium info
unbihexium index ndvi -i scene.tif -o ndvi_cli.tif --blue 1 --green 2 --red 3 --nir 4
```

```text
unbihexium, version 1.0.1
Unbihexium v1.0.1
Registered capabilities: 147
Model zoo models: 520 (catalogue 2.0.0)
Registered pipelines: 5
Wrote: ndvi_cli.tif (NDVI)
```

### 10.2 Browsing the catalogue and building a model

```bash
unbihexium zoo list --task segmentation --variant tiny --domain water
unbihexium zoo build water_surface_detector_tiny
unbihexium zoo verify water_surface_detector_tiny
unbihexium zoo where water_surface_detector_tiny
```

```text
                         Model zoo (3 models)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model ID                       ┃ Task         ┃ Domain ┃ Parameters ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ marine_pollution_detector_tiny │ segmentation │ water  │    733,971 │
│ reservoir_monitor_tiny         │ segmentation │ water  │    733,090 │
│ water_surface_detector_tiny    │ segmentation │ water  │    733,090 │
└────────────────────────────────┴──────────────┴────────┴────────────┘
Cached: <working directory>/cache/models/water_surface_detector_tiny
Verified: water_surface_detector_tiny
<working directory>/cache/models/water_surface_detector_tiny/model.pt
```

The model directory contains `model.pt`, `config.json` and the checksum file `model.sha256`.

### 10.3 Predicting, training and running a pipeline

`unbihexium predict` chooses the output format from the task of the model: a single-band class GeoTIFF for segmentation and change detection, a float32 GeoTIFF for dense outputs, GeoJSON for detection and JSON for scene values.

```bash
unbihexium predict water_surface_detector_tiny scene.tif water_mask.tif
unbihexium train water_surface_detector_tiny --synthetic 16 --epochs 2 --batch-size 4 --chip-size 64
unbihexium predict runs/water_surface_detector_tiny/best.pt scene.tif water_trained.tif
unbihexium predict ndvi_calculator_tiny red_nir.tif ndvi_model.tif
unbihexium pipeline run water_detection -i scene.tif -o water_pipeline.tif -p variant=tiny
```

```text
Wrote: water_mask.tif (water_surface_detector_tiny)
epoch 1/2 loss 1.1126 miou 0.7690
epoch 2/2 loss 0.5176 miou 0.8731
Best epoch: 2
Best checkpoint: runs/water_surface_detector_tiny/best.pt
{
  "loss": 0.350311815738678,
  "accuracy": 0.93243408203125,
  "miou": 0.8730547534372347,
  ...
}
Wrote: water_trained.tif (water_surface_detector_tiny)
Wrote: ndvi_model.tif (ndvi_calculator_tiny)
Completed: 4b6efb05-23c4-4302-a0fd-9e7156733fa5 -> water_pipeline.tif
```

The training metrics are computed on synthetic validation chips and say nothing about performance on real imagery; the run id of the pipeline is a random UUID.

### 10.4 Exporting to ONNX

An ONNX export runs without PyTorch, for example in a small deployment that installs only the `onnx` extra. `zoo export` compares ONNX Runtime and PyTorch outputs before it reports success.

```bash
unbihexium zoo export runs/water_surface_detector_tiny/best.pt water.onnx
unbihexium predict water.onnx scene.tif water_onnx.tif --backend onnx
```

```text
Exported: water.onnx
Wrote: water_onnx.tif (water_surface_detector_tiny)
```

## 11. Next steps

| Topic | Document |
| --- | --- |
| Settings, environment variables and the REST service | [configuration.md](configuration.md) |
| Every command and option | [docs/reference/cli.md](../reference/cli.md) |
| Every public module, class and function | [docs/reference/api.md](../reference/api.md) |
| Dataset layout and training | [docs/model_zoo/training.md](../model_zoo/training.md) |
| Inference, tiling and backends | [docs/model_zoo/inference.md](../model_zoo/inference.md) |
| Families, tasks and variants of the model zoo | [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md) |
| Tutorials by topic | [docs/tutorials/index.md](../tutorials/index.md) |
| Responsible use of model outputs | [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) |

## References

[1] J. W. Rouse, R. H. Haas, J. A. Schell and D. W. Deering. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[2] S. K. McFeeters. The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432. 1996. <https://doi.org/10.1080/01431169608948714>

<!--
=============================================================================
End of file docs/getting_started/quickstart.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
