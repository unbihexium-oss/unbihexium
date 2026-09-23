<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : README.md
Title       : Unbihexium
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Unbihexium

[![CI](https://github.com/unbihexium-oss/unbihexium/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/unbihexium-oss/unbihexium/actions/workflows/ci.yml)
[![CodeQL](https://github.com/unbihexium-oss/unbihexium/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/unbihexium-oss/unbihexium/actions/workflows/codeql.yml)
[![Package](https://github.com/unbihexium-oss/unbihexium/actions/workflows/package.yml/badge.svg?branch=main)](https://github.com/unbihexium-oss/unbihexium/actions/workflows/package.yml)
[![Fuzzing](https://github.com/unbihexium-oss/unbihexium/actions/workflows/fuzz.yml/badge.svg?branch=main)](https://github.com/unbihexium-oss/unbihexium/actions/workflows/fuzz.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/unbihexium-oss/unbihexium/badge)](https://scorecard.dev/viewer/?uri=github.com/unbihexium-oss/unbihexium)
[![PyPI](https://img.shields.io/pypi/v/unbihexium?label=PyPI)](https://pypi.org/project/unbihexium/)
[![Python 3.10 to 3.14](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB)](https://github.com/unbihexium-oss/unbihexium/blob/main/pyproject.toml)
[![Licence: MPL-2.0](https://img.shields.io/github/license/unbihexium-oss/unbihexium?label=Licence)](https://github.com/unbihexium-oss/unbihexium/blob/main/LICENSE.txt)

| Field | Value |
| --- | --- |
| Document | UBX-DOC-README |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](https://github.com/unbihexium-oss/unbihexium/blob/main/MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

Unbihexium is an open source Python library for Earth observation, geospatial analysis, remote sensing and synthetic aperture radar (SAR). This document is the front page of the project and its description on the Python Package Index. It is written for users who want to install the library and run a first analysis, for researchers who need to know what the software does and how to cite it, and for contributors, security researchers and auditors who need an entry point into the governance, security and supply-chain documents of the repository. It covers the scope and status of the project, including the fact that the learned models of the model zoo are untrained starter models, the installation options, tested examples for the Python API and the command line, an overview of every subpackage, the model zoo, the REST service, reproducibility and supply-chain security, the repository layout, and the contribution, security, citation and licensing arrangements.

## Contents

1. [Overview](#1-overview)
2. [Status and scope](#2-status-and-scope)
3. [Installation](#3-installation)
4. [Quick start in Python](#4-quick-start-in-python)
5. [Quick start on the command line](#5-quick-start-on-the-command-line)
6. [Feature overview by package](#6-feature-overview-by-package)
7. [Model zoo](#7-model-zoo)
8. [REST service](#8-rest-service)
9. [Configuration](#9-configuration)
10. [Reproducibility and supply-chain security](#10-reproducibility-and-supply-chain-security)
11. [Project layout](#11-project-layout)
12. [Documentation](#12-documentation)
13. [Contributing](#13-contributing)
14. [Security](#14-security)
15. [Citation](#15-citation)
16. [Licence and acknowledgements](#16-licence-and-acknowledgements)
17. [References](#references)

## 1. Overview

Unbihexium brings together, in one typed Python package, the building blocks of a typical Earth observation workflow:

- reading and writing rasters and vectors (GeoTIFF, Cloud Optimized GeoTIFF, Zarr, GeoJSON, GeoParquet) and searching SpatioTemporal Asset Catalogs (STAC);
- radiometric preprocessing for Sentinel-2 and Landsat, cloud and quality masks, pansharpening and resampling;
- 28 spectral indices, SAR calibration, speckle filtering, polarimetric decomposition and interferometry;
- terrain derivatives and hydrology, geostatistics (variograms, kriging, spatial autocorrelation) and spatial analysis (zonal statistics, suitability, least-cost paths, network analysis);
- accuracy assessment and image-quality metrics, and visualisation helpers;
- a model zoo of 130 model families in four size variants (520 models) with training, evaluation, tiled inference, ONNX export and task-level Python APIs;
- a command line interface (`unbihexium`) and a FastAPI-based REST service.

The name refers to the hypothetical chemical element with atomic number 126. The project is maintained by Olaf Yunus Laitinen Imanov (University of Helsinki) on behalf of the Unbihexium OSS Foundation, and it is distributed under the Mozilla Public License 2.0 [1].

## 2. Status and scope

### 2.1 What is ready to use

The classical processing functions (input and output, preprocessing, spectral indices, SAR, terrain, geostatistics, analysis, metrics and visualisation) are deterministic implementations of published methods. They are covered by the tests in `tests/`: the unit and integration tests run in CI on CPython 3.10, 3.11, 3.12, 3.13 and 3.14, and the end-to-end tests on the newest supported version.

### 2.2 The models are starter models

The model zoo contains 520 models: 130 families, each in the variants tiny, base, large and mega. **Apart from the 7 spectral index families (28 models), which compute exact formulas and need no training, every model is an untrained starter model.** A starter model is a complete, trainable network architecture for its task with deterministic initial weights that anyone can rebuild and verify against a published SHA-256 digest. Starter models have not been trained on Earth observation data, so their predictions are not meaningful until you train or fine-tune them on labelled data for your sensor and area of interest. The library provides the tools for this (`unbihexium train`, `unbihexium evaluate`, `unbihexium.ai.training`). No accuracy figures are published for the zoo, because there are no trained weights to measure.

### 2.3 Release status

The version declared in `pyproject.toml` is 1.0.1. The distribution published on PyPI as version 1.0.1 was built from the tag `v1.0.1` (21 December 2025). The main branch has changed substantially since that tag (among other things: model training and evaluation, the `predict` and `zoo build` commands, the REST prediction route, the licence change from Apache-2.0 to MPL-2.0 and the classifiers for Python 3.13 and 3.14) and has not yet been released under a new version number. **The examples in this document are tested against the main branch.** Until the next release, install from source (Section 3.4) to use them. Release notes are kept in [CHANGELOG.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CHANGELOG.md), and the versioning and support policy in [VERSIONING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/VERSIONING.md).

### 2.4 Out of scope

Unbihexium does not ship trained weights, labelled training data or imagery, does not provide a hosted service, and makes no claim of fitness for any operational, safety-critical or legal purpose. Read [RESPONSIBLE_USE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/RESPONSIBLE_USE.md) before deploying models whose output affects people, property or the environment.

## 3. Installation

### 3.1 Requirements

- CPython 3.10, 3.11, 3.12, 3.13 or 3.14 on Linux, macOS or Windows.
- No compiler and no system GDAL: the binary wheels of rasterio, pyproj, shapely and onnxruntime bundle GDAL, PROJ, GEOS and their native libraries.
- PyTorch only for building, training and exporting models (extra `torch`); inference on exported ONNX models needs only the extra `onnx`.

### 3.2 From PyPI

```bash
python -m pip install unbihexium
```

The core installation covers input and output, preprocessing, indices, SAR, terrain, geostatistics, analysis, metrics, visualisation, the catalogue of the model zoo and the command line. Optional features are grouped in extras:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `onnx` | onnxruntime, onnx | Inference on ONNX exports without PyTorch |
| `torch` | torch, torchvision, onnx | Building, training, evaluating and exporting zoo models |
| `gpu` | `torch` extra, cupy-cuda12x | CUDA 12 acceleration (install the matching PyTorch CUDA build first) |
| `serving` | fastapi, starlette, uvicorn, python-multipart | The REST service in `unbihexium.serving` |
| `dask` | dask[complete] | Chunked and distributed processing |
| `ray` | ray | Cluster computing |
| `zarr` | zarr, numcodecs | Zarr input and output |
| `netcdf` | netCDF4, h5py | NetCDF and HDF5 files |
| `stac` | pystac, pystac-client | STAC search and metadata |
| `parquet` | pyarrow | GeoParquet input and output |
| `test` | pytest and plugins, httpx | Running the test suite |
| `dev` | `test` extra, ruff, pyright, pre-commit, bandit, pip-audit, build, twine, tox | Development |
| `all` | every extra except `gpu` | A complete environment |

```bash
python -m pip install "unbihexium[torch,onnx,serving]"
```

### 3.3 Container image

The workflow `.github/workflows/docker.yml` builds the image from the [Dockerfile](https://github.com/unbihexium-oss/unbihexium/blob/main/Dockerfile) and pushes it to the GitHub Container Registry as `ghcr.io/unbihexium-oss/unbihexium`. Pushes to `main` are tagged `main`, version tags are tagged `<major>.<minor>.<patch>` and `<major>.<minor>`, and every image is also tagged with its commit (`sha-<short sha>`); there is no `latest` tag. The image contains the ONNX Runtime backend and the REST service, installed from the hashed lock file `requirements.txt`, runs as the unprivileged user `unbihexium` and does not contain model weights or PyTorch.

```bash
docker pull ghcr.io/unbihexium-oss/unbihexium:main
docker run --rm ghcr.io/unbihexium-oss/unbihexium:main unbihexium info
docker run --rm -p 8000:8000 ghcr.io/unbihexium-oss/unbihexium:main \
    uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000
```

To build the image locally, run `docker build -t unbihexium:local .` in the repository root. [docker-compose.yml](https://github.com/unbihexium-oss/unbihexium/blob/main/docker-compose.yml) and the manifests under [deploy/](https://github.com/unbihexium-oss/unbihexium/tree/main/deploy) (a Helm chart and a Kubernetes deployment) start the REST service; see [docs/operations/docker.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/operations/docker.md).

### 3.4 From source

```bash
git clone https://github.com/unbihexium-oss/unbihexium.git
cd unbihexium
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[torch,onnx,serving]"
unbihexium --version
```

For a reproducible development environment, install the hashed lock file of all extras and then the package without resolving dependencies again:

```bash
python -m pip install --require-hashes -r requirements-dev.txt
python -m pip install --no-deps -e .
pre-commit install
```

`requirements.txt` is the corresponding lock file of the runtime dependencies with the `onnx` and `serving` extras. Both lock files cover CPython 3.10 to 3.14 and are regenerated with `make lock`.

## 4. Quick start in Python

The examples below run in the given order in an empty working directory. They create small synthetic rasters, so no data download is needed. Examples 4.1 and 4.2 need only the core installation; 4.3 needs the `torch` extra. Models are built into the directory named by `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`).

### 4.1 Spectral index from a GeoTIFF

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.indices import ndvi
from unbihexium.io import is_cog, read_geotiff, write_geotiff

# A synthetic 4-band reflectance scene (blue, green, red, near infrared), 10 m pixels.
rng = np.random.default_rng(0)
stack = rng.uniform(0.02, 0.5, size=(4, 64, 64)).astype("float32")
write_geotiff(stack, "scene.tif", crs="EPSG:32635", transform=from_origin(500000, 6700000, 10, 10))

# Read it back with its georeferencing and compute NDVI.
data, meta = read_geotiff("scene.tif")
vegetation = ndvi(nir=data[3], red=data[2])

# Write the result as a Cloud Optimized GeoTIFF.
path = write_geotiff(vegetation.astype("float32"), "ndvi.tif", crs=meta["crs"], transform=meta["transform"], cog=True)
print(path, is_cog(path))
```

### 4.2 SAR, terrain, geostatistics and accuracy assessment

```python
import numpy as np

from unbihexium.geostat import OrdinaryKriging
from unbihexium.metrics import cohen_kappa, confusion_matrix
from unbihexium.sar import lee_filter, power_to_db
from unbihexium.terrain import hillshade, slope

rng = np.random.default_rng(1)

# Speckle filtering of a single-look backscatter image, then conversion to decibels.
sigma0 = rng.gamma(shape=1.0, scale=0.05, size=(128, 128))
sigma0_db = power_to_db(lee_filter(sigma0, window_size=7, looks=1.0))

# Slope in degrees and hillshade of a synthetic 30 m elevation model.
y, x = np.mgrid[0:100, 0:100]
dem = 200.0 + 0.5 * x + 20.0 * np.sin(y / 15.0)
slope_deg = slope(dem, resolution=30.0)
shade = hillshade(dem, resolution=30.0)

# Ordinary kriging of 50 scattered observations.
coords = rng.uniform(0, 1000, size=(50, 2))
values = 0.01 * coords[:, 0] + rng.normal(0, 0.5, size=50)
kriged = OrdinaryKriging().fit(coords, values).predict(np.array([[500.0, 500.0], [100.0, 900.0]]))
print(kriged.predictions, kriged.variance)

# Confusion matrix and Cohen's kappa of a classification against a reference.
reference = rng.integers(0, 3, size=(64, 64))
predicted = np.where(rng.random((64, 64)) < 0.9, reference, (reference + 1) % 3)
print(cohen_kappa(confusion_matrix(reference, predicted)))
```

### 4.3 Model zoo: build, train and predict

The detector trained here learns from 32 synthetic chips for two epochs. This only checks that the training setup works; for real use, train on a labelled dataset as described in [docs/model_zoo/training.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/model_zoo/training.md).

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.ai import NDVICalculator, ShipDetector
from unbihexium.ai.training import TrainConfig, train
from unbihexium.io import write_geotiff
from unbihexium.zoo import list_models, load_model

# Browse the catalogue (works without PyTorch).
for entry in list_models(task="detection", variant="tiny")[:3]:
    print(entry.model_id, entry.spec.bands, entry.num_parameters)

# Build a starter model locally; its weights are verified against the published digest.
model = load_model("ship_detector_tiny")
print(model.summary())

# Train it briefly on synthetic data to check the setup.
config = TrainConfig(epochs=2, batch_size=4, chip_size=64, output_dir="runs", verbose=False)
result = train("ship_detector_tiny", synthetic=32, config=config)
print(result.best_checkpoint, result.best_metrics["map50"])

# Run the trained checkpoint on an RGB GeoTIFF and export the detections as GeoJSON.
rgb = np.random.default_rng(2).uniform(0, 0.3, size=(3, 128, 128)).astype("float32")
write_geotiff(rgb, "harbour.tif", crs="EPSG:32635", transform=from_origin(500000, 6700000, 1, 1))
detections = ShipDetector(weights=result.best_checkpoint, threshold=0.4).predict("harbour.tif")
print(detections.count, detections.counts_by_class())
geojson = detections.to_geojson()

# Spectral index models are exact formulas and need no training (bands: red, nir).
red_nir = np.stack([rgb[0], rgb[0] + 0.2])
write_geotiff(red_nir, "red_nir.tif", crs="EPSG:32635", transform=from_origin(500000, 6700000, 1, 1))
print(NDVICalculator().predict("red_nir.tif").summary())
```

## 5. Quick start on the command line

The commands below are tested in the same working directory as the Python examples (they use `scene.tif` and `harbour.tif`). Training, prediction with checkpoints and export need the `torch` extra; prediction on the exported ONNX file needs the `onnx` extra.

```bash
# Library and catalogue information
unbihexium info
unbihexium zoo list --task detection --variant tiny
unbihexium zoo info ship_detector_base

# Exact spectral index of a GeoTIFF (1-based band numbers of the input file)
unbihexium index ndvi -i scene.tif -o ndvi_cli.tif --blue 1 --green 2 --red 3 --nir 4

# Build a starter model into the local store, verify it and print its path
unbihexium zoo build ship_detector_tiny
unbihexium zoo verify ship_detector_tiny
unbihexium zoo where ship_detector_tiny

# Check the training setup on synthetic data, then predict with the checkpoint
unbihexium train ship_detector_tiny --synthetic 32 --epochs 2 --chip-size 64
unbihexium predict runs/ship_detector_tiny/best.pt harbour.tif ships.geojson

# Export to ONNX (verified against PyTorch) and predict without PyTorch
unbihexium zoo export runs/ship_detector_tiny/best.pt ship_detector.onnx
unbihexium predict ship_detector.onnx harbour.tif ships_onnx.geojson --backend onnx

# Registered processing pipelines
unbihexium pipeline list
```

The complete command set is:

| Command | Purpose |
| --- | --- |
| `unbihexium info` | Version, number of registered capabilities, models and pipelines |
| `unbihexium index` | Compute a spectral index of a raster and write it as GeoTIFF |
| `unbihexium zoo list`, `info` | Browse the model catalogue |
| `unbihexium zoo build`, `verify`, `where`, `clear` | Manage the local model store |
| `unbihexium zoo export` | Export a model or checkpoint to ONNX and verify it |
| `unbihexium train` | Train or fine-tune a zoo model on a dataset folder or synthetic data |
| `unbihexium evaluate` | Evaluate a model on a dataset split |
| `unbihexium predict` | Run a model on a raster and write the result |
| `unbihexium pipeline list`, `run` | List and run registered processing pipelines |

Every command documents its options with `--help`; the full reference is [docs/reference/cli.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/reference/cli.md). Bash completion is provided in [scripts/unbihexium-completion.bash](https://github.com/unbihexium-oss/unbihexium/blob/main/scripts/unbihexium-completion.bash).

## 6. Feature overview by package

Importing `unbihexium` loads only the version; each subpackage is imported when needed, and importing `unbihexium.ai` does not import PyTorch.

| Package | Contents |
| --- | --- |
| `unbihexium.core` | Data model: `Raster`, `Vector`, `Tile` and `TileGrid`, `Scene`, `SensorModel` and spectral band definitions, `Product`, `ModelWrapper`, `Pipeline` and `PipelineRun`, `Evidence` and `ProvenanceRecord`, the spectral `IndexRegistry` |
| `unbihexium.io` | GeoTIFF and Cloud Optimized GeoTIFF read and write with windows, overviews and compression; Zarr; GeoJSON validation, reprojection and ring orientation; GeoParquet; STAC items, catalogue traversal and search |
| `unbihexium.preprocessing` | Sentinel-2 L2A and Landsat Collection 2 scaling, TOA reflectance, radiance and brightness temperature; SCL and QA cloud masks; dark object subtraction; stretches, histogram equalisation and matching; pansharpening (Brovey, IHS, Gram-Schmidt); resampling and aggregation; tensor transforms |
| `unbihexium.indices` | NDVI [7], EVI, EVI2, SAVI, MSAVI, OSAVI, ARVI, GNDVI, kNDVI, VARI, NDRE, CI green and red edge, NDWI, MNDWI, AWEI, NDMI, MSI, NBR, NBR2, dNBR, RdNBR and burn severity classes, NDBI, BSI, NDSI, the radar vegetation index and the cross-polarisation ratio; `compute_index` by name |
| `unbihexium.sar` | Radiometric calibration (sigma0, beta0, gamma0), decibel conversion, multilooking, Lee, refined and enhanced Lee, Frost, Kuan and Gamma MAP filters; covariance and coherency matrices, Pauli, Freeman-Durden, Yamaguchi and H/A/alpha decompositions; interferograms, coherence, Goldstein filtering, phase unwrapping, displacement and height of ambiguity |
| `unbihexium.terrain` | Slope, aspect, curvature, hillshade, TPI, TRI, VRM and roughness; depression filling, D8 flow direction and accumulation, stream extraction, watersheds and TWI; viewshed |
| `unbihexium.geostat` | Empirical and model variograms, ordinary and universal kriging, inverse distance weighting, spatial weights, global and local Moran's I, Geary's C, Getis-Ord Gi* |
| `unbihexium.analysis` | Zonal statistics, weighted overlay, AHP, fuzzy membership and reclassification for suitability analysis, cost distance and least-cost paths, A* path finding and network accessibility |
| `unbihexium.postprocessing` | Activations, thresholds and confidence masks, morphology, sieving and majority filters, connected components, tile blending and stitching, raster to polygon vectorisation and simplification |
| `unbihexium.metrics` | Confusion matrix, overall accuracy, kappa, precision, recall, F1, IoU and Dice; good-practice accuracy assessment and stratified area estimation; change and transition matrices; regression metrics; PSNR, SSIM, SAM, ERGAS and Q index |
| `unbihexium.visualization` | Colour maps and look-up tables, Sentinel-2 and Landsat 8/9 composites, class colouring and legends, hillshade and relief shading, quicklooks and PNG output with world files |
| `unbihexium.ai` | Task APIs (for example `ShipDetector`, `BuildingDetector`, `LandCoverClassifier`, `FloodMapper`, `ChangeDetector`, `TreeHeightEstimator`, `SuperResolution`), the tiled `Predictor` for PyTorch and ONNX Runtime, result objects with GeoJSON and GeoTIFF output, datasets, training and evaluation |
| `unbihexium.zoo` | The model catalogue, variants, local model store, weight digests and verification, checkpoints and ONNX export |
| `unbihexium.serving` | `create_app()`, the FastAPI REST service with request limits, optional API key and rate limiting |
| `unbihexium.cli` | The `unbihexium` command |

Supporting packages are `unbihexium.registry` (capability, model and pipeline registries), `unbihexium.config` (layered settings) and `unbihexium.utils` (logging, hashing, seeding, tiling, atomic file writes). The public names of each package are listed in its `__all__` and described in [docs/reference/api.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/reference/api.md).

## 7. Model zoo

### 7.1 Families, tasks and architectures

The catalogue [src/unbihexium/zoo/catalog.yaml](https://github.com/unbihexium-oss/unbihexium/blob/main/src/unbihexium/zoo/catalog.yaml) (version 2.0.0) defines 130 families. For each family it records the input bands, the number of acquisitions, the outputs and their units, the labels needed for training and suitable data sources.

| Task | Families | Models | Architecture |
| --- | --- | --- | --- |
| Detection | 19 | 76 | CenterNet, anchor-free, output stride 4 [8] |
| Segmentation | 26 | 104 | U-Net [9] |
| Change detection | 6 | 24 | U-Net on two stacked acquisitions |
| Dense regression | 49 | 196 | U-Net with a regression output |
| Scene regression | 11 | 44 | Residual encoder with a pooled regression head |
| Enhancement | 11 | 44 | Residual U-Net, image to image |
| Super-resolution | 1 | 4 | EDSR-style residual network with sub-pixel convolution [10] |
| Spectral index | 7 | 28 | Exact formula, no weights |
| **Total** | **130** | **520** | |

### 7.2 Variants

| Variant | Base channels | Encoder levels | Blocks per level | Tile size | Parameters per learned model | Parameters of the variant |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 16 | 3 | 1 | 256 px | 134,992 to 735,428 | 86,951,209 |
| base | 32 | 4 | 1 | 256 px | 657,264 to 7,063,428 | 825,294,793 |
| large | 48 | 4 | 2 | 512 px | 2,784,528 to 22,066,564 | 2,611,810,665 |
| mega | 64 | 5 | 2 | 512 px | 6,109,872 to 60,460,548 | 7,131,069,449 |

The 520 models have 10,655,126,116 parameters in total.

### 7.3 Starter weights and verification

No weights are downloaded. The starter weights of each model are generated locally and deterministically from its model id, and their digest (SHA-256 over the sorted state dictionary) is compared with the published value in [src/unbihexium/zoo/digests.json](https://github.com/unbihexium-oss/unbihexium/blob/main/src/unbihexium/zoo/digests.json). The workflow `.github/workflows/model-zoo.yml` rebuilds the tiny variants on pull requests and all 520 models weekly to detect platform drift. Checkpoints contain plain data only and are loaded with `torch.load(weights_only=True)`, so loading a checkpoint cannot execute code.

As stated in Section 2.2, only the 28 spectral index models produce meaningful output without training. The per-family model cards are indexed in [model_zoo/MODEL_CARDS.md](https://github.com/unbihexium-oss/unbihexium/blob/main/model_zoo/MODEL_CARDS.md), and one example notebook per family is in [examples/notebooks/](https://github.com/unbihexium-oss/unbihexium/tree/main/examples/notebooks).

## 8. REST service

`unbihexium.serving` provides a FastAPI application (extra `serving`). Start it with uvicorn; interactive OpenAPI documentation is then available at `/docs`.

```bash
uvicorn unbihexium.serving.app:app --host 127.0.0.1 --port 8000
```

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Liveness and readiness |
| `GET /capabilities`, `GET /capabilities/{capability_id}` | Registered capabilities |
| `GET /models`, `GET /models/{model_id}` | Models with task, domain and variant filters and pagination; bands, outputs and units of one model |
| `GET /pipelines` | Registered pipelines |
| `POST /predict/{model_id}` | Run any zoo model on an image sent as a nested JSON array or a base64-encoded NumPy array |
| `POST /infer/{model_id}`, `/detect/{model_id}`, `/segment/{model_id}` | Earlier task-specific routes |

The following requests were tested against a local server; the NDVI model expects the bands red and near infrared, in that order:

```bash
curl -s http://127.0.0.1:8000/health
curl -s "http://127.0.0.1:8000/models?task=detection&variant=tiny&limit=2"
curl -s -X POST http://127.0.0.1:8000/predict/ndvi_calculator_tiny \
    -H "Content-Type: application/json" \
    -d '{"image": [[[0.05, 0.06], [0.04, 0.05]], [[0.40, 0.45], [0.30, 0.35]]]}'
```

The service rejects request bodies above a size limit (413), images above pixel and value limits, unsupported media types (415), unknown models (404) and invalid input (422). An API key (header `X-API-Key`, compared in constant time and required on every route except `/health`), a per-client rate limit (429) and CORS origins are configured through `unbihexium.config`, for example `UNBIHEXIUM_SERVING__API_KEY` and `UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE`. By default no API key is set, no rate limit applies and all CORS origins are allowed, so set these before exposing the service beyond a trusted network.

## 9. Configuration

Settings are layered, later layers winning: the built-in defaults, a YAML file (passed to `load_config` or named by `UNBIHEXIUM_CONFIG`), environment variables of the form `UNBIHEXIUM_<SECTION>__<KEY>` for the sections `model`, `processing` and `serving` (for example `UNBIHEXIUM_MODEL__BATCH_SIZE=16`), and explicit overrides. `UNBIHEXIUM_LOG_LEVEL` sets the log level and `UNBIHEXIUM_CACHE` the model store (default `~/.cache/unbihexium`). Unknown keys are errors. See [docs/getting_started/configuration.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/getting_started/configuration.md) and [.env.example](https://github.com/unbihexium-oss/unbihexium/blob/main/.env.example).

## 10. Reproducibility and supply-chain security

### 10.1 Reproducible environments and results

- **Locked dependencies.** `requirements.txt` (runtime with `onnx` and `serving`), `requirements-dev.txt` (all extras) and the CI lock files in `.github/requirements/` pin every package with SHA-256 hashes; CI installs with `--require-hashes`. The container image installs binary wheels only from `requirements.txt`, and its base image is pinned by digest.
- **Deterministic models.** Starter weights are derived from the model id and verified by digest (Section 7.3). Training takes a `seed`, and the normalisation statistics estimated from the training data are stored with the checkpoint, so inference and ONNX exports apply exactly the same scaling.
- **Continuous checks.** CI runs ruff, pyright and pytest on CPython 3.10 to 3.14; further workflows run integration and end-to-end tests, a REST smoke test, packaging checks with `twine check --strict`, notebook checks, model zoo consistency and reproducibility, markdownlint, link checks, and the project text policy.

### 10.2 Release integrity

Releases are built by [.github/workflows/release.yml](https://github.com/unbihexium-oss/unbihexium/blob/main/.github/workflows/release.yml) when a version tag is pushed. With the current workflow, each GitHub release contains the sdist and wheel, `SHA256SUMS.txt`, a Sigstore signature bundle (`.sigstore.json`) for each distribution [3], and the signed SLSA provenance of the build as `unbihexium-<tag>.intoto.jsonl` [2]; GitHub artifact attestations are created for the distributions. Earlier releases were produced by earlier versions of the workflow and may not carry every one of these files. A downloaded distribution is verified with:

```bash
python -m pip install sigstore
python -m sigstore verify github \
    --cert-identity https://github.com/unbihexium-oss/unbihexium/.github/workflows/release.yml@refs/tags/<tag> \
    <file>
gh attestation verify <file> --repo unbihexium-oss/unbihexium
```

### 10.3 Security automation

| Control | Implementation |
| --- | --- |
| Static analysis | CodeQL for Python and GitHub Actions; Bandit and ruff security rules |
| Fuzzing | atheris targets for the GeoJSON and STAC parsers in [fuzz/](https://github.com/unbihexium-oss/unbihexium/tree/main/fuzz) |
| Dependencies | Dependabot (pip, GitHub Actions, Docker), pip-audit, dependency review with a licence policy |
| Secrets | TruffleHog on pushes and pull requests |
| Container | Grype scan of the image, SPDX SBOM of every pushed image |
| Repository posture | OpenSSF Scorecard [4], [security-insights.yml](https://github.com/unbihexium-oss/unbihexium/blob/main/security-insights.yml) |
| Licensing | REUSE compliance [5], licence headers, licence check of all runtime dependencies |
| Pinning | Every GitHub Action is pinned by commit SHA |

Details are in [docs/security/supply_chain_security.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/security/supply_chain_security.md) and [docs/operations/ci_cd.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/operations/ci_cd.md).

## 11. Project layout

```text
unbihexium/
  src/unbihexium/        the package (subpackages listed in Section 6)
    zoo/catalog.yaml     model catalogue, the single source of truth of the zoo
    zoo/digests.json     published starter weight digests of all 520 models
  tests/                 unit, integration, end-to-end and benchmark tests
  model_zoo/             model cards, manifests, inventory and checksums
  docs/                  user, architecture, model zoo, security and operations documentation
  examples/              notebooks (one per model family), scripts and a serving example
  fuzz/                  atheris fuzz targets and seed corpora
  deploy/                Helm chart and Kubernetes manifests
  scripts/               lock merging, model validation and shell completion
  .github/               workflows, CI lock files, check scripts and templates
  Dockerfile             container image of the CLI and the REST service
  pyproject.toml         package metadata, dependencies and tool configuration
  requirements*.txt      hashed lock files
  Makefile, tox.ini      developer tasks and test environments
```

## 12. Documentation

| Topic | Location |
| --- | --- |
| Documentation index | [docs/index.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/index.md) |
| Installation, quick start, configuration | [docs/getting_started/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/getting_started) |
| Python API and CLI reference | [docs/reference/api.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/reference/api.md), [docs/reference/cli.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/reference/cli.md) |
| Architecture | [docs/architecture/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/architecture) |
| Capability domains | [docs/capabilities/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/capabilities) |
| Model zoo: catalogue, training, inference, distribution | [docs/model_zoo/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/model_zoo) |
| Security | [docs/security/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/security) |
| Operations: CI/CD, Docker, releasing | [docs/operations/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/operations) |
| Tutorials and notebooks | [docs/tutorials/](https://github.com/unbihexium-oss/unbihexium/tree/main/docs/tutorials) |
| Frequently asked questions, glossary | [docs/faq.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/faq.md), [docs/glossary.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/glossary.md) |
| Migration between versions | [docs/MIGRATION.md](https://github.com/unbihexium-oss/unbihexium/blob/main/docs/MIGRATION.md) |

Project policies are kept in the repository root: [GOVERNANCE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/GOVERNANCE.md), [MAINTAINERS.md](https://github.com/unbihexium-oss/unbihexium/blob/main/MAINTAINERS.md), [AUTHORS.md](https://github.com/unbihexium-oss/unbihexium/blob/main/AUTHORS.md), [ROADMAP.md](https://github.com/unbihexium-oss/unbihexium/blob/main/ROADMAP.md), [SUPPORT.md](https://github.com/unbihexium-oss/unbihexium/blob/main/SUPPORT.md), [VERSIONING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/VERSIONING.md), [RESPONSIBLE_USE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/RESPONSIBLE_USE.md), [PRIVACY.md](https://github.com/unbihexium-oss/unbihexium/blob/main/PRIVACY.md) and [COMPLIANCE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/COMPLIANCE.md).

## 13. Contributing

Contributions are welcome. [CONTRIBUTING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CONTRIBUTING.md) describes the development setup, the coding and documentation standards, the tests and checks expected before review, and the pull request process; participants follow the [CODE_OF_CONDUCT.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CODE_OF_CONDUCT.md). Pull request titles follow Conventional Commits, and the local checks mirror CI:

```bash
make check        # lint, format, types, tests, licences, text policy, YAML, notebooks, model zoo
pre-commit run --all-files
```

Bugs and feature requests go to the [issue tracker](https://github.com/unbihexium-oss/unbihexium/issues); questions are answered as described in [SUPPORT.md](https://github.com/unbihexium-oss/unbihexium/blob/main/SUPPORT.md).

## 14. Security

Do not report vulnerabilities in public issues. Use a [GitHub private security advisory](https://github.com/unbihexium-oss/unbihexium/security/advisories/new) or write to <yunus.z.imanov@helsinki.fi>. The supported versions, the handling process and the scope are defined in [SECURITY.md](https://github.com/unbihexium-oss/unbihexium/blob/main/SECURITY.md).

## 15. Citation

If you use Unbihexium in research or in a product, cite the version you used. [CITATION.cff](https://github.com/unbihexium-oss/unbihexium/blob/main/CITATION.cff), in the Citation File Format [6], is the authoritative metadata; GitHub renders it as "Cite this repository", and [CITATION.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CITATION.md) explains how to cite in text. The project has no DOI at present. The following entry matches CITATION.cff for version 1.0.1:

```bibtex
@software{unbihexium_1_0_1,
  author  = {{Unbihexium OSS Foundation} and Laitinen Imanov, Olaf Yunus},
  title   = {Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python},
  version = {1.0.1},
  date    = {2025-12-21},
  url     = {https://github.com/unbihexium-oss/unbihexium},
  license = {MPL-2.0}
}
```

APA style: Unbihexium OSS Foundation, & Laitinen Imanov, O. Y. (2025). *Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python* (Version 1.0.1) [Computer software]. <https://github.com/unbihexium-oss/unbihexium>

Please also cite GDAL, PROJ and ONNX Runtime where your results depend on them directly; their references are listed in CITATION.cff.

## 16. Licence and acknowledgements

### 16.1 Licence

Copyright 2025-2026 Unbihexium OSS Foundation and contributors. Unbihexium is licensed under the Mozilla Public License 2.0 [1]; the full text is in [LICENSE.txt](https://github.com/unbihexium-oss/unbihexium/blob/main/LICENSE.txt). The MPL-2.0 is a file-level copyleft licence: modified files of Unbihexium that you distribute must remain under the MPL-2.0, while your own files that use the library may be under any licence. Notices are in [NOTICE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/NOTICE.md), and the licences of third-party material in [THIRD_PARTY_NOTICES.md](https://github.com/unbihexium-oss/unbihexium/blob/main/THIRD_PARTY_NOTICES.md). The per-file licensing information follows the REUSE specification [5] ([REUSE.toml](https://github.com/unbihexium-oss/unbihexium/blob/main/REUSE.toml)). This summary is not legal advice.

### 16.2 Acknowledgements

Unbihexium builds on the work of many open source projects, in particular NumPy, SciPy, rasterio and GDAL, pyproj and PROJ, Shapely and GEOS, GeoPandas, scikit-image, scikit-learn, PyTorch, ONNX and ONNX Runtime, FastAPI, Click and Rich. The model architectures follow the published designs cited in Section 7.1, and the spectral indices, SAR methods and geostatistical estimators follow the publications cited in the source code of each module. The author is affiliated with the University of Helsinki.

## References

[1] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[2] OpenSSF. Supply-chain Levels for Software Artifacts (SLSA), specification version 1.0. 2023. <https://slsa.dev/spec/v1.0/>

[3] Sigstore project. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

[4] OpenSSF. OpenSSF Scorecard. 2026. <https://scorecard.dev/>

[5] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

[6] Druskat, S., Spaaks, J. H., Chue Hong, N., Haines, R., Baker, J., Bliven, S., Willighagen, E., Perez-Suarez, D. and Konovalov, O. Citation File Format, version 1.2.0. 2021. <https://citation-file-format.github.io/>

[7] Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[8] Zhou, X., Wang, D. and Kraehenbuehl, P. Objects as points. arXiv:1904.07850. 2019. <https://arxiv.org/abs/1904.07850>

[9] Ronneberger, O., Fischer, P. and Brox, T. U-Net: Convolutional networks for biomedical image segmentation. MICCAI 2015, LNCS 9351, 234-241. 2015. <https://arxiv.org/abs/1505.04597>

[10] Lim, B., Son, S., Kim, H., Nah, S. and Lee, K. M. Enhanced deep residual networks for single image super-resolution. CVPR Workshops. 2017. <https://arxiv.org/abs/1707.02921>

<!--
=============================================================================
End of file README.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
