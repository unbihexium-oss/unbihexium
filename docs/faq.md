<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/faq.md
Title       : Frequently Asked Questions
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Frequently Asked Questions

| Field | Value |
| --- | --- |
| Document | UBX-DOC-FAQ |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch (model catalogue 2.0.0) |

## Abstract

This document answers the questions that users, operators and contributors ask most often about Unbihexium: what the project is and what state it is in, how to install it, how it handles data, what the models of the model zoo can and cannot do, how the command line and the REST service behave, how to resolve common errors, and where to go for security, privacy and support matters. Each answer is short, reflects the current code of the main branch, and links to the document that treats the subject in full. Commands and outputs quoted here were run against the main branch on 24 September 2026. The document does not replace the reference documentation, the policies in the repository root or the model cards.

## Contents

1. [About the project](#1-about-the-project)
2. [Installation](#2-installation)
3. [Data and processing](#3-data-and-processing)
4. [Models and the model zoo](#4-models-and-the-model-zoo)
5. [Command line and REST service](#5-command-line-and-rest-service)
6. [Troubleshooting](#6-troubleshooting)
7. [Security, privacy and support](#7-security-privacy-and-support)
8. [References](#references)

## 1. About the project

### 1.1 What is Unbihexium?

An open source Python library for Earth observation, geospatial analysis, remote sensing and synthetic aperture radar (SAR). It combines raster and vector input and output (GeoTIFF and Cloud Optimized GeoTIFF, Zarr, GeoJSON, GeoParquet, STAC), radiometric preprocessing, 27 registered spectral indices, SAR calibration, speckle filtering, polarimetry and interferometry, terrain and hydrology, geostatistics, spatial analysis, accuracy metrics and visualisation, a model zoo of 520 model definitions with training, evaluation, tiled inference and ONNX export, the `unbihexium` command and a FastAPI-based REST service. [README.md](../README.md) gives the overview; [docs/index.md](index.md) maps the documentation.

### 1.2 Is it ready for production use?

The classical processing functions are deterministic implementations of published methods and are covered by the test suite, which runs in CI on CPython 3.10 to 3.14. The learned models are not ready: they are untrained starter models (Section 4.1). The project makes no claim of fitness for any operational, safety-critical or legal purpose, has one maintainer and offers no service level. Read [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) before deploying anything whose output affects people, property or the environment.

### 1.3 Which version should I use?

The latest release on PyPI is 1.0.1 (tag `v1.0.1`, 21 December 2025). The main branch has changed substantially since then and has not yet been released; the next release will be 2.0.0 because the changes include breaking ones. The documentation under `docs/` describes the main branch. To use it, install from source ([installation guide](getting_started/installation.md)); to move existing 1.0.x code, read [MIGRATION.md](MIGRATION.md). An installation from the main branch still reports version 1.0.1 ([VERSIONING.md](../VERSIONING.md), Section 2.4), so record the commit hash with your results.

### 1.4 Can I use it in proprietary software?

Unbihexium is licensed under the Mozilla Public License 2.0 [1], a file-level copyleft licence. You may combine it with code under other licences, including proprietary code; if you distribute modified versions of Unbihexium's own files, those files must remain under the MPL-2.0 and their source must be made available. Releases v1.0.0 and v1.0.1 were published under Apache-2.0. See [LICENSE.txt](../LICENSE.txt), [NOTICE.md](../NOTICE.md) and [COMPLIANCE.md](../COMPLIANCE.md). This answer is not legal advice.

### 1.5 How do I cite it?

Cite the version you used as described in [CITATION.cff](../CITATION.cff) and [CITATION.md](../CITATION.md). The project has no DOI at present.

## 2. Installation

### 2.1 Which Python versions and platforms are supported?

CPython 3.10, 3.11, 3.12, 3.13 and 3.14 (`requires-python = ">=3.10"`). The CI test matrix runs all five versions on Linux (`ubuntu-latest`). The dependencies publish binary wheels for macOS and Windows as well, but those platforms are not tested in CI. Support ends for a Python version in the first minor release after its upstream end of life ([VERSIONING.md](../VERSIONING.md), Section 7).

### 2.2 Do I need GDAL or a compiler?

No. The binary wheels of rasterio, pyproj, Shapely and ONNX Runtime bundle GDAL, PROJ, GEOS and their native libraries.

### 2.3 Which extra do I need?

| Goal | Install |
| --- | --- |
| Input and output, preprocessing, indices, SAR, terrain, geostatistics, analysis, metrics, catalogue browsing, CLI | `unbihexium` |
| Build, train, evaluate or export models | `unbihexium[torch]` |
| Run exported ONNX models without PyTorch | `unbihexium[onnx]` |
| REST service | `unbihexium[serving]` |
| Zarr, GeoParquet | `unbihexium[zarr]`, `[parquet]` (STAC search needs no extra) |
| Everything | `unbihexium[all]` |

The full table of extras is in [README.md, Section 3.2](../README.md#32-from-pypi).

### 2.4 How do I use a GPU?

Install the CUDA build of PyTorch that matches your driver first, following the PyTorch installation instructions [2], then install `unbihexium[torch]`; pip keeps the CUDA build that is already installed. Training uses the GPU with `--device auto` (the default of `unbihexium train`) or `--device cuda`; `--amp` enables mixed precision on CUDA. Prediction defaults to `--device cpu`; pass `--device cuda` to `unbihexium predict`. Apple silicon is addressed with `--device mps`.

### 2.5 Is there a conda package?

The project publishes the Python package only to PyPI (<https://pypi.org/project/unbihexium/>) and the container image only to the GitHub Container Registry. It does not maintain a conda recipe. In a conda environment, install with `python -m pip install unbihexium`.

### 2.6 Can I install it without network access?

Yes, from pre-downloaded wheels (`pip download` on a connected machine, then `pip install --no-index --find-links`). The hashed lock files `requirements.txt` and `requirements-dev.txt` pin every dependency for CPython 3.10 to 3.14. At run time the library needs no network access: starter models are built locally (Section 4.3), and connections are opened only for STAC API searches, models you register with a URL, and URLs you pass explicitly ([PRIVACY.md](../PRIVACY.md), Section 4).

### 2.7 Is there a container image?

Yes. `.github/workflows/docker.yml` publishes `ghcr.io/unbihexium-oss/unbihexium` with the tags `main`, `<major>.<minor>.<patch>`, `<major>.<minor>` and `sha-<short sha>`; there is no `latest` tag. The image runs as a non-root user, contains the library with the REST service (`unbihexium serve`) but no model weights, and keeps the model store in a volume. Its exact contents and the Compose and Kubernetes set-ups are described in [docs/operations/docker.md](operations/docker.md).

## 3. Data and processing

### 3.1 Which file formats are supported?

GeoTIFF and Cloud Optimized GeoTIFF (read and write, windows, overviews), Zarr version 3, GeoJSON (RFC 7946 validation, reprojection, ring orientation), GeoParquet, and STAC items, catalogues and API searches (`unbihexium.io`). Training datasets may contain GeoTIFF or NumPy `.npy` images. The `netcdf` extra installs netCDF4 and h5py, but the library has no NetCDF reader of its own.

### 3.2 Are band numbers counted from 0 or from 1?

Command line options count from 1, as GDAL does: `unbihexium index NDVI -i scene.tif -o ndvi.tif --red 3 --nir 4`. `read_geotiff(path, bands=[3, 4])` also takes 1-based band numbers. Arrays in Python are indexed from 0, so the third band of the returned array is `data[2]`. The defaults of `unbihexium index` (blue 2, green 3, red 4, NIR 8, SWIR1 12, SWIR2 13) match a full Sentinel-2 stack in the order B01, B02, ..., B08, B8A, B09, B10, B11, B12; for any other band order, give every band the index needs explicitly.

### 3.3 Why does an index contain NaN values?

Where the denominator of an index is zero, the result is NaN instead of an arbitrary large number (`unbihexium.indices.safe_divide`). This is a deliberate change from 1.0.x, which added a small constant. Mask the NaN pixels or fill them with `numpy.nan_to_num` if your workflow requires finite values.

### 3.4 Do the indices need reflectance or digital numbers?

Reflectance. Scale Level-2 products first, for example with `unbihexium.preprocessing.sentinel2_reflectance` or `landsat_c2l2_reflectance`. Indices with additive constants (EVI, SAVI, OSAVI, MSAVI, EVI2) give wrong values on unscaled digital numbers.

### 3.5 How are large rasters handled?

Read parts of a file with `read_geotiff(..., window=...)`, `bounds=...` or `overview_level=...`. Model inference is tiled: `unbihexium predict` and the task APIs split the image into overlapping tiles (default overlap 0.25, tile size from the variant) and blend the results, so an image of any size can be processed within the memory of one batch of tiles. Chunked processing with Dask or Ray is possible through the `dask` and `ray` extras, but the library does not orchestrate it.

### 3.6 How do I search for imagery?

`unbihexium.io.STACClient` and `search_stac` query a STAC API [3] with paging; `walk_catalog` and `read_stac_item` read static catalogues. The library does not include credentials or data access agreements for any provider.

## 4. Models and the model zoo

### 4.1 Are the models trained?

No. The 520 models of the zoo (130 families in the variants tiny, base, large and mega) are untrained starter models: complete, trainable architectures with deterministic starter weights. Only the 28 models of the 7 spectral index families (`ndvi_calculator`, `evi_calculator`, `savi_calculator`, `ndwi_calculator`, `nbr_calculator`, `msi_calculator`, `vegetation_condition`) compute exact formulas and give meaningful output without training. `unbihexium zoo info <model_id>` reports `"requires_training": true` for every learned model, and the REST service returns the same flag with each prediction.

### 4.2 Why does a detector find nothing, or a segmenter produce nonsense?

Because it is a starter model whose weights have never seen Earth observation data. This is expected and is not a bug. Train or fine-tune the model on labelled data for your sensor and area of interest first ([docs/model_zoo/training.md](model_zoo/training.md)).

### 4.3 Where do the weights come from? Is anything downloaded?

Nothing is downloaded for catalogue models, whose entries have no download URL. `unbihexium zoo build <model_id>` generates the starter weights locally and deterministically from the model id, compares their digest with the published value in `src/unbihexium/zoo/digests.json` and writes the model to the local store; `unbihexium.zoo.load_model` and the task APIs build the same weights in memory with the same check. A download happens only for a model that you register yourself with a URL (`unbihexium.zoo.register_model`). Building `water_surface_detector_tiny` took 3.7 s of wall-clock time including interpreter start-up (measured with `time` on a 4 vCPU container, CPython 3.13, CPU only).

### 4.4 Where are models stored?

In `$UNBIHEXIUM_CACHE/models/<model_id>/`, by default `~/.cache/unbihexium/models/<model_id>/`, with the files `model.pt`, `config.json` and `model.sha256`. `unbihexium zoo where <model_id>` prints the checkpoint path and `unbihexium zoo clear [<model_id>] --yes` removes cached models. Training writes to the directory given by `--output` (default `runs/<model_id>/`).

### 4.5 What is the difference between the variants?

| Variant | Base channels | Encoder levels | Blocks per level | Tile size |
| --- | --- | --- | --- | --- |
| tiny | 16 | 3 | 1 | 256 px |
| base | 32 | 4 | 1 | 256 px |
| large | 48 | 4 | 2 | 512 px |
| mega | 64 | 5 | 2 | 512 px |

Larger variants have more parameters (for example 733,090 for `water_surface_detector_tiny`) and need more memory and training data. No accuracy figures are published for any variant, because no trained weights exist to measure. Task APIs default to `base`; use `tiny` for quick experiments.

### 4.6 How do I train a model on my own data?

Arrange images and labels in the dataset folder layout described in [docs/model_zoo/training.md](model_zoo/training.md) and run `unbihexium train <model_id> --data <folder>`; evaluate with `unbihexium evaluate <checkpoint> --data <folder> --split test`. `--synthetic N` trains on generated data to check the setup without any data; a model trained that way is not useful on real imagery. The [tutorials](tutorials/index.md) contain a complete run.

### 4.7 Can I run models without PyTorch?

Yes, after exporting them: `unbihexium zoo export <checkpoint> model.onnx` (needs the `torch` extra once) writes an ONNX [4] file that carries its configuration and normalisation statistics and is compared with PyTorch in ONNX Runtime. `unbihexium predict model.onnx input.tif output.tif --backend onnx` then needs only the `onnx` extra.

### 4.8 Are published accuracy numbers available?

No. There are no trained weights, so there is nothing to measure. Any accuracy figure you see for these models from another source does not come from this project.

## 5. Command line and REST service

### 5.1 Where is `unbihexium infer`?

It was replaced by `unbihexium predict`, and `unbihexium zoo download` by `unbihexium zoo build`. Both old names remain as hidden aliases for 1.0.x scripts but do not appear in `--help`. See [MIGRATION.md](MIGRATION.md) and [docs/reference/cli.md](reference/cli.md).

### 5.2 How do I secure the REST service?

By default no API key is set, no rate limit applies and all CORS origins are allowed, which is suitable only for a trusted network. Before exposing the service, set at least `UNBIHEXIUM_SERVING__API_KEY` (clients then send the `X-API-Key` header on every route except `/health`) and `UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE`, restrict `UNBIHEXIUM_SERVING__CORS_ORIGINS`, and terminate TLS in a reverse proxy. The service rejects bodies above 10 MiB (413), images above 4,194,304 pixels or 16,777,216 values, unsupported media types (415), unknown models (404) and invalid input (422); these limits are configurable ([docs/getting_started/configuration.md](getting_started/configuration.md)). See also [SECURITY.md](../SECURITY.md), Section 8.

### 5.3 What is the difference between `examples/serving/api.py` and `unbihexium.serving`?

`unbihexium.serving` is the maintained REST service, with limits, API keys and rate limiting. `examples/serving/api.py` is a short teaching example without authentication or upload limits that returns internal error messages; it must not be exposed to untrusted clients. See [examples/README.md](../examples/README.md).

## 6. Troubleshooting

### 6.1 "install PyTorch with pip install 'unbihexium[torch]'"

Building, training, evaluating, exporting and running checkpoints need PyTorch. Install the `torch` extra, or export the model to ONNX on another machine and use `--backend onnx`.

### 6.2 "expects 3 bands (red, green, blue), got shape (4, ...)"

Every model expects the bands listed by `unbihexium zoo info <model_id>` (`bands`), in that order. Select and reorder the bands before prediction, for example with `read_geotiff(path, bands=[3, 2, 1])` and `write_geotiff`.

### 6.3 "unknown model ..."

Model ids have the form `<family>_<variant>`, for example `water_surface_detector_tiny`. List the valid ids with `unbihexium zoo list`, filtered with `--task`, `--domain` and `--variant`.

### 6.4 `unbihexium zoo verify` fails

The checkpoint in the store no longer matches the published weights digest, for example because it was modified or partly written. Rebuild it with `unbihexium zoo build <model_id> --force`. `zoo verify` checks the weights digest stored in and computed from the checkpoint; it does not verify trained checkpoints, which have no published digest.

### 6.5 Results differ between machines

Classical functions are deterministic. Starter weights are platform independent and verified by digest. Training on the CPU with the same seed gave identical results in repeated runs during the preparation of the tutorials, but GPU kernels and different library versions can change results at the level of floating-point rounding; pin the environment with the lock files and record the commit, catalogue version and checkpoint digest ([VERSIONING.md](../VERSIONING.md), Section 6.3).

## 7. Security, privacy and support

### 7.1 How do I report a vulnerability?

Privately, through a GitHub security advisory (<https://github.com/unbihexium-oss/unbihexium/security/advisories/new>) or by e-mail to <yunus.z.imanov@helsinki.fi>, never in a public issue. The process is defined in [SECURITY.md](../SECURITY.md).

### 7.2 How can I verify a release?

Releases built by the current `.github/workflows/release.yml` carry `SHA256SUMS.txt`, a Sigstore bundle (`.sigstore.json`) for each distribution [5], the SLSA provenance `unbihexium-<tag>.intoto.jsonl` [6] and GitHub artifact attestations. Releases v1.0.0 and v1.0.1 predate signing and have none of these. The verification commands are in [SECURITY.md](../SECURITY.md), Section 7.

### 7.3 Does the library send telemetry?

No. It contains no telemetry, analytics or update checks ([PRIVACY.md](../PRIVACY.md)).

### 7.4 Where do I ask questions?

As described in [SUPPORT.md](../SUPPORT.md): issues for bugs and feature requests on <https://github.com/unbihexium-oss/unbihexium/issues>, after checking the documentation, the model cards and existing issues. Contributions are welcome under [CONTRIBUTING.md](../CONTRIBUTING.md).

## References

[1] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[2] PyTorch Foundation. PyTorch: Get started locally. 2026. <https://pytorch.org/get-started/locally/>

[3] STAC contributors. SpatioTemporal Asset Catalog specification. 2025. <https://github.com/radiantearth/stac-spec>

[4] ONNX project. Open Neural Network Exchange. 2026. <https://onnx.ai/>

[5] Sigstore project. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

[6] OpenSSF. Supply-chain Levels for Software Artifacts (SLSA), specification version 1.0. 2023. <https://slsa.dev/spec/v1.0/>

<!--
=============================================================================
End of file docs/faq.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
