<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/MIGRATION.md
Title       : Migration Guide from 1.0.x to 2.0.0
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Migration Guide from 1.0.x to 2.0.0

| Field | Value |
| --- | --- |
| Document | UBX-DOC-308 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Upgrades from Unbihexium 1.0.0 and 1.0.1 to Unbihexium 2.0.0 |

## Abstract

This guide explains how to move code, command line scripts, REST clients, model files and deployments from Unbihexium 1.0.0 or 1.0.1 to version 2.0.0, a major release because it contains breaking changes ([VERSIONING.md](../VERSIONING.md), Section 4). It is written for users who maintain code against the 1.0.x API, for operators of the command line and the REST service, and for packagers. Every change listed here was derived from the 2.0.0 section of [CHANGELOG.md](../CHANGELOG.md) and checked against the source code of the tags `v1.0.1` and `v2.0.0`; for each breaking change the guide states what changed and what to do. All code examples on the new side were executed against the main branch on 24 September 2026. The guide does not repeat the complete list of additions, which is in the changelog.

## Contents

1. [Scope and conventions](#1-scope-and-conventions)
2. [Summary of breaking changes](#2-summary-of-breaking-changes)
3. [Preparing the upgrade](#3-preparing-the-upgrade)
4. [Licence, Python versions and packaging](#4-licence-python-versions-and-packaging)
5. [Python API](#5-python-api)
6. [Command line](#6-command-line)
7. [REST service](#7-rest-service)
8. [Model zoo and model files](#8-model-zoo-and-model-files)
9. [Container image and deployment](#9-container-image-and-deployment)
10. [Upgrade checklist](#10-upgrade-checklist)
11. [References](#references)

## 1. Scope and conventions

### 1.1 Versions

| Version | Status | Tag and date | Distribution |
| --- | --- | --- | --- |
| 1.0.0 | Released | `v1.0.0`, 2025-12-21 | GitHub release only; Apache-2.0 |
| 1.0.1 | Released | `v1.0.1`, 2025-12-21 | PyPI and GitHub; Apache-2.0 |
| 2.0.0 | Released, latest | `v2.0.0`, 2026-09-24 | PyPI, GitHub and the container registry; MPL-2.0 |

No version older than 1.0.0 was ever released. `unbihexium --version` prints the installed version, so it tells the two sides of this migration apart. Install 2.0.0 from PyPI with `python -m pip install "unbihexium==2.0.0"` or as described in [docs/getting_started/installation.md](getting_started/installation.md).

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals. "1.0.x" means both 1.0.0 and 1.0.1, whose Python API is identical. Paths of Python modules are relative to `src/unbihexium/`.

### 1.3 What did not work in 1.0.x

Some 1.0.x interfaces never worked, so code that uses them has nothing to preserve. `import unbihexium.ai` (and therefore every `unbihexium.ai.*` import) failed, because a module and a package were both named `super_resolution`; `unbihexium.analysis.network` failed for the same reason; `unbihexium index` read its input without computing or writing the index; `from unbihexium.cli import cli` failed; and the detection, segmentation and super-resolution classes returned empty or interpolated placeholder results ([CHANGELOG.md](../CHANGELOG.md), Section 2.4). Treat code in these areas as new code written against the current API.

## 2. Summary of breaking changes

| Area | Change | Action |
| --- | --- | --- |
| Licence | Apache-2.0 to MPL-2.0 | Review licence obligations (Section 4.1) |
| `io.read_geotiff` | `transform` is a tuple of six coefficients, not an `Affine` | Wrap with `Affine(*transform)` where an `Affine` is needed (Section 5.1) |
| `io.write_geotiff` | Data before path; DEFLATE and tiling by default | Swap arguments; pass `compress="lzw"` for the old compression |
| Spectral indices | Zero denominator gives NaN instead of a large finite value | Handle NaN (Section 5.2) |
| SAR | Incidence angles in degrees by default | Pass degrees, or `degrees=False` (Section 5.3) |
| Terrain | `aspect` is a compass bearing; `hillshade` returns `float64` | Update consumers (Section 5.4) |
| Metrics | `ssim` uses a Gaussian window; `window_size` removed | Use `sigma` and `data_range` (Section 5.5) |
| Core | Pipeline steps return mappings; step names unique; new `Evidence` and `ProvenanceRecord` fields | Section 5.6 |
| Configuration | Defaults `variant="base"`, `device="cpu"`; strict layered loading | Set values explicitly (Section 5.7) |
| Task APIs | New constructor (`model`, `variant`, `weights`, ...); `overlap` is a fraction | Section 5.8 |
| Model zoo API | `ModelCache` and the `cache` and `downloader` modules removed; new store layout | Section 5.9 |
| Command line | `index` band options are band numbers; `pipeline run --config` removed; `zoo download --version` removed | Section 6 |
| Model files | ONNX files under `model_zoo/assets/` removed; models are built locally | Section 8 |
| Container | Python 3.14 base, non-root user, read-only root file system in Compose | Section 9 |

## 3. Preparing the upgrade

1. Pin the current environment (`python -m pip freeze > before.txt`) and keep it until the migration is verified.
2. Create a new virtual environment with CPython 3.10 to 3.14 and install the main branch from source with the extras you need, for example `python -m pip install -e ".[torch,onnx,serving]"`. For a reproducible environment, install the hashed lock file first (`python -m pip install --require-hashes -r requirements-dev.txt`, then `python -m pip install --no-deps -e .`).
3. Run your own tests with warnings shown (`python -W default -m pytest`), then work through Sections 4 to 9.
4. Record with every result the commit hash, the catalogue version (`unbihexium info`) and, for trained models, the checkpoint digest ([VERSIONING.md](../VERSIONING.md), Section 6.3).

Users SHOULD migrate in one step to the main branch; there are no intermediate releases.

## 4. Licence, Python versions and packaging

### 4.1 Licence

The project was relicensed from Apache-2.0 to the Mozilla Public License 2.0 [3] (#20). Versions 1.0.0 and 1.0.1 remain available under Apache-2.0. The MPL-2.0 is a file-level copyleft licence: if you distribute modified Unbihexium files, you MUST make those files available under the MPL-2.0; your own files that merely use the library may keep their own licence. The licence text is in [LICENSE.txt](../LICENSE.txt) and attribution notices are in [NOTICE.md](../NOTICE.md). This summary is not legal advice; see [COMPLIANCE.md](../COMPLIANCE.md).

### 4.2 Python versions

1.0.x declared CPython 3.10 to 3.12. The main branch declares and tests CPython 3.10 to 3.14 (#21). No Python version was dropped.

### 4.3 Extras and dependencies

| Extra | 1.0.x | main branch |
| --- | --- | --- |
| `onnx` | did not exist | onnxruntime, onnx |
| `torch` | torch, torchvision | torch, onnx (torchvision removed, because no code used it) |
| `serving` | did not exist (FastAPI was not declared) | fastapi, starlette, uvicorn |
| `test` | did not exist | pytest and plugins, httpx |
| `docs` | existed | removed |
| `dev` | pytest, ruff, pyright, pre-commit, bandit, pip-audit | `test` extra, ruff, pyright, scipy-stubs (Python 3.12 and newer), pre-commit, bandit, pip-audit, build, twine, tox |
| `all` | every extra except `gpu` (including `docs` and `dev`) | every extra except `gpu` |

All lower bounds of the dependencies were raised to releases that provide wheels for CPython 3.10 to 3.14 and exclude releases with known vulnerabilities (#33). Environments that pin old versions of NumPy, SciPy, rasterio, pydantic, Pillow or other dependencies MUST be updated; the exact bounds are in `pyproject.toml`. Deployments of the REST service MUST now install the `serving` extra, and deployments that run exported models without PyTorch MUST install the `onnx` extra.

## 5. Python API

The examples of this section stand for code that reads the user's own files. To run them as written, create the two synthetic scenes they use: `scene.tif`, the 6-band test scene of the [tutorials](tutorials/index.md) (blue, green, red, NIR, SWIR 1.6 um and SWIR 2.2 um with 20 m pixels in EPSG:32635), and `harbour.tif`, a 3-band red, green and blue scene.

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import write_geotiff

# Reflectance of water, vegetation and bare soil, in three strips of 30 columns.
table = np.array([[0.06, 0.05, 0.03, 0.02, 0.01, 0.01],
                  [0.03, 0.06, 0.04, 0.40, 0.20, 0.10],
                  [0.12, 0.16, 0.20, 0.26, 0.32, 0.28]], dtype="float32")
cover = np.repeat([0, 1, 2], 30)[None, :].repeat(60, axis=0)
rng = np.random.default_rng(42)
scene = table[cover].transpose(2, 0, 1) + rng.normal(0, 0.005, (6, 60, 90)).astype("float32")
transform = from_origin(500000, 6700000, 20, 20)
write_geotiff(np.clip(scene, 0.0, 1.0), "scene.tif", crs="EPSG:32635", transform=transform,
              descriptions=["blue", "green", "red", "nir", "swir16", "swir22"])
write_geotiff(rng.uniform(0, 0.3, (3, 256, 256)).astype("float32"), "harbour.tif",
              crs="EPSG:32635", transform=transform)
```

### 5.1 GeoTIFF input and output

`read_geotiff` still returns `(data, metadata)` with the CRS as a string, but `metadata["transform"]` is now a tuple of six affine coefficients `(a, b, c, d, e, f)` instead of a rasterio `Affine`, and the metadata has additional keys (`bounds` as a plain tuple, `band_count`, `descriptions`, `overviews`, `compression`, `tiled`, `block_shape`, `is_cog`, `driver` and others). `write_geotiff` now takes the data first and the path second; the 1.0.x order `write_geotiff(path, data)` is still accepted. The default compression changed from LZW to DEFLATE and the output is tiled by default.

```python
from affine import Affine

from unbihexium.io import read_geotiff, write_geotiff

data, meta = read_geotiff("scene.tif")
print(meta["crs"], meta["transform"])          # CRS string and six coefficients.
transform = Affine(*meta["transform"])         # Rebuild the rasterio Affine where needed.
print(transform.c, transform.f)

# Data first, then the path; request LZW to keep the 1.0.x compression.
write_geotiff(data, "copy.tif", crs=meta["crs"], transform=meta["transform"], compress="lzw")
```

Output for the 6-band test scene of the [tutorials](tutorials/index.md):

```text
EPSG:32635 (20.0, 0.0, 500000.0, 0.0, -20.0, 6700000.0)
500000.0 6700000.0
```

### 5.2 Spectral indices

In 1.0.x the six registered indices (NDVI, NDWI, NBR, EVI, SAVI, MSI) replaced a zero denominator by `1e-10`, which turned 0/0 into 0 and x/0 into a very large number. All indices now return NaN where the denominator is zero (`indices.safe_divide`). `core.compute_index(name, bands)` keeps its 1.0.x form and gains the optional `sensor` and `nodata` arguments and further parameters; the registry now has 27 indices. The package `unbihexium.indices`, empty in 1.0.x, provides keyword functions such as `ndvi(nir=..., red=...)`.

```python
import numpy as np

from unbihexium.indices import ndvi

values = ndvi(nir=np.array([0.0, 0.4]), red=np.array([0.0, 0.1]))
print(values)                                  # A zero denominator gives NaN.
print(np.nan_to_num(values, nan=0.0))          # Replace NaN where finite values are required.
```

```text
[nan 0.6]
[0.  0.6]
```

Code that sums or averages index rasters SHOULD use NaN-aware functions (`numpy.nanmean` and similar).

### 5.3 SAR incidence angles

In 1.0.x, `compute_sigma0` and `compute_gamma0` expected the incidence angle in radians. They now expect degrees by default and take `degrees=False` for radians. Calling the new functions with radians and without `degrees=False` gives silently wrong values, so every call MUST be checked.

```python
import numpy as np

from unbihexium.sar import compute_gamma0, compute_sigma0

amplitude = np.full((2, 2), 400.0)
theta_deg = 35.0

sigma0 = compute_sigma0(amplitude, incidence_angle=theta_deg, calibration_lut=500.0**2)
same = compute_sigma0(amplitude, incidence_angle=np.deg2rad(theta_deg), calibration_lut=500.0**2, degrees=False)
print(np.allclose(sigma0, same), round(float(sigma0[0, 0]), 4))
print(round(float(compute_gamma0(sigma0, theta_deg)[0, 0]), 4))
```

```text
True 0.3671
0.4481
```

The speckle filters are now also available as separate functions (`lee_filter`, `refined_lee_filter`, `enhanced_lee_filter`, `frost_filter`, `kuan_filter`, `gamma_map_filter`); `speckle_filter(data, filter_type="lee", window_size=5, looks=1.0)` remains as the dispatcher and gained the `looks` argument. `amplitude_to_db` keeps its -40 dB floor but returns `float64` instead of `float32`; the new `power_to_db` returns NaN for non-positive input unless a `floor` is given.

### 5.4 Terrain

- `aspect` now returns the compass bearing of the steepest descent, in degrees clockwise from north. The 1.0.x value was computed as `degrees(arctan2(-dy, dx))` wrapped to [0, 360) and was not a compass bearing. Results computed with 1.0.x MUST NOT be compared with new results without recomputing them.
- `hillshade` returns `float64` values in [0, 255] with NaN at no-data, instead of `uint8`. It gained a `z_factor` argument.
- `slope`, `aspect`, `hillshade` and `curvature` accept `resolution` as a number or as a `(x, y)` pair and default to 1.0; `slope` gained `units` and `z_factor`.

```python
import numpy as np

from unbihexium.metrics import ssim
from unbihexium.terrain import aspect, hillshade

y, x = np.mgrid[0:50, 0:50]
dem = 100.0 + 2.0 * x                          # Terrain rising towards the east.
print(aspect(dem, resolution=30.0)[25, 25])    # Faces west: compass bearing 270.

shade = hillshade(dem, resolution=30.0)        # float64 in [0, 255], NaN at no-data.
shade_u8 = np.nan_to_num(shade).round().astype(np.uint8)   # The 1.0.x output type.
print(shade.dtype, shade_u8.dtype)

rng = np.random.default_rng(0)
a = rng.random((64, 64))
print(round(ssim(a, a), 3), round(ssim(a, np.clip(a + 0.05, 0, 1), data_range=1.0, sigma=1.5), 3))
```

```text
270.0
float64 uint8
1.0 0.995
```

### 5.5 Metrics

`metrics.ssim(pred, target, window_size=11)` used an 11 by 11 uniform window. The new signature is `ssim(pred, target, data_range=1.0, sigma=1.5)` with the Gaussian window of Wang et al. [4] (see the example above). Calls that pass `window_size` fail with `TypeError`; SSIM values differ from 1.0.x. The metrics package was extended (area estimation, change metrics, regression metrics, SAM, ERGAS, Q index); existing names such as `confusion_matrix`, `cohen_kappa`, `iou` and `psnr` keep their meaning.

### 5.6 Pipelines, evidence and provenance

- Pipeline steps MUST return a mapping; the mapping is passed to the next step. In 1.0.x any return value was accepted.
- `Pipeline.add_step(step, name=None)` raises `ValueError` if a step name is used twice and `TypeError` if the step is not callable.
- `PipelineConfig` gained `seed` and `deterministic`; a run seeds the random generators and records a `ProvenanceRecord`.
- `Evidence` fields changed: `source` is the first field, `checksum`, `evidence_type` and `evidence_id` have defaults, `created_at` (a `datetime`) was replaced by `timestamp` (an ISO 8601 string), and `description` and `size_bytes` were added. Positional construction in the 1.0.x order MUST be rewritten with keywords or with `Evidence.from_file`.
- `ProvenanceRecord` takes `run_id` and `pipeline_id` first, generates `record_id`, stores `created_at` as an ISO 8601 string and gained `model_ids`.

```python
import numpy as np

from unbihexium.core import Evidence, EvidenceType, Pipeline, PipelineConfig
from unbihexium.indices import ndvi


def compute_ndvi(values):
    # Every step receives and returns a mapping.
    return {"ndvi": ndvi(nir=values["nir"], red=values["red"])}


def summarise(values):
    return {**values, "mean": float(np.nanmean(values["ndvi"]))}


pipeline = Pipeline(PipelineConfig(pipeline_id="ndvi_mean", name="NDVI mean", seed=0))
pipeline.add_step(compute_ndvi).add_step(summarise)     # Step names must be unique.
run = pipeline.run({"nir": np.array([0.4, 0.5]), "red": np.array([0.1, 0.1])})
print(run.status.value, run.provenance.pipeline_id, isinstance(run.provenance.created_at, str))

evidence = Evidence.from_file("scene.tif", EvidenceType.INPUT)   # Keyword or factory, not positional.
print(evidence.checksum[:16], evidence.timestamp[:4], evidence.verify("scene.tif"))
```

```text
completed ndvi_mean True
df750ac0baf6b63b 2026 True
```

### 5.7 Configuration

`unbihexium.config` keeps `Config`, `ModelConfig` and `get_default_config`, with these changes:

| Setting | 1.0.x default | New default |
| --- | --- | --- |
| `model.variant` | `large` | `base` |
| `model.device` | `cuda:0` | `cpu` |
| `model.backend` | did not exist | `auto` |
| `processing` section (`ProcessingConfig`) and `model.num_workers` | tile size, overlap, output format and training workers, read by nothing | removed; setting them is an error that names the key |
| `serving` section | did not exist | host, port, request limits, API key, CORS origins, rate limit, model cache size |
| `log_level` | did not exist | `WARNING` |

Settings are now loaded in layers by `load_config`: defaults, a YAML file (argument or `UNBIHEXIUM_CONFIG`), environment variables `UNBIHEXIUM_<SECTION>__<KEY>`, then explicit overrides. Unknown keys are errors, whereas 1.0.x ignored them in `Config.update`. `to_dict()` and `to_yaml()` redact `serving.api_key` unless `include_secrets=True` is passed. Code that relied on the old defaults MUST set them explicitly:

```python
from unbihexium.config import load_config

config = load_config(overrides={"model": {"variant": "large", "device": "cuda:0"}})
print(config.model.variant, config.model.device)
print(load_config(env=False).model.variant, load_config(env=False).model.device)
```

```text
large cuda:0
base cpu
```

The environment variables are listed in [docs/getting_started/configuration.md](getting_started/configuration.md) and [.env.example](../.env.example).

### 5.8 Task APIs in unbihexium.ai

The task classes run model zoo models ([docs/model_zoo/inference.md](model_zoo/inference.md)). Their constructors changed from `ObjectDetector(model_id, class_names, threshold, tile_size=512, overlap=64)` to `ObjectDetector(model=None, threshold=0.5, iou_threshold=0.5, max_detections=1000, variant=None, weights=None, device="cpu", backend="auto", tile_size=None, overlap=0.25, batch_size=4)`:

- `model` accepts a family name, a model id, a checkpoint, an ONNX file or a loaded model; `weights` takes a trained checkpoint or ONNX file and has precedence.
- The variant defaults to `base`; `SuperResolution` uses the catalogue factor 4 unless `scale_factor` is given.
- `overlap` is a fraction of the tile size (0 to 0.9), not a number of pixels.
- `predict` accepts a `Raster`, an array or a file path. Detections carry `geo_bbox` in map coordinates in addition to the pixel `bbox`; results export GeoJSON and GeoTIFF.
- `CropDetector` and `GreenhouseDetector` are detectors, as in the catalogue, and remain importable from `unbihexium.ai.segmentation`.
- The placeholder model classes in `unbihexium.ai.models`, `unbihexium.ai.change_detection`, `unbihexium.ai.super_resolution` and `unbihexium.ai.synthesis` were removed.

```python
from unbihexium.ai import ShipDetector
from unbihexium.zoo import clear_cache, download_model, get_cache_dir, list_cached, verify_model

# 1.0.x constructor arguments that remain valid: threshold. New: model, variant, weights,
# device, backend, tile_size, overlap (a fraction) and batch_size.
detector = ShipDetector(variant="tiny", threshold=0.5, tile_size=256, overlap=0.25)
result = detector.predict("harbour.tif")       # A path, a Raster or an array.
print(result.model_id, result.count)

# ModelCache is gone; the store is managed with functions.
path = download_model("ship_detector_tiny")    # Builds the starter model locally.
print(path.relative_to(get_cache_dir()).as_posix(), verify_model("ship_detector_tiny"), list_cached())
print(clear_cache("ship_detector_tiny"))
```

```text
ship_detector_tiny 0
ship_detector_tiny/model.pt True ['ship_detector_tiny']
1
```

`harbour.tif` stands for any 3-band red, green, blue GeoTIFF. The detector is an untrained starter model, so finding no ships is the expected result; see Section 8.

### 5.9 Model zoo API

- Removed: `ModelCache` and the modules `zoo.cache` and `zoo.downloader`. Use `get_cache_dir`, `model_dir`, `list_cached`, `is_model_cached`, `get_cached_model_path`, `ensure_model`, `download_model` (which now builds a catalogue model locally) and `clear_cache`.
- `list_models` and `get_model` return `ModelZooEntry` objects whose `spec` holds the family metadata (bands, outputs, units, task, domain) and whose `variant`, `weights_digest`, `num_parameters` and `version` describe the model. `list_models` filters by `task`, `domain` and `variant`.
- New: `load_model` builds or loads a model, `verify_model` checks the weights digest, `catalog_version` returns the catalogue version, and `parse_model_id` splits a model id into family and variant.

## 6. Command line

| 1.0.x command | New command | Notes |
| --- | --- | --- |
| `unbihexium infer MODEL_ID -i IN -o OUT [-t TASK]` | `unbihexium predict MODEL INPUT OUTPUT` | `infer` is a hidden alias; `-t` is ignored because the task follows from the model |
| `unbihexium zoo download MODEL_ID [-V VERSION] [--cache-dir DIR]` | `unbihexium zoo build MODEL_ID [--cache-dir DIR] [--force] [--onnx]` | `download` is a hidden alias without `-V`; model ids carry no version |
| `unbihexium zoo list [-t TASK] [--json]` | same, plus `--domain` and `--variant` | output lists the 520 catalogue models |
| `unbihexium zoo verify`, `zoo where` | unchanged names | now verify the weights digest and print the checkpoint path of the new store layout |
| `unbihexium index NAME -i IN -o OUT --red B04 --nir B08 ...` | `unbihexium index NAME -i IN -o OUT --red 4 --nir 8 ...` | band options are 1-based band numbers, not band names; the command now computes and writes the index |
| `unbihexium pipeline run ID -i IN -o OUT -c CONFIG` | `unbihexium pipeline run ID -i IN -o OUT [--input2 IN2] [-p KEY=VALUE ...]` | `--config` was removed; pass parameters with `-p` |
| none | `unbihexium train`, `evaluate`, `serve`, `zoo info`, `zoo export`, `zoo clear` | new commands |

Scripts SHOULD move to the new names, because hidden aliases are not part of the public interface ([VERSIONING.md](../VERSIONING.md), Section 3). The following commands were run to check the aliases:

```bash
unbihexium infer ship_detector_tiny -i harbour.tif -o ships_alias.geojson
unbihexium zoo download ship_detector_tiny
unbihexium predict ship_detector_tiny harbour.tif ships.geojson
```

The full reference is [docs/reference/cli.md](reference/cli.md).

## 7. REST service

- `POST /predict/{model_id}` is the new route for every zoo model. It takes an image as nested lists `(bands, rows, columns)` or as a base64-encoded `.npy` file, optional `crs`, `transform` and `nodata`, and `parameters`; the response includes `requires_training`.
- `POST /infer/{model_id}`, `/detect/{model_id}` and `/segment/{model_id}` remain with their 1.0.x request bodies (`data` for `/infer`, `image_data` and `threshold` for the others) and are documented as the earlier API. `/infer` accepts single-band images only, and `threshold` must now lie in [0, 1]. New clients SHOULD use `/predict`.
- `GET /models` gained filters (`task`, `domain`, `variant`) and pagination (`limit`, `offset`); `GET /models/{model_id}`, `GET /capabilities/{capability_id}` and `GET /pipelines` were added.
- Requests above the configured limits are rejected (413 for the body size, 415 for unsupported media types, 422 for invalid input). If `UNBIHEXIUM_SERVING__API_KEY` is set, every route except `/health` requires the `X-API-Key` header, and a rate limit answers with 429.
- The service needs the `serving` extra and can be started with `unbihexium serve` as well as with `uvicorn unbihexium.serving.app:app`.

See [README.md, Section 8](../README.md#8-rest-service) and [docs/reference/api.md](reference/api.md).

## 8. Model zoo and model files

The 520 ONNX files of 1.0.x, stored with Git LFS under `model_zoo/assets/`, and the 520 per-variant model cards with metrics were removed, because they were not the result of training or evaluation on real data (#37). Scripts that read files from `model_zoo/assets/` or that fetch them with Git LFS MUST be changed; the directory no longer exists.

The model zoo is now defined by `src/unbihexium/zoo/catalog.yaml` (catalogue version 2.0.0) with 130 families in the variants tiny, base, large and mega. Starter weights are built locally and deterministically and verified against `src/unbihexium/zoo/digests.json`. **These are untrained starter models**: apart from the 28 models of the 7 spectral index families, which compute exact formulas, their predictions are not meaningful until you train them ([docs/model_zoo/training.md](model_zoo/training.md)). Model ids follow `<family>_<variant>`; check each id used by your code with `unbihexium zoo info <model_id>`, and list the valid ids with `unbihexium zoo list`.

The store keeps its root, `$UNBIHEXIUM_CACHE/models` (default `~/.cache/unbihexium/models`), but the layout changed from one file `<model_id>.pt` per model to a directory `<model_id>/` with `model.pt`, `config.json` and `model.sha256`. The new code does not read the old files and `unbihexium zoo clear` does not remove them; delete the `*.pt` files directly under the `models` directory by hand.

## 9. Container image and deployment

- The image is based on Python 3.14 on Debian 13 (trixie), pinned by digest, and installs its dependencies from hashed lock files. It runs as the non-root user `unbihexium` (UID 1000), has a health check and keeps the model store in the volume `/home/unbihexium/.cache/unbihexium`. It contains no model weights; the default command prints `unbihexium --help`.
- Images are tagged `main`, `<major>.<minor>.<patch>`, `<major>.<minor>` and `sha-<short sha>`; the workflow does not set a `latest` tag. Deployments SHOULD pin a version tag or an image digest.
- The Docker Compose service starts the REST service, uses a read-only root file system, drops all capabilities and sets `no-new-privileges`. Volumes written by the 1.0.x image as root MAY need their ownership changed for the new user.
- The Helm chart and the Kubernetes manifest under `deploy/` use the placeholder host `unbihexium.example.com`, which MUST be replaced.

See [docs/operations/docker.md](operations/docker.md).

## 10. Upgrade checklist

- [ ] Licence obligations of the MPL-2.0 reviewed (Section 4.1).
- [ ] New environment on CPython 3.10 to 3.14 with the required extras (`torch`, `onnx`, `serving`) and raised dependency versions.
- [ ] `read_geotiff` transforms and `write_geotiff` argument order and compression checked.
- [ ] NaN handling added after spectral index computations.
- [ ] SAR incidence angles converted to degrees or `degrees=False` added.
- [ ] Consumers of `aspect`, `hillshade` and `ssim` updated; old results recomputed where compared.
- [ ] Pipeline steps return mappings; `Evidence` and `ProvenanceRecord` built with keywords.
- [ ] Configuration defaults (`variant`, `device`) set explicitly where the old values were relied on.
- [ ] Task API constructors, `overlap` fractions and zoo store functions updated; `ModelCache` removed.
- [ ] Command line scripts moved to `predict`, `zoo build`, numeric `index` band options and `pipeline run -p`.
- [ ] REST clients moved to `/predict/{model_id}`; API key, rate limit and CORS configured.
- [ ] References to `model_zoo/assets/` removed; old `*.pt` files deleted from the store; models trained before use.
- [ ] Container tags pinned; volume ownership and deployment host names updated.

## References

[1] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[4] Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

<!--
=============================================================================
End of file docs/MIGRATION.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
