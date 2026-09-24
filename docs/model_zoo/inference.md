<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/model_zoo/inference.md
Title       : Running Models
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Running Models

| Field | Value |
| --- | --- |
| Document | UBX-DOC-703 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium, model zoo catalogue version 2.0.0 (not part of release 1.0.1) |

## Abstract

This document explains how to run models of the Unbihexium model zoo on imagery of any size. It is written for users who apply a trained model to rasters, for developers who embed inference in their own code or services, and for reviewers who need to know how tiling, blending and post-processing work. It covers the accepted model sources and the choice between the PyTorch and ONNX Runtime backends, the `unbihexium predict` command and its output formats, the task APIs of `unbihexium.ai` and their result objects, the tiled inference algorithm of `Predictor`, inference without PyTorch through ONNX exports, and the pipeline and REST interfaces that build on the same code. Every learned model of the zoo is an untrained starter model, so catalogue models produce meaningful output only after [training](training.md); the 28 spectral index models compute exact formulas and can be used directly.

## Contents

1. [Introduction](#1-introduction)
2. [Model sources and backends](#2-model-sources-and-backends)
3. [Command line](#3-command-line)
4. [Python task APIs](#4-python-task-apis)
5. [Tiled inference](#5-tiled-inference)
6. [ONNX Runtime](#6-onnx-runtime)
7. [Pipelines and the REST service](#7-pipelines-and-the-rest-service)
8. [Limitations and responsible use](#8-limitations-and-responsible-use)
9. [References](#references)

## 1. Introduction

### 1.1 Status of the models

A catalogue model identifier such as `ship_detector_base` runs the model with its deterministic starter weights. These weights have **not** been trained on Earth observation data, and the output is meaningless: it is useful only to test a processing chain. For real results, run a checkpoint or ONNX file produced by [training](training.md). The 7 spectral index families (NDVI, NDWI, EVI, SAVI, MSI, NBR and VCI) have no weights and compute published formulas; they give correct results directly.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

The examples were run on 2026-09-24 in a container with 4 vCPUs, CPython 3.13, PyTorch 2.14 and ONNX Runtime on the CPU. The file `runs/water_surface_detector_tiny/best.pt` is the checkpoint trained in [training.md](training.md) on toy data, and `scene_rgb.tif`, `scene_rgb_2025.tif`, `scene_rgbn.tif` and `scene_red_nir.tif` are synthetic GeoTIFFs of 300 by 200 pixels with 3, 3, 4 and 2 bands of reflectance. Printed values from starter models or toy models only illustrate the format.

### 1.3 Requirements

| Model source | Requirement |
| --- | --- |
| Catalogue model identifier, checkpoint (`.pt`), `ZooModel` | Extra `torch` |
| ONNX export (`.onnx`) | Extra `onnx` only; PyTorch is not needed |

The model zoo is only on the main branch; install from source as described in [README.md](../../README.md) until a release contains it.

## 2. Model sources and backends

### 2.1 Accepted sources

| Source | Example | Backend |
| --- | --- | --- |
| Family name | `ship_detector` (the `base` variant, or the variant given with `variant=` or `--variant`) | PyTorch, starter weights |
| Model identifier | `ship_detector_large` | PyTorch, starter weights |
| Checkpoint | `runs/ship_detector_base/best.pt` | PyTorch |
| ONNX export | `ship_detector.onnx` | ONNX Runtime |
| Model object | `unbihexium.ai.models.build_model("ship_detector", "tiny")` or `unbihexium.zoo.load_model(...)` | PyTorch |

Every source carries a configuration (`unbihexium.zoo.BuildConfig`) with the task, the input channels, the outputs, the units, the tile size and, for trained models, the normalisation statistics. The task APIs and `Predictor` read it, so the caller never has to state the task or normalise the input.

### 2.2 Choice of the backend

The option `backend` (`--backend` on the command line) accepts `auto` (default), `torch` and `onnx`:

1. a model object always runs with PyTorch;
2. a path ending in `.onnx` that exists always runs with ONNX Runtime;
3. for a model identifier or family name, `onnx` builds the model into the local store with an ONNX export (`ensure_model(..., onnx=True)`, which needs PyTorch once) and runs the export with ONNX Runtime;
4. otherwise the model is loaded with PyTorch (`load_model`), which verifies the digest of a catalogue model or the recorded digest of a checkpoint.

The PyTorch backend runs on the device given by `device` (`--device`, default `cpu`), for example `cuda` or `mps`. The ONNX Runtime backend uses the `CPUExecutionProvider`; `unbihexium.ai.inference.OnnxBackend` accepts other providers when it is constructed directly.

## 3. Command line

### 3.1 Synopsis

```text
unbihexium predict [OPTIONS] MODEL INPUT_PATH OUTPUT_PATH
```

`MODEL` is any source of Section 2.1 given as a string; `INPUT_PATH` is a raster that rasterio can read, with exactly the input channels of the model in model order.

| Option | Default | Meaning |
| --- | --- | --- |
| `--second FILE` | none | Image of the second date for change detection models |
| `--variant TEXT` | from the identifier | Variant for family names |
| `--threshold FLOAT` | 0.5 (task API default) | Detection score threshold, or segmentation probability threshold (Section 4.3) |
| `--tile-size INTEGER` | tile size of the model (256 or 512) | Tile size in pixels, rounded up to the size multiple of the model |
| `--overlap FLOAT` | 0.25 | Overlap between neighbouring tiles as a fraction of the tile size |
| `--backend [auto\|torch\|onnx]` | `auto` | Inference backend (Section 2.2) |
| `--device TEXT` | `cpu` | PyTorch device |

### 3.2 Outputs

The output format follows from the task of the model; the file extension of `OUTPUT_PATH` SHOULD match it:

| Task | Output file |
| --- | --- |
| Detection | GeoJSON FeatureCollection with one `Polygon` per box, properties `class_id`, `class_name` and `confidence`, the model identifier and the CRS of the input |
| Segmentation, change detection | Single-band GeoTIFF of class indices (`uint8`) with no-data value 255, used for missing input and, for models with more than two classes, for pixels whose highest probability is below the threshold |
| Dense regression, spectral index | Float32 GeoTIFF with one band per output, NaN for no data |
| Enhancement | Float32 GeoTIFF of the output bands on the input grid |
| Super-resolution | Float32 GeoTIFF on a grid `scale` times finer, covering the same area |
| Scene regression | JSON document with the model identifier, the units and one value per output |

Rasters keep the CRS and the transform of the input; the transform of a super-resolution output is scaled accordingly.

### 3.3 Examples

```bash
unbihexium predict runs/water_surface_detector_tiny/best.pt scene_rgbn.tif water.tif
unbihexium predict ndvi_calculator_tiny scene_red_nir.tif ndvi.tif
unbihexium predict change_detector --variant tiny scene_rgb.tif change.tif --second scene_rgb_2025.tif
unbihexium predict ship_detector_tiny scene_rgb.tif ships.geojson --threshold 0.3 --backend onnx
unbihexium predict economic_spatial_assessor_tiny scene_rgb.tif value.json
unbihexium predict super_resolution_tiny scene_rgb.tif sr.tif
```

```text
Wrote: water.tif (water_surface_detector_tiny)
Wrote: ndvi.tif (ndvi_calculator_tiny)
Wrote: change.tif (change_detector_tiny)
Wrote: ships.geojson (ship_detector_tiny)
Wrote: value.json (economic_spatial_assessor_tiny)
Wrote: sr.tif (super_resolution_tiny)
```

Only the first two results are meaningful: the first comes from a trained checkpoint and the second computes NDVI [3]. The other commands run starter models. `sr.tif` has 1200 by 800 pixels of 2.5 m, four times the resolution of the 10 m input. `value.json` has the form:

```json
{
  "model_id": "economic_spatial_assessor_tiny",
  "units": [
    "currency m-2"
  ],
  "values": {
    "median_value": -0.00143384316470474
  }
}
```

The hidden command `unbihexium infer MODEL_ID -i INPUT -o OUTPUT` is an alias of `predict` with default options, kept for compatibility with earlier scripts.

## 4. Python task APIs

### 4.1 Generic functions

`unbihexium.ai` provides three functions that work for every model:

| Function | Purpose |
| --- | --- |
| `task_api(model, **options)` | Open the model and return the task API that matches its task (table in Section 4.2) |
| `predict(model, image, **options)` | `task_api(model, **options).predict(image)` |
| `write_result(result, path)` | Write a result as GeoJSON, GeoTIFF or JSON as in Section 3.2 |

```python
import numpy as np

from unbihexium.ai import predict

red_nir = np.array([[[0.1, 0.2], [0.0, 0.05]], [[0.5, 0.2], [0.0, 0.45]]], dtype="float32")
result = predict("ndvi_calculator_tiny", red_nir)
print(result.names, result.values[0])
```

```text
['ndvi'] [[0.6666666  0.        ]
 [       nan 0.79999995]]
```

The pixel with red and near infrared both zero has an undefined index and is NaN; the other values are the NDVI of the pixels in float32 precision.

### 4.2 Task API classes

| Task of the model | Class | Method | Result |
| --- | --- | --- | --- |
| Detection | `ObjectDetector` | `predict(image)` | `DetectionResult` |
| Segmentation | `SemanticSegmenter` | `predict(image)` | `SegmentationResult` |
| Change detection | `ChangeDetector` | `predict_pair(before, after)` or `predict(stacked)` | `SegmentationResult` |
| Dense regression, spectral index | `DenseRegressor` | `predict(image)` | `RegressionResult` with values `(K, H, W)` |
| Scene regression | `SceneRegressor` | `predict(image)` | `RegressionResult` with values `(K,)` |
| Enhancement | `Enhancer` | `predict(image)` | `EnhancementResult` |
| Super-resolution | `SuperResolution` | `enhance(image)` or `predict(image)` | `SuperResolutionResult` |

Subclasses only change the default family used when no model is given: `ShipDetector` (`ship_detector`), `SARShipDetector`, `BuildingDetector`, `AircraftDetector`, `VehicleDetector`, `GreenhouseDetector`, `CropDetector` (`crop_detector`), `PivotDetector` (`pivot_inventory`), `FireDetector` (`fire_monitor`); `LandCoverClassifier` (`lulc_classifier`), `WaterDetector` (`water_surface_detector`), `CloudMasker` (`cloud_mask`), `CropClassifier`, `FloodMapper` (`sar_flood_detector`), `OilSpillDetector` (`sar_oil_spill_detector`); `TreeHeightEstimator`, `LandSurfaceTemperature`, `NDVICalculator` (`ndvi_calculator`), `YieldPredictor`. A class raises `ValueError` when it is given a model of another task.

Images MAY be file paths, `unbihexium.core.raster.Raster` objects or NumPy arrays of shape `(bands, rows, columns)` (a two-dimensional array is one band). Files and rasters keep their CRS and transform; for a raster with a finite no-data value, pixels whose bands all equal it become missing. A NumPy array has no georeference; its results use pixel coordinates with the identity transform and are labelled `EPSG:4326`, which is only a placeholder.

### 4.3 Options

Every task API accepts:

| Argument | Default | Meaning |
| --- | --- | --- |
| `model` | the class default family | Any source of Section 2.1 |
| `weights` | none | Checkpoint or ONNX file; overrides `model` |
| `variant` | none | Variant for family names |
| `device` | `cpu` | PyTorch device |
| `backend` | `auto` | Section 2.2 |
| `tile_size` | tile size of the model | Tile size of the inference (256 for `SuperResolution` when not given) |
| `overlap` | 0.25 | Tile overlap fraction, clamped to [0, 0.9] |
| `batch_size` | 4 | Tiles per forward pass |

`ObjectDetector` adds `threshold` (0.5), `iou_threshold` (0.5) for non-maximum suppression and `max_detections` (1000). `SemanticSegmenter` and `ChangeDetector` add `threshold` (0.5) and `return_probabilities` (`False`): for a two-class model a pixel gets class 1 when its probability is at least `threshold`; for more classes it gets the most probable class, or 255 when that probability is below `threshold`; `threshold=None` disables the threshold. `SuperResolution` adds `scale_factor`; a factor other than the catalogue scale builds an untrained network with that factor.

### 4.4 Results

| Result | Main attributes and methods |
| --- | --- |
| `DetectionResult` | `detections` (list of `Detection` with `bbox` in pixels, `geo_bbox` in map units, `confidence`, `class_id`, `class_name`), `count`, `counts_by_class()`, `filter_by_confidence(t)`, `filter_by_class(*names)`, `as_arrays()`, `to_geojson(pixel_coordinates=False)` |
| `SegmentationResult` | `mask`, `classes`, `probabilities` (with `return_probabilities=True`), `class_mask(c)`, `class_fractions()` (fractions of the valid pixels), `class_areas()` (pixel count times pixel area in squared CRS units), `to_raster()` |
| `RegressionResult` | `values`, `names`, `units`, `is_dense`, `output(name)`, `summary()` (mean, min, max and standard deviation per output), `to_dict()`, `to_raster()` for dense results |
| `EnhancementResult`, `SuperResolutionResult` | `raster` (a `Raster` with the output bands), `bands` or `scale_factor` |

### 4.5 Examples

```python
from unbihexium.ai import ChangeDetector, SuperResolution, WaterDetector, write_result

water = WaterDetector(weights="runs/water_surface_detector_tiny/best.pt").predict("scene_rgbn.tif")
print(water.model_id, water.mask.shape, water.classes, water.mask.dtype)
print({name: round(area / 1e6, 3) for name, area in water.class_areas().items()}, "km2")
write_result(water, "water_python.tif")

change = ChangeDetector("change_detector_tiny").predict_pair("scene_rgb.tif", "scene_rgb_2025.tif")
print(change.classes, change.crs)

upscaled = SuperResolution("super_resolution_tiny").enhance("scene_rgb.tif")
print(upscaled.scale_factor, upscaled.raster.data.shape)
```

```text
water_surface_detector_tiny (200, 300) ['background', 'water'] uint8
{'background': 3.816, 'water': 2.184} km2
['no_change', 'change'] EPSG:32635
4 (3, 800, 1200)
```

The areas are pixel counts times the pixel area of 100 m2. The toy checkpoint was trained for three epochs on four synthetic tiles, so its map is not meaningful; a model trained on real reference data is needed for real maps.

## 5. Tiled inference

### 5.1 Algorithm

`unbihexium.ai.inference.Predictor` runs every model and is used by all task APIs:

1. **Preparation.** The image is converted to float32 with shape `(C, H, W)`; the number of bands MUST equal the number of model inputs. A pixel is invalid when none of its bands is finite. For learned models the bands are standardised with the statistics stored in the configuration, and missing values become 0; spectral index models receive the raw values.
2. **Tiling.** The tile size $s$ is the requested or configured size rounded up to the size multiple $m = 2^{\mathrm{depth}}$ of the variant (8, 16, 16 or 32). Tiles start every $\max(m, \lfloor s(1 - o)/m \rfloor m)$ pixels for the overlap $o$, and a last tile is aligned with the far edge. An image smaller than a tile is padded to a multiple of $m$. Padding uses reflection (edge replication for images one pixel wide).
3. **Dense tasks.** Class logits are converted to probabilities with a softmax. The outputs of all tiles are accumulated with the weight $w(i, j) = \max(10^{-3}, u_H(i)\, u_W(j))$, where $u_n(k) = \min(k + 1, n - k) / ((n + 1)/2)$ falls linearly towards the tile edges, and divided by the sum of the weights. This removes seams between tiles. Super-resolution outputs are accumulated on the grid that is `scale` times finer.
4. **Detection.** Each tile is decoded separately: local maxima of the class heatmaps above the threshold become boxes centred at $((j + \delta_x) \cdot 4, (i + \delta_y) \cdot 4)$ with the predicted size times 4 [4]. Boxes whose centre lies within $s \cdot o / 2$ pixels of an inner tile edge are dropped, because the neighbouring tile sees them better. The remaining boxes are shifted to image coordinates, clipped to the image, removed when their centre pixel is invalid, and passed to class-aware greedy non-maximum suppression with the IoU threshold; at most `max_detections` boxes are kept.
5. **Scene regression.** The whole image is padded to a multiple of $m$ and passed through the network once; no tiling is applied.
6. **Missing data.** Output pixels of invalid input pixels are NaN (and 255 in class maps).

### 5.2 Direct use

```python
import numpy as np

from unbihexium.ai.inference import Predictor

image = np.random.default_rng(0).uniform(0.0, 0.3, size=(6, 300, 400)).astype("float32")
predictor = Predictor("change_detector_tiny", tile_size=128, overlap=0.25)
print(predictor.tile_size, len(predictor.windows(300, 400)), predictor.windows(300, 400)[:3])
probabilities = predictor.dense(image)
print(probabilities.shape, round(float(probabilities.sum(axis=0).mean()), 4))
```

```text
128 12 [(0, 0), (0, 96), (0, 192)]
(2, 300, 400) 1.0
```

`Predictor(source, variant=None, device="cpu", backend="auto", tile_size=None, overlap=0.25, batch_size=4, normalization=None)` offers `dense(image)` for all dense tasks, `detect(image, threshold=0.3, max_detections=1000, iou_threshold=0.5)` returning pixel boxes, scores and class indices, and `scene(image)` for scene regression. `normalization` overrides the stored statistics.

## 6. ONNX Runtime

`unbihexium zoo export` (or `unbihexium.zoo.export.export_onnx`) writes an ONNX file with opset 18 [5], a dynamic batch axis and dynamic height and width, and compares ONNX Runtime with PyTorch before it returns ([download_and_verify.md](download_and_verify.md#35-exporting-to-onnx)). The file stores the model configuration, including the normalisation statistics, under the metadata key `unbihexium_config`, and the weights digest under `unbihexium_weights_digest`. An export therefore runs without PyTorch and without the catalogue:

```python
from unbihexium.ai import predict
from unbihexium.ai.inference import Predictor

predictor = Predictor("water.onnx")
print(type(predictor.backend).__name__, predictor.model_id, predictor.config.channel_names)
print(predict("water.onnx", "scene_rgbn.tif").mask.shape)
```

```text
OnnxBackend water_surface_detector_tiny ('blue', 'green', 'red', 'nir')
(200, 300)
```

An environment with ONNX Runtime but without PyTorch runs models this way; the container image contains both (see [../operations/docker.md](../operations/docker.md)). Tile sizes of an ONNX model MUST remain multiples of the size multiple of its variant, which `Predictor` ensures.

## 7. Pipelines and the REST service

The same task APIs back five registered pipelines, `ship_detection`, `building_detection`, `water_detection`, `change_detection` (inputs `--input` and `--input2`) and `super_resolution`, which run with their default families and starter weights unless a `weights` or `model` parameter is given:

```bash
unbihexium pipeline list
unbihexium pipeline run water_detection -i scene_rgbn.tif -o water_pipeline.tif -p weights=runs/water_surface_detector_tiny/best.pt
```

The REST service in `unbihexium.serving` exposes `POST /predict/{model_id}` for every model of the zoo, with images posted as JSON lists or base64-encoded NumPy files, and `GET /models` and `GET /models/{model_id}` for the catalogue. It is started with `unbihexium serve` (extra `serving`) or `uvicorn unbihexium.serving.app:app`; its configuration and deployment are described in [../operations/docker.md](../operations/docker.md).

## 8. Limitations and responsible use

- The output of a starter model MUST NOT be used as a result: it is untrained.
- Inputs MUST have the bands, order, units and processing level of the training data; a model trained on reflectance gives wrong results on digital numbers, and a Sentinel-2 model does not transfer to another sensor without retraining.
- Tiling reduces but does not eliminate edge effects; objects larger than a tile cannot be detected as one box.
- Detection and segmentation thresholds trade precision against recall and SHOULD be chosen on validation data.
- The accuracy of a trained model SHOULD be reported with every result derived from it, and uses that affect people, property or security MUST follow [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[4] Zhou, X., Wang, D. and Kraehenbuehl, P. Objects as points. arXiv:1904.07850. 2019. <https://arxiv.org/abs/1904.07850>

[5] ONNX project. ONNX concepts: operator sets and versioning. 2026. <https://onnx.ai/onnx/intro/concepts.html>

<!--
=============================================================================
End of file docs/model_zoo/inference.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
