<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/tutorials/index.md
Title       : Tutorials
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Tutorials

| Field | Value |
| --- | --- |
| Document | UBX-DOC-TUTORIALS |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium (declared version 1.0.1, model catalogue 2.0.0) |

## Abstract

This document contains five short, end-to-end tutorials that take a new user from a synthetic GeoTIFF to spectral indices, to building, running and training a model of the model zoo, to serving models over HTTP and to interpolating point observations by kriging. It is written for users who have installed the library and want to see complete workflows that they can run and adapt, and for teachers who need small, self-contained exercises. Every tutorial creates its own synthetic input data, so no download or account is needed, and every command and Python block below was executed in the given order in one empty directory on 24 September 2026; the outputs shown are the real outputs of that run. The tutorials complement, and do not replace, the [quick start](../getting_started/quickstart.md) and the reference documentation.

## Contents

1. [Before you start](#1-before-you-start)
2. [Tutorial 1: spectral indices from a GeoTIFF](#2-tutorial-1-spectral-indices-from-a-geotiff)
3. [Tutorial 2: build, inspect and run a tiny model](#3-tutorial-2-build-inspect-and-run-a-tiny-model)
4. [Tutorial 3: train on synthetic data, predict and export](#4-tutorial-3-train-on-synthetic-data-predict-and-export)
5. [Tutorial 4: serve models over HTTP](#5-tutorial-4-serve-models-over-http)
6. [Tutorial 5: interpolate point observations by kriging](#6-tutorial-5-interpolate-point-observations-by-kriging)
7. [Where to go next](#7-where-to-go-next)
8. [References](#references)

## 1. Before you start

### 1.1 Requirements

| Tutorial | Extras | Time on the test machine |
| --- | --- | --- |
| 1. Spectral indices | none | seconds |
| 2. Tiny model | `torch` | seconds |
| 3. Training and export | `torch`, `onnx` | about 6 s per training run |
| 4. REST service | `serving` | seconds |
| 5. Kriging | none | seconds |

The times were measured with `time` on a 4 vCPU container with CPython 3.13 and PyTorch on the CPU only; they are indications, not benchmarks. Install the main branch from source with all three extras as described in the [installation guide](../getting_started/installation.md), for example:

```bash
python -m pip install -e ".[torch,onnx,serving]"
```

### 1.2 Working directory and model store

Run all tutorials in the same new, empty directory, in order: later tutorials use files written by earlier ones. Point the model store at a directory inside it so that nothing is written to your home directory:

```bash
mkdir unbihexium-tutorials && cd unbihexium-tutorials
export UNBIHEXIUM_CACHE="$PWD/cache"
```

Save each Python block to a file (the suggested name is given before it) and run it with `python <file>`, or paste it into any Python interpreter started in that directory.

### 1.3 A note on the models

The 520 models of the model zoo (130 families in the variants tiny, base, large and mega) are untrained starter models with deterministic weights, except the 28 models of the 7 spectral index families, which compute exact formulas. Tutorials 2 and 3 show the complete mechanics of building, training and running models, but the predictions of an untrained or synthetically trained model carry no information about real imagery. See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## 2. Tutorial 1: spectral indices from a GeoTIFF

### 2.1 Goal

Write a small 6-band surface reflectance scene with three land cover types, compute vegetation, water and soil indices in Python and on the command line, write the result as a Cloud Optimized GeoTIFF and map burn severity from a simulated fire.

### 2.2 Create the scene

The scene has 60 rows and 90 columns of 20 m pixels in UTM zone 35N. Columns 0 to 29 are water, 30 to 59 vegetation and 60 to 89 bare soil, with typical reflectances in the bands blue, green, red, near infrared (NIR), SWIR 1.6 um and SWIR 2.2 um, plus a little noise. File `make_scene.py`:

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import write_geotiff

# Surface reflectance of three cover types in the bands blue, green, red,
# near infrared (NIR), SWIR 1.6 um and SWIR 2.2 um.
signatures = {
    "water": [0.06, 0.05, 0.03, 0.02, 0.01, 0.01],
    "vegetation": [0.03, 0.06, 0.04, 0.40, 0.20, 0.10],
    "bare soil": [0.12, 0.16, 0.20, 0.26, 0.32, 0.28],
}
cover = np.zeros((60, 90), dtype=int)   # Columns 0-29 water,
cover[:, 30:60] = 1                      # 30-59 vegetation,
cover[:, 60:] = 2                        # 60-89 bare soil.
table = np.array(list(signatures.values()), dtype="float32")   # (3 covers, 6 bands)
rng = np.random.default_rng(42)
scene = table[cover].transpose(2, 0, 1) + rng.normal(0, 0.005, (6, 60, 90)).astype("float32")

path = write_geotiff(
    np.clip(scene, 0.0, 1.0),
    "scene.tif",
    crs="EPSG:32635",                               # UTM zone 35N
    transform=from_origin(500000, 6700000, 20, 20),  # 20 m pixels
    descriptions=["blue", "green", "red", "nir", "swir16", "swir22"],
)
print(path)
```

```text
scene.tif
```

### 2.3 Compute indices in Python

`read_geotiff` returns the bands as a `(bands, rows, columns)` array and the georeferencing as a dictionary. The index functions of `unbihexium.indices` take bands as keyword arguments; `compute_index` evaluates an index by name. File `indices.py`:

```python
import numpy as np

from unbihexium.indices import mndwi, ndvi, compute_index
from unbihexium.io import read_geotiff, write_geotiff

data, meta = read_geotiff("scene.tif")
print(data.shape, data.dtype, meta["crs"], meta["transform"])
blue, green, red, nir, swir1, swir2 = data

vegetation = ndvi(nir=nir, red=red)
water = mndwi(green=green, swir1=swir1)
bare = compute_index("bsi", blue=blue, red=red, nir=nir, swir1=swir1)

for name, values in [("NDVI", vegetation), ("MNDWI", water), ("BSI", bare)]:
    means = [values[:, a:a + 30].mean() for a in (0, 30, 60)]
    print(f"{name:6s} water {means[0]:+.2f}  vegetation {means[1]:+.2f}  bare soil {means[2]:+.2f}")

out = write_geotiff(vegetation.astype("float32"), "ndvi.tif", crs=meta["crs"],
                    transform=meta["transform"], nodata=np.nan, cog=True)
print(out)
```

```text
(6, 60, 90) float32 EPSG:32635 (20.0, 0.0, 500000.0, 0.0, -20.0, 6700000.0)
NDVI   water -0.20  vegetation +0.82  bare soil +0.13
MNDWI  water +0.67  vegetation -0.54  bare soil -0.33
BSI    water -0.34  vegetation -0.28  bare soil +0.16
ndvi.tif
```

Each index separates the cover type it was designed for: NDVI [1] is high over vegetation, MNDWI [2] positive over water and the bare soil index (BSI) positive over soil. The formulas and sources of all indices are listed in the [glossary](../glossary.md#4-spectral-indices).

### 2.4 Compute an index on the command line

`unbihexium index` evaluates the 27 indices of the registry. Band options are 1-based band numbers of the input file; their defaults follow a full Sentinel-2 stack, so give every band explicitly for this 6-band file:

```bash
unbihexium index NDVI -i scene.tif -o ndvi_cli.tif --blue 1 --green 2 --red 3 --nir 4 --swir1 5 --swir2 6
unbihexium index NBR -i scene.tif -o nbr_cli.tif --blue 1 --green 2 --red 3 --nir 4 --swir1 5 --swir2 6
```

```text
Wrote: ndvi_cli.tif (NDVI)
Wrote: nbr_cli.tif (NBR)
```

Without the band options the command stops with `Error: NIR is band 8, but the raster has 6 bands`, and an unknown index name prints the list of available names. `ndvi_cli.tif` holds the same values as `ndvi.tif` from Section 2.3.

### 2.5 Map burn severity

The differenced Normalized Burn Ratio (dNBR) compares NBR before and after a fire; `burn_severity` assigns the classes of Key and Benson [3]. File `burn.py` simulates a fire in the vegetated strip by lowering the NIR and raising the SWIR 2.2 um reflectance:

```python
import numpy as np

from unbihexium.indices import BURN_SEVERITY_CLASSES, burn_severity, dnbr, nbr
from unbihexium.io import read_geotiff

data, meta = read_geotiff("scene.tif")
nir, swir2 = data[3], data[5]

# Simulate a fire in the vegetated strip: NIR falls and SWIR 2.2 um rises.
burnt_nir, burnt_swir2 = nir.copy(), swir2.copy()
burnt_nir[:, 30:60] *= 0.4
burnt_swir2[:, 30:60] += 0.15

delta = dnbr(nbr(nir, swir2), nbr(burnt_nir, burnt_swir2))
classes = burn_severity(delta)
for code, count in zip(*np.unique(classes, return_counts=True)):
    print(f"{BURN_SEVERITY_CLASSES[code]:24s} {count:5d} pixels")
```

```text
unburned                  3600 pixels
high severity             1800 pixels
```

The 1,800 burnt pixels are exactly the vegetated strip (60 rows by 30 columns).

## 3. Tutorial 2: build, inspect and run a tiny model

### 3.1 Goal

Find a model in the catalogue, build its starter weights into the local store, verify them, and run the model and an exact spectral index model on the scene of Tutorial 1.

### 3.2 Find and build a model

```bash
unbihexium zoo list --task segmentation --variant tiny
unbihexium zoo info water_surface_detector_tiny
unbihexium zoo build water_surface_detector_tiny
unbihexium zoo verify water_surface_detector_tiny
unbihexium zoo where water_surface_detector_tiny
```

`zoo list` prints a table of the 26 tiny segmentation models. `zoo info` prints the catalogue entry as JSON; the relevant fields for this tutorial are (excerpt):

```text
  "model_id": "water_surface_detector_tiny",
  "tile_size": 256,
  "num_parameters": 733090,
  "requires_training": true,
  "task": "segmentation",
  "bands": ["blue", "green", "red", "nir"],
  "outputs": ["background", "water"],
```

`zoo build` generates the deterministic starter weights, compares their digest with the published one and writes `model.pt`, `config.json` and `model.sha256` to `$UNBIHEXIUM_CACHE/models/water_surface_detector_tiny/`; `zoo verify` prints `Verified: water_surface_detector_tiny`, and `zoo where` prints the path of `model.pt`. Building took 3.7 s of wall-clock time on the test machine, including interpreter start-up.

### 3.3 Run the model from Python

The model expects the four bands blue, green, red and NIR, in this order, so the script first writes those bands to a separate file. File `run_model.py`:

```python
import numpy as np

from unbihexium.ai import NDVICalculator, WaterDetector
from unbihexium.io import read_geotiff, write_geotiff
from unbihexium.zoo import get_model, load_model

entry = get_model("water_surface_detector_tiny")         # Catalogue metadata only.
print(entry.spec.bands, entry.spec.outputs, entry.num_parameters)

model = load_model("water_surface_detector_tiny")        # Built and verified locally.
print(type(model).__name__)
print(model.summary())

# The model expects blue, green, red and NIR: write the first four bands.
data, meta = read_geotiff("scene.tif", bands=[1, 2, 3, 4])
write_geotiff(data, "bgrn.tif", crs=meta["crs"], transform=meta["transform"])

result = WaterDetector(variant="tiny").predict("bgrn.tif")
print(result.model_id, result.mask.shape, result.classes)
print({name: round(share, 3) for name, share in result.class_fractions().items()})

# The spectral index families compute exact formulas and need no training.
red_nir, _ = read_geotiff("scene.tif", bands=[3, 4])
write_geotiff(red_nir, "red_nir.tif", crs=meta["crs"], transform=meta["transform"])
exact = NDVICalculator(variant="tiny").predict("red_nir.tif")
print(exact.model_id, {k: round(v, 3) for k, v in exact.summary()["ndvi"].items()})
```

```text
('blue', 'green', 'red', 'nir') ('background', 'water') 733090
ZooModel
water_surface_detector_tiny: segmentation, 4 input channels, 2 outputs, 733,090 parameters
water_surface_detector_tiny (60, 90) ['background', 'water']
{'background': 0.372, 'water': 0.628}
ndvi_calculator_tiny {'mean': 0.25, 'min': -0.919, 'max': 0.883, 'std': 0.434}
```

The scene is one third water, yet the untrained water detector labels 62.8 % of it as water: its output is arbitrary. The NDVI calculator, in contrast, computes the exact formula; its mean of 0.25 is the mean of the three strips of Section 2.3.

## 4. Tutorial 3: train on synthetic data, predict and export

### 4.1 Goal

Run the complete training, prediction and export workflow on generated data. Training on synthetic data only checks that the setup works; for a useful model, train on a labelled dataset as described in [docs/model_zoo/training.md](../model_zoo/training.md).

### 4.2 Train on the command line

`--synthetic 32` replaces a dataset folder with 32 generated samples for the task of the model:

```bash
unbihexium train water_surface_detector_tiny --synthetic 32 --epochs 2 --chip-size 64 --batch-size 4
```

```text
epoch 1/2 loss 0.9595 miou 0.8900
epoch 2/2 loss 0.2669 miou 0.9256
Best epoch: 2
Best checkpoint: runs/water_surface_detector_tiny/best.pt
```

The command then prints the validation metrics of the best epoch as JSON (loss, accuracy, mean IoU, mean F1, kappa, and IoU, F1, precision and recall per class). They refer to the synthetic validation split and say nothing about real imagery. The run writes `best.pt`, `last.pt` and `history.json` to `runs/water_surface_detector_tiny/`; it took 6.0 s of wall-clock time on the test machine, and a second run with the same default seed (`--seed 0`) printed identical values.

### 4.3 Predict, export to ONNX and predict without PyTorch

```bash
unbihexium predict runs/water_surface_detector_tiny/best.pt bgrn.tif water.tif
unbihexium zoo export runs/water_surface_detector_tiny/best.pt water.onnx
unbihexium predict water.onnx bgrn.tif water_onnx.tif --backend onnx
```

```text
Wrote: water.tif (water_surface_detector_tiny)
Exported: water.onnx
Wrote: water_onnx.tif (water_surface_detector_tiny)
```

`water.tif` is a single-band `uint8` class map in the CRS of the input. `zoo export` compares the ONNX model with PyTorch in ONNX Runtime before it writes the file (skip this with `--no-verify`), and the exported file carries the configuration and the normalisation statistics of the checkpoint, so the last command needs only the `onnx` extra. In this run the two class maps were identical in every pixel.

### 4.4 Train and predict from Python

The same workflow through the Python API; `TrainConfig` holds the options of `unbihexium train`. File `train.py`:

```python
from unbihexium.ai import WaterDetector
from unbihexium.ai.training import TrainConfig, train

config = TrainConfig(epochs=2, batch_size=4, chip_size=64, output_dir="runs_py", seed=0, verbose=False)
result = train("water_surface_detector_tiny", synthetic=32, config=config)
print(result.best_checkpoint, result.best_epoch, round(result.best_metrics["miou"], 4))

mask = WaterDetector(weights=result.best_checkpoint).predict("bgrn.tif")
print(mask.model_id, {k: round(v, 3) for k, v in mask.class_fractions().items()})
```

```text
runs_py/best.pt 2 0.9256
water_surface_detector_tiny {'background': 0.371, 'water': 0.629}
```

The checkpoint is written directly into `output_dir`. The water fraction is again far from the true one third, as expected from a model that has seen only synthetic data.

## 5. Tutorial 4: serve models over HTTP

### 5.1 Goal

Start the REST service of `unbihexium.serving`, query the catalogue and compute NDVI of the tutorial scene over HTTP.

### 5.2 Start the service

In a second terminal, in the same directory and with the same `UNBIHEXIUM_CACHE`:

```bash
unbihexium serve --host 127.0.0.1 --port 8000
```

`unbihexium serve` starts uvicorn with the settings of `unbihexium.config` (options `--config` and `--proxy-headers`); `uvicorn unbihexium.serving.app:app --host 127.0.0.1 --port 8000` is equivalent. The interactive OpenAPI documentation is then available at `http://127.0.0.1:8000/docs`.

### 5.3 Query the service with curl

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models/ndvi_calculator_tiny
curl -s -X POST http://127.0.0.1:8000/predict/ndvi_calculator_tiny \
    -H "Content-Type: application/json" \
    -d '{"image": [[[0.05, 0.06], [0.04, 0.05]], [[0.40, 0.45], [0.30, 0.35]]]}'
```

```text
{"status":"healthy","version":"1.0.1","ready":true,"models_available":520,"models_loaded":0}
{"model_id":"ndvi_calculator_tiny","task":"spectral_index","description":"Normalized Difference Vegetation Index, (NIR - RED) / (NIR + RED) (Rouse et al., 1974).","name":"NDVI Calculator (tiny)","domain":"indices","variant":"tiny","in_channels":2,"channels":["red","nir"],"outputs":["ndvi"],"units":[],"requires_training":false}
{"model_id":"ndvi_calculator_tiny","task":"spectral_index","success":true,"input_shape":[2,2,2],"elapsed_ms":1731.767,"requires_training":false,"result":{"outputs":["ndvi"],"units":[],"shape":[1,2,2],"statistics":{"ndvi":{"count":4,"missing":0,"mean":0.764297366142273,"std":0.009829425528553128,"min":0.7499999403953552,"max":0.7777777314186096}}}}
```

curl prints no line break after a response; the three responses are shown on separate lines. The image is a nested list `(bands, rows, columns)` with the bands in the order given by `channels` (red, then NIR). `elapsed_ms` of the first request includes loading the model and differs between runs. A request with the wrong number of bands is answered with status 422 and `{"detail":"ndvi_calculator_tiny expects 2 bands (red, nir), got 1"}`.

### 5.4 Send a raster from Python

For larger images, send the array as a base64-encoded NumPy `.npy` file and ask for the output values. File `client.py` uses only the standard library and NumPy:

```python
import base64
import io
import json
import urllib.request

import numpy as np

from unbihexium.io import read_geotiff

# Red and NIR bands of the tutorial scene, sent as a base64-encoded .npy file.
data, meta = read_geotiff("scene.tif", bands=[3, 4])
buffer = io.BytesIO()
np.save(buffer, data)
body = {
    "image_npy_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    "crs": meta["crs"],
    "transform": list(meta["transform"]),
    "parameters": {"return_values": True},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/predict/ndvi_calculator_tiny",
    data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    reply = json.load(response)

print(reply["model_id"], reply["input_shape"], reply["requires_training"])
print(sorted(reply["result"]))
ndvi = np.array(reply["result"]["values"], dtype="float32")
print(ndvi.shape, round(float(ndvi[0, :, 30:60].mean()), 2))
```

```text
ndvi_calculator_tiny [2, 60, 90] False
['outputs', 'shape', 'statistics', 'units', 'values']
(1, 60, 90) 0.82
```

The mean NDVI of the vegetated strip, 0.82, matches Section 2.3.

### 5.5 Protect the service

By default the service has no API key, no rate limit and allows all CORS origins. Before exposing it beyond your own machine, configure at least an API key and a rate limit (and put the service behind a TLS-terminating reverse proxy):

```bash
UNBIHEXIUM_SERVING__API_KEY=change-me UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE=60 \
    unbihexium serve --host 127.0.0.1 --port 8001
curl -s http://127.0.0.1:8001/models/ndvi_calculator_tiny
curl -s -o /dev/null -w "%{http_code}\n" -H "X-API-Key: change-me" http://127.0.0.1:8001/models/ndvi_calculator_tiny
```

```text
{"detail":"missing API key"}
200
```

The two responses are shown on separate lines. `/health` stays reachable without a key for liveness probes. Use a long random key in practice; the settings are described in [docs/getting_started/configuration.md](../getting_started/configuration.md). Stop the servers with Ctrl+C when you are done.

## 6. Tutorial 5: interpolate point observations by kriging

### 6.1 Goal

Estimate a continuous surface from 120 scattered observations: fit a variogram, check the kriging model by cross-validation and write the estimate and its variance as a GeoTIFF.

### 6.2 Fit, validate and predict

Ordinary kriging [4] predicts each location as a weighted mean of the observations, with weights derived from a variogram model fitted to the empirical semivariances; it also returns the kriging variance of each prediction. File `kriging.py`:

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.geostat import OrdinaryKriging, Variogram
from unbihexium.io import write_geotiff

# 120 observations of a smooth field with a little noise on a 2 km square.
rng = np.random.default_rng(7)
coords = rng.uniform(0, 2000, size=(120, 2))
values = 10 + 2 * np.sin(coords[:, 0] / 150) * np.cos(coords[:, 1] / 200) + rng.normal(0, 0.2, 120)

# Empirical variogram and a fitted spherical model.
variogram = Variogram(n_lags=12, max_lag=800, model="spherical")
fit = variogram.fit(coords, values)
print(f"nugget {fit.nugget:.3f}  partial sill {fit.sill:.3f}  range {fit.range_param:.0f} m")

# Ordinary kriging with the fitted variogram, checked by leave-one-out cross-validation.
kriging = OrdinaryKriging(variogram=variogram).fit(coords, values)
print({k: round(v, 3) for k, v in kriging.cross_validate().items()})

# Predict on a 50 m grid and write the estimate and the kriging variance.
x = np.arange(25, 2000, 50.0)
y = np.arange(1975, 0, -50.0)
estimate, variance = kriging.predict_grid(x, y)
transform = from_origin(0, 2000, 50, 50)
write_geotiff(np.stack([estimate, variance]).astype("float32"), "kriged.tif",
              crs="EPSG:3067", transform=transform, descriptions=["estimate", "variance"])
print(estimate.shape, round(float(variance.max()), 3))
```

```text
nugget 0.000  partial sill 1.304  range 467 m
{'rmse': 0.425, 'mae': 0.297, 'bias': 0.007, 'msse': 0.292}
(40, 40) 1.136
```

The coordinates are treated as metres in EPSG:3067 (ETRS89 / TM35FIN) only to give the output a CRS. The cross-validation reports the root mean square error, the mean absolute error, the bias and the mean standardised squared error (MSSE). An MSSE well below 1, as here, means that the kriging variance overstates the actual errors; try another variogram model (`"exponential"`, `"gaussian"`, `"matern"`) or `max_lag` and compare. The grid rows run from north to south, matching the transform.

## 7. Where to go next

| Topic | Document |
| --- | --- |
| All installation options | [docs/getting_started/installation.md](../getting_started/installation.md) |
| Settings and environment variables | [docs/getting_started/configuration.md](../getting_started/configuration.md) |
| Every command and option | [docs/reference/cli.md](../reference/cli.md) |
| The Python API | [docs/reference/api.md](../reference/api.md) |
| Dataset layout, training and evaluation | [docs/model_zoo/training.md](../model_zoo/training.md) |
| Inference, results and ONNX | [docs/model_zoo/inference.md](../model_zoo/inference.md) |
| Families and variants of the model zoo | [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md) |
| Capabilities by application domain | [docs/capabilities/index.md](../capabilities/index.md) |
| Example scripts | [examples/README.md](../../examples/README.md) |
| Terms and formulas | [docs/glossary.md](../glossary.md) |

## References

[1] Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[2] Xu, H. Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing 27(14), 3025-3033. 2006. <https://doi.org/10.1080/01431160600589179>

[3] Key, C. H. and Benson, N. C. Landscape assessment: ground measure of severity, the Composite Burn Index, and remote sensing of severity, the Normalized Burn Ratio. USDA Forest Service General Technical Report RMRS-GTR-164-CD. 2006.

[4] Matheron, G. Principles of geostatistics. Economic Geology 58(8), 1246-1266. 1963. <https://doi.org/10.2113/gsecongeo.58.8.1246>

<!--
=============================================================================
End of file docs/tutorials/index.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
