<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/reference/cli.md
Title       : Command Line Reference
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Command Line Reference

| Field | Value |
| --- | --- |
| Document | UBX-DOC-REF-CLI |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document is the reference of the `unbihexium` command: every command and subcommand, every option with its default as printed by `--help`, the arguments that name models, the output formats, the exit status and the environment variables that influence the command. It is written for users who script the command line interface, for operators who run it in containers or batch jobs, and for contributors who change `src/unbihexium/cli/main.py`. The help texts were captured from the main branch with a terminal width of 80 columns, and every example was executed; outputs are shown where they are short. A guided introduction is given in [docs/getting_started/quickstart.md](../getting_started/quickstart.md).

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Synopsis](#2-synopsis)
- [3. Exit status and error reporting](#3-exit-status-and-error-reporting)
- [4. Model arguments and requirements](#4-model-arguments-and-requirements)
- [5. unbihexium info](#5-unbihexium-info)
- [6. unbihexium index](#6-unbihexium-index)
- [7. unbihexium zoo](#7-unbihexium-zoo)
- [8. unbihexium train](#8-unbihexium-train)
- [9. unbihexium evaluate](#9-unbihexium-evaluate)
- [10. unbihexium predict](#10-unbihexium-predict)
- [11. unbihexium pipeline](#11-unbihexium-pipeline)
- [12. unbihexium serve](#12-unbihexium-serve)
- [13. Hidden compatibility aliases](#13-hidden-compatibility-aliases)
- [14. Environment and shell completion](#14-environment-and-shell-completion)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

The command is installed as the console script `unbihexium` (entry point `unbihexium.cli.main:main` in `pyproject.toml`) and is built with Click [1]. It can also be started as `python -m unbihexium.cli.main`. The document describes the main branch; the release 1.0.1 on PyPI predates several commands (see [installation.md](../getting_started/installation.md#3-choosing-an-installation-method)).

### 1.2 Conventions

Placeholders are written in upper case (`MODEL`, `INPUT_PATH`). Options in square brackets are optional. Paths are relative to the current directory. Band numbers are 1-based, as in GDAL. The key words MUST, SHOULD and MAY are used as described in RFC 2119 [2] and RFC 8174 [3] when they appear in capitals.

## 2. Synopsis

```text
unbihexium [--version] [-v | --verbose] [--help] COMMAND [ARGS]...

unbihexium info
unbihexium index INDEX_NAME -i INPUT -o OUTPUT [--blue N] [--green N] [--red N] [--nir N] ...
unbihexium zoo list [--task TASK] [--domain DOMAIN] [--variant VARIANT] [--json]
unbihexium zoo info MODEL_ID
unbihexium zoo build MODEL_ID [--onnx] [--force] [--cache-dir PATH]
unbihexium zoo export MODEL OUTPUT [--no-verify]
unbihexium zoo verify MODEL_ID
unbihexium zoo where MODEL_ID
unbihexium zoo clear [MODEL_ID] [--yes]
unbihexium train MODEL (--data DIR | --synthetic N) [options]
unbihexium evaluate MODEL --data DIR [options]
unbihexium predict MODEL INPUT_PATH OUTPUT_PATH [options]
unbihexium pipeline list [--domain DOMAIN]
unbihexium pipeline run PIPELINE_ID -i INPUT [--input2 INPUT2] -o OUTPUT [-p KEY=VALUE]...
unbihexium serve [--host HOST] [--port PORT] [--config FILE] [--proxy-headers]
```

```text
$ unbihexium --help
Usage: unbihexium [OPTIONS] COMMAND [ARGS]...

  Unbihexium: Earth observation, geospatial, remote sensing and SAR library.

Options:
  --version      Show the version and exit.
  -v, --verbose  Enable verbose output.
  --help         Show this message and exit.

Commands:
  evaluate  Evaluate a model on a dataset split.
  index     Compute a spectral index of a raster and write it as GeoTIFF.
  info      Display library information.
  pipeline  Registered processing pipelines.
  predict   Run a model on a raster and write the result.
  serve     Start the REST service with uvicorn (needs the serving extra).
  train     Train or fine-tune a model zoo model.
  zoo       Browse the model catalogue and manage the local model store.
```

| Global option | Effect |
| --- | --- |
| `--version` | Prints `unbihexium, version 1.0.1` and exits with status 0 |
| `-v`, `--verbose` | Accepted and stored for the subcommands; no command currently changes its output |
| `--help` | Prints the help of the command or subcommand it follows |

## 3. Exit status and error reporting

| Status | Meaning |
| --- | --- |
| 0 | The command completed |
| 1 | Invalid input, a missing or unknown model, pipeline or index, a failed verification, a declined confirmation (`Aborted!`), or an unexpected exception |
| 2 | Usage error detected by Click: unknown command or option, missing argument, invalid option value or a path that does not exist |

Errors detected by the command are printed as a single line starting with `Error:`, for example:

```text
$ unbihexium zoo info nonexistent_model
Error: unknown model nonexistent_model; see `unbihexium zoo list`
$ unbihexium predict
Usage: unbihexium predict [OPTIONS] MODEL INPUT_PATH OUTPUT_PATH
Try 'unbihexium predict --help' for help.

Error: Missing argument 'MODEL'.
```

The first command exits with status 1 and the second with status 2. Exceptions that the command does not anticipate, for example a missing PyTorch installation in `predict` or a failure inside `zoo export`, end with a Python traceback and status 1.

## 4. Model arguments and requirements

### 4.1 Model arguments

Commands that take `MODEL` or `MODEL_ID` accept the following forms:

| Form | Example | Accepted by |
| --- | --- | --- |
| Model id: family and variant | `water_surface_detector_tiny` | every model command |
| Family name; the variant comes from `--variant`, otherwise `base` | `water_surface_detector` | `train`, `evaluate`, `predict`, `zoo export`, `zoo info`, `zoo build` |
| Checkpoint written by `train` | `runs/water_surface_detector_tiny/best.pt` | `train` (fine-tuning), `evaluate`, `predict`, `zoo export` |
| ONNX export | `water.onnx` | `predict` |

The catalogue contains 520 models: 130 families in the variants `tiny`, `base`, `large` and `mega` (`unbihexium zoo list`). Apart from the 28 models of the 7 spectral index families, which compute exact formulas, every model is an untrained starter model with deterministic weights. Its predictions are not meaningful until it has been trained with `unbihexium train`; see [docs/model_zoo/training.md](../model_zoo/training.md).

### 4.2 Optional dependencies

| Commands | Needs |
| --- | --- |
| `info`, `index`, `zoo list`, `zoo info`, `zoo where`, `zoo clear`, `pipeline list` | Core installation |
| `zoo build`, `zoo verify`, `zoo export`, `train`, `evaluate` | Extra `torch` (`zoo export` also verifies with ONNX Runtime unless `--no-verify` is given, so it needs `onnx` as well) |
| `predict`, `pipeline run` with a model id or checkpoint | Extra `torch` |
| `predict` with an `.onnx` file | Extra `onnx` only |
| `serve` | Extra `serving`; predictions with catalogue models also need `torch` |

## 5. unbihexium info

```text
$ unbihexium info --help
Usage: unbihexium info [OPTIONS]

  Display library information.

Options:
  --help  Show this message and exit.
```

Prints the version, the number of registered capabilities, the number of models in the catalogue with the catalogue version, and the number of registered pipelines:

```text
$ unbihexium info
Unbihexium v1.0.1
Registered capabilities: 147
Model zoo models: 520 (catalogue 2.0.0)
Registered pipelines: 5
```

## 6. unbihexium index

### 6.1 Help

```text
$ unbihexium index --help
Usage: unbihexium index [OPTIONS] INDEX_NAME

  Compute a spectral index of a raster and write it as GeoTIFF.

Options:
  -i, --input TEXT    Input raster file.  [required]
  -o, --output TEXT   Output GeoTIFF file.  [required]
  --blue INTEGER      1-based band number of blue.  [default: 2]
  --green INTEGER     1-based band number of green.  [default: 3]
  --red INTEGER       1-based band number of red.  [default: 4]
  --nir INTEGER       1-based band number of near infrared.  [default: 8]
  --swir1 INTEGER     1-based band number of SWIR 1.6 um.  [default: 12]
  --swir2 INTEGER     1-based band number of SWIR 2.2 um.  [default: 13]
  --coastal INTEGER   1-based band number of coastal.  [default: 1]
  --rededge1 INTEGER  1-based band number of red edge 1.  [default: 5]
  --rededge2 INTEGER  1-based band number of red edge 2.  [default: 6]
  --rededge3 INTEGER  1-based band number of red edge 3.  [default: 7]
  --nir08 INTEGER     1-based band number of narrow NIR.  [default: 9]
  --help              Show this message and exit.
```

### 6.2 Behaviour

The command looks up `INDEX_NAME` (case-insensitive) in `unbihexium.core.IndexRegistry`, reads the whole input raster, takes the bands the formula needs by their band numbers, evaluates the formula with its default parameters and writes a single-band float32 GeoTIFF on the grid and CRS of the input. The default band numbers follow the 13-band order of Sentinel-2 Level-1C products (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12) [4]; for any other band order the numbers MUST be given explicitly. Only the options of bands that the formula uses are checked.

```text
$ unbihexium index ndvi -i scene.tif -o ndvi.tif --blue 1 --green 2 --red 3 --nir 4
Wrote: ndvi.tif (NDVI)
$ unbihexium index ndvi -i scene.tif -o ndvi.tif
Error: NIR is band 8, but the raster has 4 bands
```

### 6.3 Indices

| Name | Bands | Formula | Reference |
| --- | --- | --- | --- |
| `NDVI` | nir, red | (NIR - RED) / (NIR + RED) | Rouse et al. (1974) [5] |
| `GNDVI` | nir, green | (NIR - GREEN) / (NIR + GREEN) | Gitelson et al. (1996) |
| `NDRE` | nir, rededge1 | (NIR - REDEDGE1) / (NIR + REDEDGE1) | Gitelson and Merzlyak (1994) |
| `EVI` | nir, red, blue | G (NIR - RED) / (NIR + C1 RED - C2 BLUE + L) | Huete et al. (2002) [6] |
| `EVI2` | nir, red | 2.5 (NIR - RED) / (NIR + 2.4 RED + 1) | Jiang et al. (2008) |
| `SAVI` | nir, red | (1 + L) (NIR - RED) / (NIR + RED + L) | Huete (1988) |
| `MSAVI` | nir, red | (2 NIR + 1 - sqrt((2 NIR + 1)^2 - 8 (NIR - RED))) / 2 | Qi et al. (1994) |
| `OSAVI` | nir, red | (NIR - RED) / (NIR + RED + 0.16) | Rondeaux et al. (1996) |
| `ARVI` | nir, red, blue | (NIR - RB) / (NIR + RB), RB = RED - gamma (BLUE - RED) | Kaufman and Tanre (1992) |
| `VARI` | green, red, blue | (GREEN - RED) / (GREEN + RED - BLUE) | Gitelson et al. (2002) |
| `SR` | nir, red | NIR / RED | Jordan (1969) |
| `WDRVI` | nir, red | (alpha NIR - RED) / (alpha NIR + RED) | Gitelson (2004) |
| `CIgreen` | nir, green | NIR / GREEN - 1 | Gitelson et al. (2003) |
| `CIre` | nir, rededge1 | NIR / REDEDGE1 - 1 | Gitelson et al. (2003) |
| `NDWI` | green, nir | (GREEN - NIR) / (GREEN + NIR) | McFeeters (1996) [7] |
| `MNDWI` | green, swir1 | (GREEN - SWIR1) / (GREEN + SWIR1) | Xu (2006) |
| `NDMI` | nir, swir1 | (NIR - SWIR1) / (NIR + SWIR1) | Gao (1996) |
| `AWEInsh` | green, swir1, nir, swir2 | 4 (GREEN - SWIR1) - (0.25 NIR + 2.75 SWIR2) | Feyisa et al. (2014) |
| `AWEIsh` | blue, green, nir, swir1, swir2 | BLUE + 2.5 GREEN - 1.5 (NIR + SWIR1) - 0.25 SWIR2 | Feyisa et al. (2014) |
| `NDTI` | red, green | (RED - GREEN) / (RED + GREEN) | Lacaux et al. (2007) |
| `NDCI` | rededge1, red | (REDEDGE1 - RED) / (REDEDGE1 + RED) | Mishra and Mishra (2012) |
| `NBR` | nir, swir2 | (NIR - SWIR2) / (NIR + SWIR2) | Key and Benson (2006) |
| `NBR2` | swir1, swir2 | (SWIR1 - SWIR2) / (SWIR1 + SWIR2) | USGS Landsat spectral indices product guide |
| `NDBI` | swir1, nir | (SWIR1 - NIR) / (SWIR1 + NIR) | Zha et al. (2003) |
| `BSI` | swir1, red, nir, blue | ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE)) | Rikimaru et al. (2002) |
| `NDSI` | green, swir1 | (GREEN - SWIR1) / (GREEN + SWIR1) | Hall et al. (1995) |
| `MSI` | swir1, nir | SWIR1 / NIR | Hunt and Rock (1989) |

The table is generated from the registry of the main branch; an unknown name lists the available ones. The full references of the formulas are given in [docs/reference/api.md](api.md#5-unbihexiumindices) and in the source of `unbihexium.core.index`. The options `--coastal`, `--rededge2`, `--rededge3` and `--nir08` exist but no registered index uses these bands.

## 7. unbihexium zoo

```text
$ unbihexium zoo --help
Usage: unbihexium zoo [OPTIONS] COMMAND [ARGS]...

  Browse the model catalogue and manage the local model store.

Options:
  --help  Show this message and exit.

Commands:
  build   Build a model into the local store and verify it.
  clear   Remove one or all models from the local store.
  export  Export a model or checkpoint to ONNX and verify it.
  info    Show the inputs, outputs and metadata of a model.
  list    List model zoo models.
  verify  Verify the files and weights digest of a cached model.
  where   Print the checkpoint path of a cached model.
```

### 7.1 zoo list

```text
Usage: unbihexium zoo list [OPTIONS]

  List model zoo models.

Options:
  -t, --task TEXT    Filter by task, for example detection.
  -d, --domain TEXT  Filter by capability domain.
  --variant TEXT     Filter by variant: tiny, base, large or mega.
  --json             Output as JSON.
  --help             Show this message and exit.
```

Tasks are `detection`, `segmentation`, `change_detection`, `dense_regression`, `scene_regression`, `enhancement`, `super_resolution` and `spectral_index`. The domains of the catalogue are `agriculture`, `ai`, `analysis`, `assets`, `defense`, `energy`, `environment`, `forestry`, `imaging`, `indices`, `risk`, `sar`, `tourism`, `urban` and `water`. An unknown task or variant is an error (status 1, for example `Error: 'bogus' is not a valid Variant`); an unknown domain gives an empty list. Without `--json`, a table with model id, task, domain and parameter count is printed; with `--json`, one object per model with the fields shown by `zoo info`.

```text
$ unbihexium zoo list --task segmentation --variant tiny --domain water
                         Model zoo (3 models)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model ID                       ┃ Task         ┃ Domain ┃ Parameters ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ marine_pollution_detector_tiny │ segmentation │ water  │    733,971 │
│ reservoir_monitor_tiny         │ segmentation │ water  │    733,090 │
│ water_surface_detector_tiny    │ segmentation │ water  │    733,090 │
└────────────────────────────────┴──────────────┴────────┴────────────┘
```

### 7.2 zoo info

```text
Usage: unbihexium zoo info [OPTIONS] MODEL_ID

  Show the inputs, outputs and metadata of a model.

Options:
  --help  Show this message and exit.
```

Prints the catalogue entry as JSON: model id, variant, tile size, published weights digest, parameter count, source, catalogue version, whether the model requires training, family, name, task, domain, description, input bands and their order, number of acquisitions (`dates`), outputs, units, value range, scale factor, formula, the labels needed for training, suitable data sources and licence. Excerpt:

```text
$ unbihexium zoo info ndvi_calculator_tiny
{
  "model_id": "ndvi_calculator_tiny",
  "variant": "tiny",
  "tile_size": 256,
  "weights_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "num_parameters": 0,
  "source": "build",
  "download_url": null,
  "version": "2.0.0",
  "requires_training": false,
  "family": "ndvi_calculator",
  "name": "NDVI Calculator",
  "task": "spectral_index",
  "domain": "indices",
  ...
  "bands": [
    "red",
    "nir"
  ],
  ...
}
```

### 7.3 zoo build

```text
Usage: unbihexium zoo build [OPTIONS] MODEL_ID

  Build a model into the local store and verify it.

Options:
  --onnx            Also export the model to ONNX.
  --force           Rebuild even if the model is cached.
  --cache-dir PATH  Cache root directory.
  --help            Show this message and exit.
```

Builds the starter weights of the model locally and deterministically, compares their digest with the published value in `src/unbihexium/zoo/digests.json`, and writes `model.pt`, `config.json` and `model.sha256` (and `model.onnx` with `--onnx`) to `$UNBIHEXIUM_CACHE/models/<model_id>/`. No weights are downloaded for catalogue models. An existing checkpoint is kept unless `--force` is given. `--cache-dir DIR` stores the model in `DIR/<model_id>/`, without the `models` subdirectory.

```text
$ unbihexium zoo build ndvi_calculator_tiny --cache-dir ./store
Cached: store/ndvi_calculator_tiny
```

### 7.4 zoo export

```text
Usage: unbihexium zoo export [OPTIONS] MODEL OUTPUT

  Export a model or checkpoint to ONNX and verify it.

Options:
  --no-verify  Skip the ONNX Runtime comparison.
  --help       Show this message and exit.
```

Exports a model id, family or checkpoint to ONNX [8] with a dynamic batch size, height and width and stores the model configuration (including the normalisation statistics of a trained checkpoint) in the ONNX metadata, so that `predict` can run the file without PyTorch. Unless `--no-verify` is given, the outputs of ONNX Runtime and PyTorch are compared on a random input before the export is reported.

```text
$ unbihexium zoo export runs/water_surface_detector_tiny/best.pt water.onnx
Exported: water.onnx
```

### 7.5 zoo verify

```text
Usage: unbihexium zoo verify [OPTIONS] MODEL_ID

  Verify the files and weights digest of a cached model.

Options:
  --help  Show this message and exit.
```

Loads the cached checkpoint, checks the weights digest stored in it and compares it with the published digest of the model. It prints `Verified: <model id>` with status 0, or reports that the model is not cached or does not verify with status 1. The command checks the weights, not the other files of the model directory; to check every file against `model.sha256`, run `sha256sum -c model.sha256` in the model directory.

### 7.6 zoo where

```text
Usage: unbihexium zoo where [OPTIONS] MODEL_ID

  Print the checkpoint path of a cached model.

Options:
  --help  Show this message and exit.
```

Prints the absolute path of `model.pt` of a cached model, or fails with status 1 when the model is not cached:

```text
$ unbihexium zoo where water_surface_detector_tiny
Error: water_surface_detector_tiny is not cached; run `unbihexium zoo build
water_surface_detector_tiny`
```

### 7.7 zoo clear

```text
Usage: unbihexium zoo clear [OPTIONS] [MODEL_ID]

  Remove one or all models from the local store.

Options:
  --yes   Do not ask for confirmation.
  --help  Show this message and exit.
```

With `MODEL_ID`, removes that model; without it, asks for confirmation and removes every cached model. `--yes` skips the question, which is required in non-interactive use. The number of removed models is printed; declining the question prints `Aborted!` and exits with status 1.

```text
$ unbihexium zoo clear --yes
Removed 1 model(s)
```

## 8. unbihexium train

### 8.1 Help

```text
$ unbihexium train --help
Usage: unbihexium train [OPTIONS] MODEL

  Train or fine-tune a model zoo model.

Options:
  --data DIRECTORY                Dataset root.
  --synthetic INTEGER             Train on this many synthetic samples
                                  instead.
  --variant TEXT                  Variant for family names: tiny, base, large
                                  or mega.
  --epochs INTEGER                Number of epochs.  [default: 50]
  --batch-size INTEGER            Chips per step.  [default: 8]
  --lr FLOAT                      Peak learning rate.  [default: 0.001]
  --weight-decay FLOAT            AdamW weight decay.  [default: 0.0001]
  --chip-size INTEGER             Chip size in pixels; default is the tile
                                  size.
  --samples-per-epoch INTEGER     Random chips per epoch.
  --device TEXT                   auto, cpu, cuda or mps.  [default: auto]
  --workers INTEGER               Data loader processes.  [default: 0]
  --seed INTEGER                  Random seed.  [default: 0]
  --amp                           Mixed precision on CUDA.
  --patience INTEGER              Stop after this many epochs without
                                  improvement.
  --regression-loss [l1|mse|huber]
                                  Loss of regression targets.  [default: l1]
  --no-augment                    Disable data augmentation.
  --output TEXT                   Output directory.  [default: runs]
  --help                          Show this message and exit.
```

### 8.2 Behaviour

`MODEL` is a model id, a family name with `--variant`, or a checkpoint to fine-tune. Exactly one data source is needed: `--data DIR`, a dataset folder with `train/images`, `train/labels`, `val/images` and `val/labels` (layout and label formats per task in [docs/model_zoo/training.md](../model_zoo/training.md)), or `--synthetic N`, which generates N synthetic samples to check the setup. Without either, the command fails with `Error: pass --data DIR or --synthetic N`.

Training uses AdamW with linear warm-up and cosine decay, gradient clipping, validation after every epoch and early stopping with `--patience`. `--device auto` selects CUDA when available, then Apple `mps`, then the CPU. The per-band normalisation statistics estimated from the training data are stored in the checkpoint. With the default `--output runs`, files are written to `runs/<model_id>/`; any other value is used as the directory itself. The directory receives `best.pt`, `last.pt` and `history.json`. The command prints one line per epoch, the best epoch, the best checkpoint and its validation metrics as JSON:

```text
$ unbihexium train water_surface_detector --variant tiny --synthetic 8 --epochs 1 --batch-size 4 --chip-size 64 --output runs_cli
epoch 1/1 loss 1.3591 miou 0.6618
Best epoch: 1
Best checkpoint: runs_cli/best.pt
{
  "loss": 0.7347595691680908,
  ...
}
```

Metrics obtained on synthetic data only show that the setup works; they are not a measure of model quality.

## 9. unbihexium evaluate

```text
$ unbihexium evaluate --help
Usage: unbihexium evaluate [OPTIONS] MODEL

  Evaluate a model on a dataset split.

Options:
  --data DIRECTORY      Dataset root.  [required]
  --split TEXT          Split: train, val or test.  [default: val]
  --chip-size INTEGER   Chip size in pixels; default is the tile size.
  --batch-size INTEGER  Chips per forward pass.  [default: 8]
  --device TEXT         auto, cpu, cuda or mps.  [default: auto]
  --threshold FLOAT     Detection score threshold.  [default: 0.3]
  --help                Show this message and exit.
```

Evaluates a model id or checkpoint on a split of a dataset folder and prints the metrics of its task as JSON: for segmentation and change detection loss, overall accuracy, mean IoU, mean F1, Cohen's kappa and per-class IoU, F1, precision and recall; for detection average precision (mAP at IoU 0.5 and averaged over 0.5 to 0.95); for regression MAE, RMSE, bias and R^2; for enhancement and super-resolution PSNR and SSIM. The following command evaluates a trained checkpoint on a small labelled folder `water_data` created for this example:

```text
$ unbihexium evaluate runs/water_surface_detector_tiny/best.pt --data water_data --split val --chip-size 64
{
  "loss": 0.36787939071655273,
  "accuracy": 0.941162109375,
  "miou": 0.8627755800171304,
  "mf1": 0.9252245322158705,
  "kappa": 0.8507277794982967,
  ...
  "pixels": 8192
}
```

## 10. unbihexium predict

### 10.1 Help

```text
$ unbihexium predict --help
Usage: unbihexium predict [OPTIONS] MODEL INPUT_PATH OUTPUT_PATH

  Run a model on a raster and write the result.

Options:
  --second FILE                Second date for change detection.
  --variant TEXT               Variant for family names.
  --threshold FLOAT            Detection or segmentation threshold.
  --tile-size INTEGER          Tile size in pixels.
  --overlap FLOAT              Tile overlap fraction.  [default: 0.25]
  --backend [auto|torch|onnx]  Inference backend.  [default: auto]
  --device TEXT                Torch device.  [default: cpu]
  --help                       Show this message and exit.
```

### 10.2 Behaviour

The command opens the model, selects the task API that matches its task, checks that the input has the bands of the catalogue entry in the documented order (`zoo info` lists them), and runs tiled inference: tiles of `--tile-size` pixels (default: the tile size of the variant, 256 or 512) overlap by the fraction `--overlap`, dense outputs are blended across tiles and detections pass non-maximum suppression. `--threshold` overrides the default threshold of detection and binary segmentation tasks. With `--backend auto`, `.onnx` files run on ONNX Runtime and everything else on PyTorch; `--backend onnx` with a model id builds and exports the model into the store first. `--device` applies to PyTorch only; ONNX Runtime uses its CPU execution provider.

The output format follows from the task, not from the file extension:

| Task | Output |
| --- | --- |
| `detection` | GeoJSON [9] FeatureCollection of boxes in the coordinates of the input raster, whose CRS is recorded in a top-level `crs` member |
| `segmentation`, `change_detection` | Single-band GeoTIFF of class labels |
| `dense_regression`, `spectral_index`, `enhancement`, `super_resolution` | Multi-band float32 GeoTIFF |
| `scene_regression` | JSON document with one value per output |

Change detection models need a second acquisition with `--second`. A band mismatch is reported with status 1, for example `Error: ship_detector_tiny expects 3 bands (red, green, blue), got shape (4, 128, 128)`.

### 10.3 Examples

```text
$ unbihexium predict water_surface_detector_tiny scene.tif water_mask.tif
Wrote: water_mask.tif (water_surface_detector_tiny)
$ unbihexium predict water_surface_detector scene.tif water_base_as_tiny.tif --variant tiny
Wrote: water_base_as_tiny.tif (water_surface_detector_tiny)
$ unbihexium predict ship_detector_tiny rgb.tif ships.geojson --threshold 0.3
Wrote: ships.geojson (ship_detector_tiny)
$ unbihexium predict change_detector_tiny rgb.tif change.tif --second rgb_later.tif
Wrote: change.tif (change_detector_tiny)
$ unbihexium predict water.onnx scene.tif water_onnx.tif --backend onnx
Wrote: water_onnx.tif (water_surface_detector_tiny)
```

The starter models in these examples are untrained, so their outputs demonstrate the file formats only; the untrained ship detector, for instance, returned an empty FeatureCollection.

## 11. unbihexium pipeline

### 11.1 pipeline list

```text
Usage: unbihexium pipeline list [OPTIONS]

  List available pipelines.

Options:
  -d, --domain TEXT  Filter by domain.
  --help             Show this message and exit.
```

```text
$ unbihexium pipeline list
                             Pipelines
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Pipeline ID        ┃ Name                        ┃ Domains      ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ building_detection │ Building Detection Pipeline │ ai, urban    │
│ change_detection   │ Change Detection Pipeline   │ ai, change   │
│ ship_detection     │ Ship Detection Pipeline     │ ai, maritime │
│ super_resolution   │ Super Resolution Pipeline   │ ai, imaging  │
│ water_detection    │ Water Detection Pipeline    │ ai, water    │
└────────────────────┴─────────────────────────────┴──────────────┘
```

### 11.2 pipeline run

```text
Usage: unbihexium pipeline run [OPTIONS] PIPELINE_ID

  Run a pipeline on raster files and write its result.

Options:
  -i, --input TEXT   Input file path.  [required]
  --input2 TEXT      Second input for two-date pipelines.
  -o, --output TEXT  Output file path.  [required]
  -p, --param TEXT   Pipeline parameter as KEY=VALUE.
  --help             Show this message and exit.
```

Each registered pipeline wraps a task API of `unbihexium.ai` and writes its result in the format of Section 10.2:

| Pipeline | Task API and default family | Inputs |
| --- | --- | --- |
| `building_detection` | `BuildingDetector`, `building_detector` | `-i` |
| `change_detection` | `ChangeDetector`, `change_detector` | `-i` (first date) and `--input2` (second date) |
| `ship_detection` | `ShipDetector`, `ship_detector` | `-i` |
| `super_resolution` | `SuperResolution`, `super_resolution` | `-i` |
| `water_detection` | `WaterDetector`, `water_surface_detector` | `-i` |

`-p KEY=VALUE` passes keyword arguments to the task API, for example `variant`, `weights` (a checkpoint), `threshold`, `tile_size`, `overlap` or `device`; the option can be repeated. Values are parsed as JSON when possible (`0.4` becomes a number, `true` a boolean) and kept as strings otherwise. The command prints the run id (a random UUID) and the output path.

```text
$ unbihexium pipeline run water_detection -i scene.tif -o water_pipeline.tif -p variant=tiny
Completed: 4b6efb05-23c4-4302-a0fd-9e7156733fa5 -> water_pipeline.tif
$ unbihexium pipeline run water_detection -i scene.tif -o wp.tif -p weights=runs/water_surface_detector_tiny/best.pt
Completed: 65b0ae66-e2ab-4564-8586-db853dfac9c7 -> wp.tif
$ unbihexium pipeline run change_detection -i rgb.tif --input2 rgb_later.tif -o change_pipeline.tif -p variant=tiny -p threshold=0.4
Completed: 45245f3f-dd64-4290-99c0-7875f958e9a9 -> change_pipeline.tif
```

`weights` and `variant` SHOULD NOT be combined: the variant is then appended to the checkpoint path, which fails with an unknown model error.

## 12. unbihexium serve

```text
$ unbihexium serve --help
Usage: unbihexium serve [OPTIONS]

  Start the REST service with uvicorn (needs the serving extra).

Options:
  --host TEXT      Listen address; default from the serving configuration.
  --port INTEGER   Port; default from the serving configuration.
  --config FILE    YAML configuration file; default UNBIHEXIUM_CONFIG.
  --proxy-headers  Trust X-Forwarded-* headers from a proxy.
  --help           Show this message and exit.
```

Starts the FastAPI application of `unbihexium.serving` with uvicorn and serves until it is interrupted. The settings are loaded from `unbihexium.config`: `--config FILE` is exported as `UNBIHEXIUM_CONFIG` for the process, then defaults, the file and the `UNBIHEXIUM_<SECTION>__<KEY>` variables are combined. `--host` and `--port` override `serving.host` and `serving.port` (defaults 127.0.0.1 and 8000), and `log_level` becomes the log level of uvicorn. `--proxy-headers` lets uvicorn take the client address from `X-Forwarded-For`, which SHOULD be used only behind a trusted reverse proxy. Without the `serving` extra the command fails with status 1. The routes, request limits, API key, rate limit and CORS settings are described in [docs/getting_started/configuration.md](../getting_started/configuration.md#8-rest-service-settings); the OpenAPI document is served at `/docs` and `/openapi.json`.

```text
$ unbihexium serve --port 8791 &
$ curl -s http://127.0.0.1:8791/health
{"status":"healthy","version":"1.0.1","ready":true,"models_available":520,"models_loaded":0}
```

The service has no API key, no rate limit and allows every CORS origin by default; before it is reachable beyond a trusted network, `UNBIHEXIUM_SERVING__API_KEY` and `UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE` SHOULD be set.

## 13. Hidden compatibility aliases

Two commands of earlier releases remain available but are not listed by `--help`:

| Alias | Equivalent | Options |
| --- | --- | --- |
| `unbihexium infer MODEL_ID -i INPUT -o OUTPUT [-t TASK]` | `unbihexium predict MODEL_ID INPUT OUTPUT` with default options | `-t/--task` is ignored; the task follows from the model |
| `unbihexium zoo download MODEL_ID [-f] [--cache-dir PATH]` | `unbihexium zoo build MODEL_ID [--force] [--cache-dir PATH]` | No ONNX export |

New scripts SHOULD use `predict` and `zoo build`.

## 14. Environment and shell completion

### 14.1 Environment variables

The command reads `UNBIHEXIUM_CACHE`, the root of the model store (default `~/.cache/unbihexium`, models in its subdirectory `models/`). Only `unbihexium serve` also reads the settings of `unbihexium.config` (`UNBIHEXIUM_CONFIG`, `UNBIHEXIUM_<SECTION>__<KEY>` and `UNBIHEXIUM_LOG_LEVEL`); see [docs/getting_started/configuration.md](../getting_started/configuration.md). Terminal colours and table widths follow the terminal as detected by the `rich` library; `COLUMNS` sets the width of tables and wrapped messages.

### 14.2 Shell completion

Click generates completion scripts for Bash, Zsh and Fish from the command definitions [1]. For Bash, add the following line to `~/.bashrc`:

```bash
eval "$(_UNBIHEXIUM_COMPLETE=bash_source unbihexium)"
```

Use `zsh_source` or `fish_source` for the other shells. The file `scripts/unbihexium-completion.bash` in the repository contains the same Bash function with comments; because it asks the installed command for its candidates, it always matches the installed version.

## References

[1] Pallets. Click documentation: Shell Completion. 2026. <https://click.palletsprojects.com/en/stable/shell-completion/>

[2] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[3] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[4] European Space Agency. Sentinel-2 User Handbook, Issue 1 Revision 2. 2015. <https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook>

[5] J. W. Rouse, R. H. Haas, J. A. Schell and D. W. Deering. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[6] A. Huete, K. Didan, T. Miura, E. P. Rodriguez, X. Gao and L. G. Ferreira. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sensing of Environment 83(1-2), 195-213. 2002. <https://doi.org/10.1016/S0034-4257(02)00096-2>

[7] S. K. McFeeters. The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing 17(7), 1425-1432. 1996. <https://doi.org/10.1080/01431169608948714>

[8] ONNX Project Contributors. Open Neural Network Exchange (ONNX). 2026. <https://onnx.ai/>

[9] H. Butler, M. Daly, A. Doyle, S. Gillies, S. Hagen and T. Schaub. RFC 7946: The GeoJSON Format. IETF, 2016. <https://www.rfc-editor.org/rfc/rfc7946>

<!--
=============================================================================
End of file docs/reference/cli.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
