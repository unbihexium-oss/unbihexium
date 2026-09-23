# Tile Radiometric Normaliser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `raster_tiler` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Normalises the radiometry of individual tiles before tiling into a web map.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `red` | - |
| 1 | `green` | - |
| 2 | `blue` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `raster_tiler_tiny` | 732,963 | 256 | `5268c881613c557f` |
| `raster_tiler_base` | 7,058,499 | 256 | `8114bf9e2887936d` |
| `raster_tiler_large` | 22,059,171 | 512 | `7904d1bbf6dc4c55` |
| `raster_tiler_mega` | 60,450,691 | 512 | `dbe51a5aa0e4abf1` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("raster_tiler_base")  # verified starter weights
```

## Training

Required reference data: Radiometrically normalised reference tiles.

```bash
unbihexium train raster_tiler_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- RGB tiles

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
