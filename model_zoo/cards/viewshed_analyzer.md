# Visibility Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `viewshed_analyzer` |
| Task | dense_regression |
| Domain | tourism |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the fraction of the surrounding area visible from each pixel.

## Inputs

1 channels, float32, shape (N, 1, H, W):

1. `elevation`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `visible_fraction` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `viewshed_analyzer_tiny` | 732,641 | 256 | `042a2cbe3266348e` |
| `viewshed_analyzer_base` | 7,057,857 | 256 | `3058491fbce3824e` |
| `viewshed_analyzer_large` | 22,058,209 | 512 | `8116adff75828b28` |
| `viewshed_analyzer_mega` | 60,449,409 | 512 | `8747b85f567acb75` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("viewshed_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Visibility rasters from viewshed analysis.

```bash
unbihexium train viewshed_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Copernicus DEM GLO-30
- national DEMs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
