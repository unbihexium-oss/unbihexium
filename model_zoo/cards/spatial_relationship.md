# Proximity Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `spatial_relationship` |
| Task | dense_regression |
| Domain | analysis |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the distance to the nearest mapped feature of interest.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `distance` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `spatial_relationship_tiny` | 733,073 | 256 | `34a626d26f59a6f1` |
| `spatial_relationship_base` | 7,058,721 | 256 | `5b5080873cae7ce4` |
| `spatial_relationship_large` | 22,059,505 | 512 | `0368cb8a6a3c087a` |
| `spatial_relationship_mega` | 60,451,137 | 512 | `4837ca2f8e275f88` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("spatial_relationship_base")  # verified starter weights
```

## Training

Required reference data: Distance rasters derived from reference vector data.

```bash
unbihexium train spatial_relationship_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
