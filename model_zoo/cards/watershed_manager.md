# Runoff Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `watershed_manager` |
| Task | dense_regression |
| Domain | water |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates a runoff coefficient per pixel for watershed management.

## Inputs

12 channels, float32, shape (N, 12, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B05`
5. `B06`
6. `B07`
7. `B08`
8. `B8A`
9. `B11`
10. `B12`
11. `elevation`
12. `slope`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `runoff_coefficient` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `watershed_manager_tiny` | 734,225 | 256 | `2a9a91c2481cf440` |
| `watershed_manager_base` | 7,061,025 | 256 | `0b2fbc471ae2e2d3` |
| `watershed_manager_large` | 22,062,961 | 512 | `56eb558146cf7fd1` |
| `watershed_manager_mega` | 60,455,745 | 512 | `a2e82c8329b20cab` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("watershed_manager_base")  # verified starter weights
```

## Training

Required reference data: Runoff coefficients from hydrological models.

```bash
unbihexium train watershed_manager_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
