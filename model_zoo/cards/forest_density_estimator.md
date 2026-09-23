# Forest Density Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `forest_density_estimator` |
| Task | dense_regression |
| Domain | forestry |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates tree canopy cover.

## Inputs

10 channels, float32, shape (N, 10, H, W):

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

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `canopy_cover` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `forest_density_estimator_tiny` | 733,937 | 256 | `5d88e2cd5d37b224` |
| `forest_density_estimator_base` | 7,060,449 | 256 | `ea62e26be18ddf91` |
| `forest_density_estimator_large` | 22,062,097 | 512 | `1567584361645295` |
| `forest_density_estimator_mega` | 60,454,593 | 512 | `e6b4ce48bb0c3322` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("forest_density_estimator_base")  # verified starter weights
```

## Training

Required reference data: Canopy cover fraction from LiDAR or high resolution imagery.

```bash
unbihexium train forest_density_estimator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
