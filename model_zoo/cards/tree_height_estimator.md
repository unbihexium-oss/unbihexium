# Canopy Height Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `tree_height_estimator` |
| Task | dense_regression |
| Domain | forestry |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates canopy height from optical and radar data.

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
11. `VV`
12. `VH`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `canopy_height` | m |

Outputs are bounded to [0.0, 60.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `tree_height_estimator_tiny` | 734,225 | 256 | `83ff75687ca30c52` |
| `tree_height_estimator_base` | 7,061,025 | 256 | `d99e479c31d30a1a` |
| `tree_height_estimator_large` | 22,062,961 | 512 | `a1526ad385bd3d36` |
| `tree_height_estimator_mega` | 60,455,745 | 512 | `e9f1db06849d98b9` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("tree_height_estimator_base")  # verified starter weights
```

## Training

Required reference data: Canopy height from airborne or spaceborne LiDAR.

```bash
unbihexium train tree_height_estimator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Sentinel-1 GRD

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
