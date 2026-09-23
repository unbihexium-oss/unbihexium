# Natural Resources Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `natural_resources_monitor` |
| Task | dense_regression |
| Domain | environment |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates above-ground biomass.

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
| 0 | `above_ground_biomass` | Mg ha-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `natural_resources_monitor_tiny` | 734,225 | 256 | `d27319da1335cfe9` |
| `natural_resources_monitor_base` | 7,061,025 | 256 | `f2980a3e09949456` |
| `natural_resources_monitor_large` | 22,062,961 | 512 | `19b4bf3a8c8247fd` |
| `natural_resources_monitor_mega` | 60,455,745 | 512 | `3f2eee761aeb22bf` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("natural_resources_monitor_base")  # verified starter weights
```

## Training

Required reference data: Biomass plots or LiDAR-derived biomass rasters.

```bash
unbihexium train natural_resources_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Sentinel-1 GRD

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
