# Drought Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `drought_monitor` |
| Task | dense_regression |
| Domain | environment |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates a drought severity index.

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
| 0 | `drought_severity` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `drought_monitor_tiny` | 733,937 | 256 | `3bc526979c8d9afc` |
| `drought_monitor_base` | 7,060,449 | 256 | `605e7f0d127e2a4e` |
| `drought_monitor_large` | 22,062,097 | 512 | `13a10c5806c0bff2` |
| `drought_monitor_mega` | 60,454,593 | 512 | `d7a3c0baa8dc8554` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("drought_monitor_base")  # verified starter weights
```

## Training

Required reference data: Drought index rasters, for example from soil moisture products.

```bash
unbihexium train drought_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Landsat 8/9

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
