# Hydroelectric Reservoir Level Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `hydroelectric_monitor` |
| Task | dense_regression |
| Domain | energy |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates water surface elevation of reservoirs.

## Inputs

5 channels, float32, shape (N, 5, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`
5. `elevation`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `water_surface_elevation` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `hydroelectric_monitor_tiny` | 733,217 | 256 | `023633fb7cd6098e` |
| `hydroelectric_monitor_base` | 7,059,009 | 256 | `b1026387fdcb363f` |
| `hydroelectric_monitor_large` | 22,059,937 | 512 | `14ea833799a5c3fd` |
| `hydroelectric_monitor_mega` | 60,451,713 | 512 | `7208aa646eb293fb` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("hydroelectric_monitor_base")  # verified starter weights
```

## Training

Required reference data: Reservoir level records.

```bash
unbihexium train hydroelectric_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
