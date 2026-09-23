# Subsidence Velocity Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_subsidence_monitor` |
| Task | dense_regression |
| Domain | sar |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates mean subsidence velocity from a stack of six displacement maps.

## Inputs

6 channels, float32, shape (N, 6, H, W):

1. `los_t1`
2. `los_t2`
3. `los_t3`
4. `los_t4`
5. `los_t5`
6. `los_t6`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `velocity` | mm a-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_subsidence_monitor_tiny` | 733,361 | 256 | `6faaf230e3c6d5bd` |
| `sar_subsidence_monitor_base` | 7,059,297 | 256 | `12ee598aecc20da1` |
| `sar_subsidence_monitor_large` | 22,060,369 | 512 | `984a5a2c1b1b8618` |
| `sar_subsidence_monitor_mega` | 60,452,289 | 512 | `6be7aae4b311d71a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_subsidence_monitor_base")  # verified starter weights
```

## Training

Required reference data: Velocity from persistent scatterer or SBAS processing.

```bash
unbihexium train sar_subsidence_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 InSAR time series

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
