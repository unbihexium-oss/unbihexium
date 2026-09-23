# Surface Temperature Anomaly

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `onshore_monitor` |
| Task | dense_regression |
| Domain | energy |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates land surface temperature anomalies around onshore industrial facilities.

## Inputs

7 channels, float32, shape (N, 7, H, W):

1. `SR_B2`
2. `SR_B3`
3. `SR_B4`
4. `SR_B5`
5. `SR_B6`
6. `SR_B7`
7. `ST_B10`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `temperature_anomaly` | K |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `onshore_monitor_tiny` | 733,505 | 256 | `17b2cf445cc758ec` |
| `onshore_monitor_base` | 7,059,585 | 256 | `d43f55b617fcedd3` |
| `onshore_monitor_large` | 22,060,801 | 512 | `ef78cf86f6db730d` |
| `onshore_monitor_mega` | 60,452,865 | 512 | `67f3ff455ba4b446` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("onshore_monitor_base")  # verified starter weights
```

## Training

Required reference data: Surface temperature anomaly rasters.

```bash
unbihexium train onshore_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Landsat 8/9 Collection 2 Level 2

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
