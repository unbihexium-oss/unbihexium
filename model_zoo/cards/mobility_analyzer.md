# Mobility Analyser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `mobility_analyzer` |
| Task | dense_regression |
| Domain | urban |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates traffic intensity per pixel on the road network.

## Inputs

10 channels, float32, shape (N, 10, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B08`
5. `B11`
6. `B12`
7. `elevation`
8. `slope`
9. `population`
10. `road_distance`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `traffic_intensity` | vehicles h-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `mobility_analyzer_tiny` | 733,937 | 256 | `ac453351dc880200` |
| `mobility_analyzer_base` | 7,060,449 | 256 | `68631806686b2eed` |
| `mobility_analyzer_large` | 22,062,097 | 512 | `837e639377e8835e` |
| `mobility_analyzer_mega` | 60,454,593 | 512 | `7c870303e9e603d1` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("mobility_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Traffic counts interpolated along roads.

```bash
unbihexium train mobility_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- road distance and population rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
