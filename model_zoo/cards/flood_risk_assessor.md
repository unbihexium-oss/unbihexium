# Flood Depth Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `flood_risk_assessor` |
| Task | dense_regression |
| Domain | water |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates flood water depth.

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
| 0 | `water_depth` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `flood_risk_assessor_tiny` | 734,225 | 256 | `a2fc58c46dfdedab` |
| `flood_risk_assessor_base` | 7,061,025 | 256 | `6dc538b46911625c` |
| `flood_risk_assessor_large` | 22,062,961 | 512 | `fd7a706be8d0cd35` |
| `flood_risk_assessor_mega` | 60,455,745 | 512 | `cf1547c89f2514af` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("flood_risk_assessor_base")  # verified starter weights
```

## Training

Required reference data: Flood depth rasters from hydraulic models or surveys.

```bash
unbihexium train flood_risk_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
