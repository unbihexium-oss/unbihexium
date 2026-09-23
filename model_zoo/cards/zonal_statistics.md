# Zonal Cover Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `zonal_statistics` |
| Task | scene_regression |
| Domain | analysis |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the fractional cover of vegetation, water and built-up area in a chip.

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

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `vegetation_fraction` | 1 |
| 1 | `water_fraction` | 1 |
| 2 | `built_up_fraction` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `zonal_statistics_tiny` | 495,555 | 256 | `b00c71736648950d` |
| `zonal_statistics_base` | 3,747,203 | 256 | `a0b611c7ecede719` |
| `zonal_statistics_large` | 14,609,283 | 512 | `6e8f0367219299c3` |
| `zonal_statistics_mega` | 37,766,659 | 512 | `a07fea2e7c23f5e1` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("zonal_statistics_base")  # verified starter weights
```

## Training

Required reference data: Cover fractions per chip from reference land cover maps.

```bash
unbihexium train zonal_statistics_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
