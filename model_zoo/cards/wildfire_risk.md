# Wildfire Risk

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `wildfire_risk` |
| Task | dense_regression |
| Domain | risk |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates wildfire susceptibility.

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
| 0 | `susceptibility` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `wildfire_risk_tiny` | 734,225 | 256 | `33faf6ca26e52932` |
| `wildfire_risk_base` | 7,061,025 | 256 | `14fd53f7ebcf0ebd` |
| `wildfire_risk_large` | 22,062,961 | 512 | `ddc112b4846a51bd` |
| `wildfire_risk_mega` | 60,455,745 | 512 | `87734b483cb3f584` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("wildfire_risk_base")  # verified starter weights
```

## Training

Required reference data: Burned area history converted to susceptibility targets.

```bash
unbihexium train wildfire_risk_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
