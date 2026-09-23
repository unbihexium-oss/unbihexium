# Seismic Risk

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `seismic_risk` |
| Task | dense_regression |
| Domain | risk |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores seismic risk from exposure and terrain.

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
| 0 | `risk_score` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `seismic_risk_tiny` | 733,937 | 256 | `ebc8cac01fa1e33c` |
| `seismic_risk_base` | 7,060,449 | 256 | `2ced93fed2c94627` |
| `seismic_risk_large` | 22,062,097 | 512 | `b49844d8e9ce0b57` |
| `seismic_risk_mega` | 60,454,593 | 512 | `825018c40b201c5a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("seismic_risk_base")  # verified starter weights
```

## Training

Required reference data: Seismic risk scores from hazard and exposure models.

```bash
unbihexium train seismic_risk_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM
- population rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
