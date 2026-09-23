# Energy Potential

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `energy_potential` |
| Task | dense_regression |
| Domain | energy |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates annual solar energy potential per pixel.

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
| 0 | `solar_potential` | kWh m-2 a-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `energy_potential_tiny` | 734,225 | 256 | `26b6eb59188a0b75` |
| `energy_potential_base` | 7,061,025 | 256 | `04f0687ccc5ae149` |
| `energy_potential_large` | 22,062,961 | 512 | `eaf866889a7f18cf` |
| `energy_potential_mega` | 60,455,745 | 512 | `a1092a7b3d5f51b7` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("energy_potential_base")  # verified starter weights
```

## Training

Required reference data: Solar potential rasters from irradiance models.

```bash
unbihexium train energy_potential_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
