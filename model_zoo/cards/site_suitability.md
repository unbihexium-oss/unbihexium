# Site Suitability

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `site_suitability` |
| Task | dense_regression |
| Domain | analysis |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores general site suitability for development.

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
| 0 | `suitability` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `site_suitability_tiny` | 733,937 | 256 | `6e4c894cb7b3643b` |
| `site_suitability_base` | 7,060,449 | 256 | `e35ffc57d3e136d9` |
| `site_suitability_large` | 22,062,097 | 512 | `e35dcb805c63cd64` |
| `site_suitability_mega` | 60,454,593 | 512 | `9b759070ce0ca391` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("site_suitability_base")  # verified starter weights
```

## Training

Required reference data: Suitability scores from multi-criteria analysis.

```bash
unbihexium train site_suitability_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM
- road distance rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
