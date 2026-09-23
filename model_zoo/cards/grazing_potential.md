# Grazing Potential

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `grazing_potential` |
| Task | dense_regression |
| Domain | agriculture |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates forage biomass available for grazing.

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

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `forage_biomass` | kg ha-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `grazing_potential_tiny` | 733,937 | 256 | `23252e07c03ca52e` |
| `grazing_potential_base` | 7,060,449 | 256 | `53b0a9ccf21aee3f` |
| `grazing_potential_large` | 22,062,097 | 512 | `065da7ba6a061d3d` |
| `grazing_potential_mega` | 60,454,593 | 512 | `088328ece2f1b745` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("grazing_potential_base")  # verified starter weights
```

## Training

Required reference data: Biomass samples from rangeland surveys.

```bash
unbihexium train grazing_potential_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
