# Wildlife Habitat Suitability

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `wildlife_habitat_analyzer` |
| Task | dense_regression |
| Domain | environment |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores habitat suitability for a target species.

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
| 0 | `habitat_suitability` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `wildlife_habitat_analyzer_tiny` | 734,225 | 256 | `b5553f676697ba0a` |
| `wildlife_habitat_analyzer_base` | 7,061,025 | 256 | `b8e53acd2c9e04b3` |
| `wildlife_habitat_analyzer_large` | 22,062,961 | 512 | `04baebf01ddb5542` |
| `wildlife_habitat_analyzer_mega` | 60,455,745 | 512 | `3b6153490bce14dd` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("wildlife_habitat_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Species occurrence data converted to suitability targets.

```bash
unbihexium train wildlife_habitat_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
