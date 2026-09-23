# Perennial Garden Suitability

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `perennial_garden_suitability` |
| Task | dense_regression |
| Domain | agriculture |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores land suitability for orchards and perennial crops.

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
| 0 | `suitability` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `perennial_garden_suitability_tiny` | 734,225 | 256 | `937709c66a3acf2c` |
| `perennial_garden_suitability_base` | 7,061,025 | 256 | `c10450176b61e721` |
| `perennial_garden_suitability_large` | 22,062,961 | 512 | `0437638b42b9bbe8` |
| `perennial_garden_suitability_mega` | 60,455,745 | 512 | `101bd71e9bee1ab6` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("perennial_garden_suitability_base")  # verified starter weights
```

## Training

Required reference data: Suitability scores from land evaluation.

```bash
unbihexium train perennial_garden_suitability_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
