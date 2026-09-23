# Crop Health Assessor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `crop_health_assessor` |
| Task | dense_regression |
| Domain | agriculture |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Scores crop health per pixel.

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
| 0 | `health_score` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `crop_health_assessor_tiny` | 733,937 | 256 | `bafa66ea9630fa6c` |
| `crop_health_assessor_base` | 7,060,449 | 256 | `9e21b03166f3a4c2` |
| `crop_health_assessor_large` | 22,062,097 | 512 | `d2f7c6efb92441e4` |
| `crop_health_assessor_mega` | 60,454,593 | 512 | `68873cc562ebd002` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("crop_health_assessor_base")  # verified starter weights
```

## Training

Required reference data: Crop health scores from field inspection.

```bash
unbihexium train crop_health_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
