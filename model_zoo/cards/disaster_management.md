# Disaster Impact Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `disaster_management` |
| Task | scene_regression |
| Domain | risk |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the fraction of an area affected by a disaster.

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
| 0 | `affected_fraction` | 1 |

Outputs are bounded to [0.0, 1.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `disaster_management_tiny` | 495,489 | 256 | `b3f38a14f38f502a` |
| `disaster_management_base` | 3,747,073 | 256 | `06b7bdc48cb4713d` |
| `disaster_management_large` | 14,609,089 | 512 | `640b137d2c061705` |
| `disaster_management_mega` | 37,766,401 | 512 | `ac8395efc2a2397f` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("disaster_management_base")  # verified starter weights
```

## Training

Required reference data: Affected area fractions from damage assessments.

```bash
unbihexium train disaster_management_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
