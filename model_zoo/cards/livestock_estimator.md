# Livestock Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `livestock_estimator` |
| Task | scene_regression |
| Domain | agriculture |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the number of livestock in a chip.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `head_count` | animals |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `livestock_estimator_tiny` | 494,481 | 256 | `65dfec3048c347ee` |
| `livestock_estimator_base` | 3,745,057 | 256 | `1fb5ced7b33c9858` |
| `livestock_estimator_large` | 14,606,065 | 512 | `f6a07af02917c873` |
| `livestock_estimator_mega` | 37,762,369 | 512 | `dc61bb8cc359a352` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("livestock_estimator_base")  # verified starter weights
```

## Training

Required reference data: Livestock counts per chip.

```bash
unbihexium train livestock_estimator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
