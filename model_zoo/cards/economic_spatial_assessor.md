# Property Value Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `economic_spatial_assessor` |
| Task | scene_regression |
| Domain | analysis |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates median property value of an area.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `median_value` | currency m-2 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `economic_spatial_assessor_tiny` | 494,481 | 256 | `2ff065cce8c5be3e` |
| `economic_spatial_assessor_base` | 3,745,057 | 256 | `6e8167e448e26323` |
| `economic_spatial_assessor_large` | 14,606,065 | 512 | `70912f0e6f7064f6` |
| `economic_spatial_assessor_mega` | 37,762,369 | 512 | `f9795d6d0a1d4084` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("economic_spatial_assessor_base")  # verified starter weights
```

## Training

Required reference data: Property transaction statistics aggregated to chips.

```bash
unbihexium train economic_spatial_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
