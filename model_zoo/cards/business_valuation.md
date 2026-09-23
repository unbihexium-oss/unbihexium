# Economic Activity Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `business_valuation` |
| Task | scene_regression |
| Domain | analysis |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates an economic activity index of an area from imagery.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `activity_index` | 1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `business_valuation_tiny` | 494,625 | 256 | `7ce1c3ef9d13c2bb` |
| `business_valuation_base` | 3,745,345 | 256 | `50cbc4e2ab9a289f` |
| `business_valuation_large` | 14,606,497 | 512 | `8d7330be36baa260` |
| `business_valuation_mega` | 37,762,945 | 512 | `d1d06d6e0343c509` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("business_valuation_base")  # verified starter weights
```

## Training

Required reference data: Economic indicators aggregated to chips.

```bash
unbihexium train business_valuation_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- night lights composites

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
