# Water Quality Assessor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `water_quality_assessor` |
| Task | dense_regression |
| Domain | water |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates chlorophyll-a concentration and turbidity in water bodies.

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
| 0 | `chlorophyll_a` | mg m-3 |
| 1 | `turbidity` | FNU |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `water_quality_assessor_tiny` | 733,954 | 256 | `28d5cc931ec67edf` |
| `water_quality_assessor_base` | 7,060,482 | 256 | `556f7848d6629f85` |
| `water_quality_assessor_large` | 22,062,146 | 512 | `d27c1dcc7022093a` |
| `water_quality_assessor_mega` | 60,454,658 | 512 | `9cc5040098f1714e` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("water_quality_assessor_base")  # verified starter weights
```

## Training

Required reference data: In situ water quality samples.

```bash
unbihexium train water_quality_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
