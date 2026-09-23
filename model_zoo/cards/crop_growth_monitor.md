# Crop Growth Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `crop_growth_monitor` |
| Task | dense_regression |
| Domain | agriculture |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates leaf area index as a measure of crop growth.

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
| 0 | `leaf_area_index` | m2 m-2 |

Outputs are bounded to [0.0, 8.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `crop_growth_monitor_tiny` | 733,937 | 256 | `41415443d7adb45b` |
| `crop_growth_monitor_base` | 7,060,449 | 256 | `9335837c4c8bbe75` |
| `crop_growth_monitor_large` | 22,062,097 | 512 | `c2add671dd1b168d` |
| `crop_growth_monitor_mega` | 60,454,593 | 512 | `90522b169abc57a8` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("crop_growth_monitor_base")  # verified starter weights
```

## Training

Required reference data: Leaf area index measurements or reference products.

```bash
unbihexium train crop_growth_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
