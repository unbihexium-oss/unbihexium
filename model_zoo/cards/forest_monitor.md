# Forest Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `forest_monitor` |
| Task | segmentation |
| Domain | forestry |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps forest cover by forest type.

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

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `non_forest` | - |
| 1 | `broadleaf` | - |
| 2 | `coniferous` | - |
| 3 | `mixed` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `forest_monitor_tiny` | 733,988 | 256 | `15c28d3cddebfe17` |
| `forest_monitor_base` | 7,060,548 | 256 | `cf183d227d6fba36` |
| `forest_monitor_large` | 22,062,244 | 512 | `dbc5d8b9770434fe` |
| `forest_monitor_mega` | 60,454,788 | 512 | `0487b7c2974eec87` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("forest_monitor_base")  # verified starter weights
```

## Training

Required reference data: Forest type masks.

```bash
unbihexium train forest_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
