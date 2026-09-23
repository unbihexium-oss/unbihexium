# Infrastructure Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `infrastructure_monitor` |
| Task | segmentation |
| Domain | assets |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments roads, railways, buildings and bridges.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `background` | - |
| 1 | `road` | - |
| 2 | `railway` | - |
| 3 | `building` | - |
| 4 | `bridge` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `infrastructure_monitor_tiny` | 732,997 | 256 | `704c2313dd9b8c84` |
| `infrastructure_monitor_base` | 7,058,565 | 256 | `5a60a7a931a5be9a` |
| `infrastructure_monitor_large` | 22,059,269 | 512 | `b60fcc002935ab9a` |
| `infrastructure_monitor_mega` | 60,450,821 | 512 | `d55b313f21a3a3bf` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("infrastructure_monitor_base")  # verified starter weights
```

## Training

Required reference data: Infrastructure masks.

```bash
unbihexium train infrastructure_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
