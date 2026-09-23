# Corridor Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `corridor_monitor` |
| Task | segmentation |
| Domain | assets |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments vegetation encroachment and structures in infrastructure corridors.

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
| 1 | `vegetation_encroachment` | - |
| 2 | `structure` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `corridor_monitor_tiny` | 732,963 | 256 | `ec2bc9ff88a8991f` |
| `corridor_monitor_base` | 7,058,499 | 256 | `ba47e3406cc352a1` |
| `corridor_monitor_large` | 22,059,171 | 512 | `bcd3af775ed9e810` |
| `corridor_monitor_mega` | 60,450,691 | 512 | `d55e90f1478f3d71` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("corridor_monitor_base")  # verified starter weights
```

## Training

Required reference data: Masks of vegetation encroachment and structures.

```bash
unbihexium train corridor_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery
- LiDAR-derived orthophotos

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
