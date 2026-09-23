# 2D Digitisation

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `digitization_2d` |
| Task | segmentation |
| Domain | imaging |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments buildings, roads, water and vegetation for map digitisation.

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
| 1 | `building` | - |
| 2 | `road` | - |
| 3 | `water` | - |
| 4 | `vegetation` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `digitization_2d_tiny` | 732,997 | 256 | `4558c9f31f4e03b2` |
| `digitization_2d_base` | 7,058,565 | 256 | `03d0537b05aacc0a` |
| `digitization_2d_large` | 22,059,269 | 512 | `baf627394fc363b2` |
| `digitization_2d_mega` | 60,450,821 | 512 | `629e71d90579798b` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("digitization_2d_base")  # verified starter weights
```

## Training

Required reference data: Masks of the four map feature classes.

```bash
unbihexium train digitization_2d_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
