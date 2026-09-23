# Transportation Mapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `transportation_mapper` |
| Task | segmentation |
| Domain | urban |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps roads, railways, airports and ports.

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
| 3 | `airport` | - |
| 4 | `port` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `transportation_mapper_tiny` | 732,997 | 256 | `f3ed543f5d34bcd3` |
| `transportation_mapper_base` | 7,058,565 | 256 | `84246a158a97156c` |
| `transportation_mapper_large` | 22,059,269 | 512 | `96d78953a4d9fe76` |
| `transportation_mapper_mega` | 60,450,821 | 512 | `6cc303347eaf848f` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("transportation_mapper_base")  # verified starter weights
```

## Training

Required reference data: Transport infrastructure masks.

```bash
unbihexium train transportation_mapper_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
