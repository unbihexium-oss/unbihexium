# Urban Land Use Mapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `urban_planner` |
| Task | segmentation |
| Domain | urban |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps urban land use.

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
| 1 | `residential` | - |
| 2 | `commercial` | - |
| 3 | `industrial` | - |
| 4 | `green_space` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `urban_planner_tiny` | 732,997 | 256 | `412a16c713d0c962` |
| `urban_planner_base` | 7,058,565 | 256 | `cf9017655261676b` |
| `urban_planner_large` | 22,059,269 | 512 | `8220a8b857053559` |
| `urban_planner_mega` | 60,450,821 | 512 | `5e6da9101ca19a9c` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("urban_planner_base")  # verified starter weights
```

## Training

Required reference data: Urban land use masks.

```bash
unbihexium train urban_planner_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
