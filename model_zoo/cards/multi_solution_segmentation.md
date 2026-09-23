# General Semantic Segmentation

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `multi_solution_segmentation` |
| Task | segmentation |
| Domain | ai |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

General-purpose semantic segmentation with six common classes.

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
| 5 | `bare_ground` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `multi_solution_segmentation_tiny` | 733,014 | 256 | `ef1c3ceb84eca902` |
| `multi_solution_segmentation_base` | 7,058,598 | 256 | `760ee50b06dad5fa` |
| `multi_solution_segmentation_large` | 22,059,318 | 512 | `2e6ea30b07cf4f97` |
| `multi_solution_segmentation_mega` | 60,450,886 | 512 | `57db453795136c0a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("multi_solution_segmentation_base")  # verified starter weights
```

## Training

Required reference data: Masks of the six classes.

```bash
unbihexium train multi_solution_segmentation_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
