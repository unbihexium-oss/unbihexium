# Utility Mapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `utility_mapper` |
| Task | segmentation |
| Domain | assets |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments power lines, pipeline corridors and substations.

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
| 1 | `power_line` | - |
| 2 | `pipeline_corridor` | - |
| 3 | `substation` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `utility_mapper_tiny` | 732,980 | 256 | `fb5efffedf73011e` |
| `utility_mapper_base` | 7,058,532 | 256 | `a252fe69151e816d` |
| `utility_mapper_large` | 22,059,220 | 512 | `4ae023127252c720` |
| `utility_mapper_mega` | 60,450,756 | 512 | `d1bece03b36705c9` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("utility_mapper_base")  # verified starter weights
```

## Training

Required reference data: Utility masks.

```bash
unbihexium train utility_mapper_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
