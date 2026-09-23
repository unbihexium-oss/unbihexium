# Building Damage Assessor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `damage_assessor` |
| Task | detection |
| Domain | risk |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects damaged and destroyed buildings in post-event imagery for rapid damage assessment.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `damaged_building` | - |
| 1 | `destroyed_building` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `damage_assessor_tiny` | 730,614 | 256 | `0e02bc124b958f0e` |
| `damage_assessor_base` | 7,049,190 | 256 | `2e90285660f2ff9b` |
| `damage_assessor_large` | 22,038,294 | 512 | `0cd8ac975169662a` |
| `damage_assessor_mega` | 60,413,638 | 512 | `033c3101a1d39ad1` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("damage_assessor_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of buildings graded as damaged or destroyed.

```bash
unbihexium train damage_assessor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution post-event satellite or aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
