# Maritime Awareness Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `maritime_awareness` |
| Task | detection |
| Domain | defense |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects vessels and offshore platforms for maritime domain awareness.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `vessel` | - |
| 1 | `offshore_platform` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `maritime_awareness_tiny` | 730,614 | 256 | `c06677f0608186b4` |
| `maritime_awareness_base` | 7,049,190 | 256 | `18cfca599543c261` |
| `maritime_awareness_large` | 22,038,294 | 512 | `ea1a63a644f1db5b` |
| `maritime_awareness_mega` | 60,413,638 | 512 | `b3e384d172bfa174` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("maritime_awareness_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of vessels and platforms.

```bash
unbihexium train maritime_awareness_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
