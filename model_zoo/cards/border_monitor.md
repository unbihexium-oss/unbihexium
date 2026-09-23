# Border Area Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `border_monitor` |
| Task | detection |
| Domain | defense |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects vehicles, vessels and new structures along borders and coastlines for situational awareness.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `vehicle` | - |
| 1 | `vessel` | - |
| 2 | `structure` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `border_monitor_tiny` | 730,647 | 256 | `02a7f3bdc9dd3afb` |
| `border_monitor_base` | 7,049,255 | 256 | `d78418c944e8e0d3` |
| `border_monitor_large` | 22,038,391 | 512 | `2aa0509b87a47bf9` |
| `border_monitor_mega` | 60,413,767 | 512 | `eedae95b2333e254` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("border_monitor_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of vehicles, vessels and structures.

```bash
unbihexium train border_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery
- 0.3 to 1 m

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
