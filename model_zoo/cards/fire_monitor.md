# Active Fire Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `fire_monitor` |
| Task | detection |
| Domain | environment |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects active fire fronts and hotspots in short-wave infrared composites.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `B12`
2. `B8A`
3. `B04`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `active_fire` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `fire_monitor_tiny` | 730,581 | 256 | `f07c42bd7e34d780` |
| `fire_monitor_base` | 7,049,125 | 256 | `98cdb80cdda80cd6` |
| `fire_monitor_large` | 22,038,197 | 512 | `e192caed083dae29` |
| `fire_monitor_mega` | 60,413,509 | 512 | `9d2173610d42705f` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("fire_monitor_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of active fire areas.

```bash
unbihexium train fire_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L1C or L2A
- Landsat 8/9

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
