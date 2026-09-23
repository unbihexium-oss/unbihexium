# Vehicle Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `vehicle_detector` |
| Task | detection |
| Domain | ai |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects cars, trucks and buses.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `car` | - |
| 1 | `truck` | - |
| 2 | `bus` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `vehicle_detector_tiny` | 730,647 | 256 | `ace45134bb6477c1` |
| `vehicle_detector_base` | 7,049,255 | 256 | `302618924bf38f07` |
| `vehicle_detector_large` | 22,038,391 | 512 | `c97ae7f19ec468d3` |
| `vehicle_detector_mega` | 60,413,767 | 512 | `10fec40187f79831` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("vehicle_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of vehicles by type.

```bash
unbihexium train vehicle_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery
- 0.1 to 0.5 m

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
