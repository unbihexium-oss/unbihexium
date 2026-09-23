# Military Objects Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `military_objects_detector` |
| Task | detection |
| Domain | defense |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects vehicles, aircraft, vessels and fortified structures for neutral monitoring and verification.

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
| 1 | `aircraft` | - |
| 2 | `vessel` | - |
| 3 | `fortified_structure` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `military_objects_detector_tiny` | 730,680 | 256 | `da8e557122f6a489` |
| `military_objects_detector_base` | 7,049,320 | 256 | `e0916faf98614359` |
| `military_objects_detector_large` | 22,038,488 | 512 | `250c2b2b87e06c07` |
| `military_objects_detector_mega` | 60,413,896 | 512 | `9c53ae3e8d9c940a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("military_objects_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of the four object classes.

```bash
unbihexium train military_objects_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
