# Building Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `building_detector` |
| Task | detection |
| Domain | urban |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects individual buildings as bounding boxes.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `building` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `building_detector_tiny` | 730,581 | 256 | `f2eda71812b340ad` |
| `building_detector_base` | 7,049,125 | 256 | `be8ccba216a4e95f` |
| `building_detector_large` | 22,038,197 | 512 | `42117c7487e2c3de` |
| `building_detector_mega` | 60,413,509 | 512 | `ff999a8b6ee9c2f1` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("building_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of building footprints.

```bash
unbihexium train building_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery
- 0.3 to 1 m

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
