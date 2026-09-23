# Generic Object Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `object_detector` |
| Task | detection |
| Domain | ai |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Multi-class detector for common objects in overhead imagery.

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
| 1 | `vehicle` | - |
| 2 | `ship` | - |
| 3 | `aircraft` | - |
| 4 | `storage_tank` | - |
| 5 | `bridge` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `object_detector_tiny` | 730,746 | 256 | `0908c3d9e8120f99` |
| `object_detector_base` | 7,049,450 | 256 | `72b8f4f99e9a5e4e` |
| `object_detector_large` | 22,038,682 | 512 | `c196b30570fa3ce2` |
| `object_detector_mega` | 60,414,154 | 512 | `bf03ed34396bf550` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("object_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of the six object classes.

```bash
unbihexium train object_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
