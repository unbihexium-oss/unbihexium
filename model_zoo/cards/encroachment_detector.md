# Encroachment Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `encroachment_detector` |
| Task | detection |
| Domain | assets |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects structures and vehicles inside protected corridors such as pipeline and power line rights of way.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `structure` | - |
| 1 | `vehicle` | - |
| 2 | `excavation` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `encroachment_detector_tiny` | 730,647 | 256 | `f51d5e3bc7ea8ab5` |
| `encroachment_detector_base` | 7,049,255 | 256 | `4729c16b77eae29b` |
| `encroachment_detector_large` | 22,038,391 | 512 | `7a6a11f4c623fdbb` |
| `encroachment_detector_mega` | 60,413,767 | 512 | `bf7281658719c190` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("encroachment_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of structures, vehicles and excavations.

```bash
unbihexium train encroachment_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
