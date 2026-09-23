# Target Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `target_detector` |
| Task | detection |
| Domain | defense |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Single-class detector for a user-defined object of interest.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `object_of_interest` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `target_detector_tiny` | 730,581 | 256 | `a607765a5a079f28` |
| `target_detector_base` | 7,049,125 | 256 | `f0a0f0bad4b91a5d` |
| `target_detector_large` | 22,038,197 | 512 | `f53fc2076c5a50fa` |
| `target_detector_mega` | 60,413,509 | 512 | `01fa987b13593a33` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("target_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of the object of interest.

```bash
unbihexium train target_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
