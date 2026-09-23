# Built-up Area Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `builtup_detector` |
| Task | detection |
| Domain | urban |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects compact built-up areas such as settlements and industrial sites in multispectral imagery.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `built_up_area` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `builtup_detector_tiny` | 730,725 | 256 | `398c1da594a2f8f7` |
| `builtup_detector_base` | 7,049,413 | 256 | `3216539785397214` |
| `builtup_detector_large` | 22,038,629 | 512 | `cb6eee51789b36d9` |
| `builtup_detector_mega` | 60,414,085 | 512 | `478025e741f31572` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("builtup_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of built-up areas.

```bash
unbihexium train builtup_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- PlanetScope
- SPOT
- Sentinel-2 10 m bands

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
