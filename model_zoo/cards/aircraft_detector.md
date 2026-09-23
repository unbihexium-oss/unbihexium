# Aircraft Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `aircraft_detector` |
| Task | detection |
| Domain | ai |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects aircraft on aprons, runways and in flight in very high resolution optical imagery.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `aircraft` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `aircraft_detector_tiny` | 730,581 | 256 | `2aff48b3ea0d3144` |
| `aircraft_detector_base` | 7,049,125 | 256 | `81eae07c9662fc86` |
| `aircraft_detector_large` | 22,038,197 | 512 | `5d1fb2e5295d24b6` |
| `aircraft_detector_mega` | 60,413,509 | 512 | `fda24e591f2d1169` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("aircraft_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of aircraft.

```bash
unbihexium train aircraft_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery
- 0.3 to 1 m

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
