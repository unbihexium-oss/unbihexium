# Crop Parcel Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `crop_detector` |
| Task | detection |
| Domain | agriculture |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects cultivated parcels and orchards as bounding boxes.

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
| 0 | `crop_parcel` | - |
| 1 | `orchard` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `crop_detector_tiny` | 730,758 | 256 | `b9dd7637620ab55c` |
| `crop_detector_base` | 7,049,478 | 256 | `388283deb110251e` |
| `crop_detector_large` | 22,038,726 | 512 | `54bdcec5753dda86` |
| `crop_detector_mega` | 60,414,214 | 512 | `ce200a246c4bd449` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("crop_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of crop parcels and orchards.

```bash
unbihexium train crop_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- PlanetScope
- SPOT
- aerial RGBN imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
