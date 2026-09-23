# SAR Ship Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_ship_detector` |
| Task | detection |
| Domain | sar |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects ships in synthetic aperture radar backscatter, independent of cloud cover and daylight.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `VV`
2. `VH`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `ship` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_ship_detector_tiny` | 730,437 | 256 | `ab387c54219cc513` |
| `sar_ship_detector_base` | 7,048,837 | 256 | `5a15158cf77437fa` |
| `sar_ship_detector_large` | 22,037,765 | 512 | `425d657f352865e1` |
| `sar_ship_detector_mega` | 60,412,933 | 512 | `dc0bfbfdf1665667` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_ship_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of ships in SAR images.

```bash
unbihexium train sar_ship_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD (sigma0 in dB)
- other C- or X-band SAR

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
