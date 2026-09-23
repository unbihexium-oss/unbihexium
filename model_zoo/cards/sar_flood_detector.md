# SAR Flood Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_flood_detector` |
| Task | segmentation |
| Domain | sar |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps flooded areas in radar backscatter.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `VV`
2. `VH`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `non_flooded` | - |
| 1 | `flooded` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_flood_detector_tiny` | 732,802 | 256 | `8cb4bef626f8e2b6` |
| `sar_flood_detector_base` | 7,058,178 | 256 | `cd201256c738235d` |
| `sar_flood_detector_large` | 22,058,690 | 512 | `a56c5ae23375e40c` |
| `sar_flood_detector_mega` | 60,450,050 | 512 | `1038f575654e3f49` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_flood_detector_base")  # verified starter weights
```

## Training

Required reference data: Flood masks.

```bash
unbihexium train sar_flood_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD (sigma0 in dB)

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
