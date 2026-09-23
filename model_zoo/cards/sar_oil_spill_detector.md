# SAR Oil Spill Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_oil_spill_detector` |
| Task | segmentation |
| Domain | sar |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments oil spills and look-alike dark areas on the sea surface.

## Inputs

1 channels, float32, shape (N, 1, H, W):

1. `VV`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `sea` | - |
| 1 | `oil_spill` | - |
| 2 | `look_alike` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_oil_spill_detector_tiny` | 732,675 | 256 | `34e6315e38428d44` |
| `sar_oil_spill_detector_base` | 7,057,923 | 256 | `8e3dba451538dab9` |
| `sar_oil_spill_detector_large` | 22,058,307 | 512 | `09b1edf6c7018fe0` |
| `sar_oil_spill_detector_mega` | 60,449,539 | 512 | `166cc2baf3aa8ee6` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_oil_spill_detector_base")  # verified starter weights
```

## Training

Required reference data: Oil spill and look-alike masks.

```bash
unbihexium train sar_oil_spill_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD VV

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
