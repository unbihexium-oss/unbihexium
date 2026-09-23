# DEM from Stereo

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `dem_generator` |
| Task | dense_regression |
| Domain | imaging |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates terrain elevation from a panchromatic stereo pair.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `pan_t1`
2. `pan_t2`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `elevation` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `dem_generator_tiny` | 732,785 | 256 | `7cef49a6076fe6eb` |
| `dem_generator_base` | 7,058,145 | 256 | `3b3c0112bda94efc` |
| `dem_generator_large` | 22,058,641 | 512 | `7f858eb0f820ce6c` |
| `dem_generator_mega` | 60,449,985 | 512 | `e6d192cc1323f71b` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("dem_generator_base")  # verified starter weights
```

## Training

Required reference data: Reference DEMs, for example LiDAR.

```bash
unbihexium train dem_generator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Epipolar-rectified panchromatic stereo pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
