# DSM from Stereo

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `dsm_generator` |
| Task | dense_regression |
| Domain | imaging |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates surface elevation including buildings and trees from a stereo pair.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `pan_t1`
2. `pan_t2`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `surface_elevation` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `dsm_generator_tiny` | 732,785 | 256 | `9d7516759b69d2d6` |
| `dsm_generator_base` | 7,058,145 | 256 | `b2c644df2849605d` |
| `dsm_generator_large` | 22,058,641 | 512 | `86e5271d2c1b947c` |
| `dsm_generator_mega` | 60,449,985 | 512 | `52f5bc5050c9db2f` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("dsm_generator_base")  # verified starter weights
```

## Training

Required reference data: Reference DSMs, for example LiDAR.

```bash
unbihexium train dsm_generator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Epipolar-rectified panchromatic stereo pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
