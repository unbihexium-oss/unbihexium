# Stereo Disparity Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `stereo_processor` |
| Task | dense_regression |
| Domain | imaging |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates disparity between the two images of an epipolar stereo pair.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `pan_t1`
2. `pan_t2`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `disparity` | px |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `stereo_processor_tiny` | 732,785 | 256 | `bc84236000a00763` |
| `stereo_processor_base` | 7,058,145 | 256 | `028350c0c0f29bd5` |
| `stereo_processor_large` | 22,058,641 | 512 | `a856f5b52ef5f280` |
| `stereo_processor_mega` | 60,449,985 | 512 | `83bf5b565608c22d` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("stereo_processor_base")  # verified starter weights
```

## Training

Required reference data: Reference disparity maps.

```bash
unbihexium train stereo_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Epipolar-rectified stereo pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
