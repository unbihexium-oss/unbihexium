# Ground Displacement Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `ground_displacement` |
| Task | dense_regression |
| Domain | sar |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates line-of-sight ground displacement from a wrapped interferogram.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `cos_phase`
2. `sin_phase`
3. `coherence`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `los_displacement` | mm |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `ground_displacement_tiny` | 732,929 | 256 | `5f012425b3fc7e2c` |
| `ground_displacement_base` | 7,058,433 | 256 | `1ca8eea69f89bcc0` |
| `ground_displacement_large` | 22,059,073 | 512 | `76646afd2a20587c` |
| `ground_displacement_mega` | 60,450,561 | 512 | `c6ac3460fcd32826` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("ground_displacement_base")  # verified starter weights
```

## Training

Required reference data: Unwrapped displacement from InSAR processing or GNSS.

```bash
unbihexium train ground_displacement_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 SLC interferograms

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
