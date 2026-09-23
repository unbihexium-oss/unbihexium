# DSM from Tri-Stereo

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `tri_stereo_processor` |
| Task | dense_regression |
| Domain | imaging |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates surface elevation from a tri-stereo acquisition.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `pan_forward`
2. `pan_nadir`
3. `pan_backward`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `surface_elevation` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `tri_stereo_processor_tiny` | 732,929 | 256 | `f95e7601951c7cfd` |
| `tri_stereo_processor_base` | 7,058,433 | 256 | `5317c09300949edd` |
| `tri_stereo_processor_large` | 22,059,073 | 512 | `0ea9cf065ef30e6e` |
| `tri_stereo_processor_mega` | 60,450,561 | 512 | `ade61cbb56a909fc` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("tri_stereo_processor_base")  # verified starter weights
```

## Training

Required reference data: Reference DSMs.

```bash
unbihexium train tri_stereo_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Epipolar-rectified tri-stereo panchromatic imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
