# DTM from DSM

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `dtm_generator` |
| Task | dense_regression |
| Domain | imaging |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Removes buildings and vegetation from a surface model to estimate bare ground elevation.

## Inputs

1 channels, float32, shape (N, 1, H, W):

1. `surface_height`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `ground_elevation` | m |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `dtm_generator_tiny` | 732,641 | 256 | `2711fc566be59637` |
| `dtm_generator_base` | 7,057,857 | 256 | `52ddbaf0e53adfc3` |
| `dtm_generator_large` | 22,058,209 | 512 | `65f88e1c7df2366f` |
| `dtm_generator_mega` | 60,449,409 | 512 | `f298126df1027389` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("dtm_generator_base")  # verified starter weights
```

## Training

Required reference data: Reference DTMs, for example LiDAR ground returns.

```bash
unbihexium train dtm_generator_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Photogrammetric or radar DSMs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
