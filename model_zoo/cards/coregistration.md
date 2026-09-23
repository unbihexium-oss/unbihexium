# Co-registration Flow Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `coregistration` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the pixel displacement field that aligns a moving image to a reference image.

## Inputs

6 channels, float32, shape (N, 6, H, W):

1. `red_t1`
2. `green_t1`
3. `blue_t1`
4. `red_t2`
5. `green_t2`
6. `blue_t2`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `dx` | px |
| 1 | `dy` | px |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `coregistration_tiny` | 733,378 | 256 | `1fd8aa56f46a3e0f` |
| `coregistration_base` | 7,059,330 | 256 | `db8fba7f29de6ebd` |
| `coregistration_large` | 22,060,418 | 512 | `0e2da520f4cadb6f` |
| `coregistration_mega` | 60,452,354 | 512 | `6d076d2139e94632` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("coregistration_base")  # verified starter weights
```

## Training

Required reference data: Displacement fields, for example from synthetic warps of a reference image.

```bash
unbihexium train coregistration_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- RGB image pairs of the same area

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
