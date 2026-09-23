# Super-Resolution

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `super_resolution` |
| Task | super_resolution |
| Domain | imaging |
| Architecture | residual_subpixel |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Increases the spatial resolution of RGB imagery by a factor of four.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K, sH, sW) at s times the input resolution.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `red` | - |
| 1 | `green` | - |
| 2 | `blue` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `super_resolution_tiny` | 134,992 | 256 | `3d992b6eebd75e54` |
| `super_resolution_base` | 657,264 | 256 | `9977f84d1fa420f6` |
| `super_resolution_large` | 2,784,528 | 512 | `5254b6e34d269585` |
| `super_resolution_mega` | 6,109,872 | 512 | `f8bf8132935aefe0` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("super_resolution_base")  # verified starter weights
```

## Training

Required reference data: High resolution reference images.

```bash
unbihexium train super_resolution_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Pairs of low and high resolution RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
