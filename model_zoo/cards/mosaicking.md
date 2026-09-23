# Seamline Blender

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `mosaicking` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Blends two overlapping scenes into one seamless image.

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
| 0 | `red` | - |
| 1 | `green` | - |
| 2 | `blue` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `mosaicking_tiny` | 733,395 | 256 | `403614302e879610` |
| `mosaicking_base` | 7,059,363 | 256 | `fbfb7266ae6ae528` |
| `mosaicking_large` | 22,060,467 | 512 | `459b98625b89ef2d` |
| `mosaicking_mega` | 60,452,419 | 512 | `ec0ef3dd1fb5f478` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("mosaicking_base")  # verified starter weights
```

## Training

Required reference data: Seamlessly blended reference mosaics.

```bash
unbihexium train mosaicking_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Overlapping RGB scenes

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
