# Learned Orthorectification

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `orthorectification` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Produces an orthorectified image from a raw image and a DEM.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `red`
2. `green`
3. `blue`
4. `elevation`

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
| `orthorectification_tiny` | 733,107 | 256 | `412f0b8a49382308` |
| `orthorectification_base` | 7,058,787 | 256 | `f18ccb8f0429706b` |
| `orthorectification_large` | 22,059,603 | 512 | `80c53359d1d4e57e` |
| `orthorectification_mega` | 60,451,267 | 512 | `18d13f351c760556` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("orthorectification_base")  # verified starter weights
```

## Training

Required reference data: Orthorectified reference images.

```bash
unbihexium train orthorectification_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Raw imagery with a co-registered DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
