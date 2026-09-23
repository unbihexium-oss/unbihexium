# Pansharpening

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `pansharpening` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Fuses upsampled multispectral bands with the panchromatic band into a sharpened multispectral image.

## Inputs

5 channels, float32, shape (N, 5, H, W):

1. `ms_blue`
2. `ms_green`
3. `ms_red`
4. `ms_nir`
5. `pan`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `blue` | - |
| 1 | `green` | - |
| 2 | `red` | - |
| 3 | `nir` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `pansharpening_tiny` | 733,268 | 256 | `ced1ba26767ec108` |
| `pansharpening_base` | 7,059,108 | 256 | `41a56a41d18801e4` |
| `pansharpening_large` | 22,060,084 | 512 | `eaa6049acd06d8ae` |
| `pansharpening_mega` | 60,451,908 | 512 | `fb1bcd5dd5792dca` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("pansharpening_base")  # verified starter weights
```

## Training

Required reference data: Sharpened references, for example by Wald's protocol.

```bash
unbihexium train pansharpening_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Multispectral and panchromatic bands of the same sensor

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
