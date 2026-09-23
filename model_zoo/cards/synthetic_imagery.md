# SAR to Optical Translator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `synthetic_imagery` |
| Task | enhancement |
| Domain | ai |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Synthesises an optical RGB image from SAR backscatter, for example to fill cloud gaps.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `VV`
2. `VH`

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
| `synthetic_imagery_tiny` | 732,819 | 256 | `806a26075803bace` |
| `synthetic_imagery_base` | 7,058,211 | 256 | `135a1442059b58b6` |
| `synthetic_imagery_large` | 22,058,739 | 512 | `36fa6a39b89c2423` |
| `synthetic_imagery_mega` | 60,450,115 | 512 | `6b9f2458c6e5e55d` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("synthetic_imagery_base")  # verified starter weights
```

## Training

Required reference data: Co-located cloud-free optical images.

```bash
unbihexium train synthetic_imagery_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD with co-located Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
