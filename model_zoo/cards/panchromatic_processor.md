# Panchromatic Denoiser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `panchromatic_processor` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Reduces noise in panchromatic imagery.

## Inputs

1 channels, float32, shape (N, 1, H, W):

1. `pan`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `pan` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `panchromatic_processor_tiny` | 732,641 | 256 | `0f89f1d05898c3bd` |
| `panchromatic_processor_base` | 7,057,857 | 256 | `5f6c270966844d6a` |
| `panchromatic_processor_large` | 22,058,209 | 512 | `961f3c7c0ada9ec3` |
| `panchromatic_processor_mega` | 60,449,409 | 512 | `efdad6103c61a4b6` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("panchromatic_processor_base")  # verified starter weights
```

## Training

Required reference data: Clean panchromatic references.

```bash
unbihexium train panchromatic_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Panchromatic satellite imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
