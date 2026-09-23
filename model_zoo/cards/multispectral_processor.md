# Multispectral Denoiser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `multispectral_processor` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Reduces noise and striping in multispectral imagery.

## Inputs

10 channels, float32, shape (N, 10, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B05`
5. `B06`
6. `B07`
7. `B08`
8. `B8A`
9. `B11`
10. `B12`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `B02` | - |
| 1 | `B03` | - |
| 2 | `B04` | - |
| 3 | `B05` | - |
| 4 | `B06` | - |
| 5 | `B07` | - |
| 6 | `B08` | - |
| 7 | `B8A` | - |
| 8 | `B11` | - |
| 9 | `B12` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `multispectral_processor_tiny` | 734,090 | 256 | `3068d414cf406a2c` |
| `multispectral_processor_base` | 7,060,746 | 256 | `b416431bfcf936c9` |
| `multispectral_processor_large` | 22,062,538 | 512 | `144291057dcbcb07` |
| `multispectral_processor_mega` | 60,455,178 | 512 | `3cff22ff06869ff2` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("multispectral_processor_base")  # verified starter weights
```

## Training

Required reference data: Clean reference images, for example temporal composites.

```bash
unbihexium train multispectral_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
