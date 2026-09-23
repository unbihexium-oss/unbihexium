# Mosaic Colour Harmoniser

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `mosaic_processor` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Harmonises the radiometry of a scene to a reference for seamless mosaics.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

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
| `mosaic_processor_tiny` | 732,963 | 256 | `2dcbf9b255b9d225` |
| `mosaic_processor_base` | 7,058,499 | 256 | `73d9224169e17fcb` |
| `mosaic_processor_large` | 22,059,171 | 512 | `1de31e0db9503c4c` |
| `mosaic_processor_mega` | 60,450,691 | 512 | `519a9a69b9cc2000` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("mosaic_processor_base")  # verified starter weights
```

## Training

Required reference data: Radiometrically harmonised reference scenes.

```bash
unbihexium train mosaic_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- RGB scenes of overlapping areas

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
