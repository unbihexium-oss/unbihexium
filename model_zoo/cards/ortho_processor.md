# Orthorectification Flow Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `ortho_processor` |
| Task | enhancement |
| Domain | imaging |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates the displacement field that removes relief displacement using a DEM.

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
| 0 | `dx` | px |
| 1 | `dy` | px |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `ortho_processor_tiny` | 733,090 | 256 | `03b2e9be32a2e972` |
| `ortho_processor_base` | 7,058,754 | 256 | `cfad3d55f0c245d6` |
| `ortho_processor_large` | 22,059,554 | 512 | `d1df292375fe6447` |
| `ortho_processor_mega` | 60,451,202 | 512 | `2801d62bf66d9571` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("ortho_processor_base")  # verified starter weights
```

## Training

Required reference data: Displacement fields from rigorous orthorectification.

```bash
unbihexium train ortho_processor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Raw imagery with a co-registered DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
