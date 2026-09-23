# Crop Boundary Delineation

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `crop_boundary_delineation` |
| Task | segmentation |
| Domain | agriculture |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Delineates agricultural field boundaries by segmenting field interiors and boundary lines.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `background` | - |
| 1 | `field_interior` | - |
| 2 | `field_boundary` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `crop_boundary_delineation_tiny` | 733,107 | 256 | `7b11fd0e62bb4521` |
| `crop_boundary_delineation_base` | 7,058,787 | 256 | `843a3c4f05d4201f` |
| `crop_boundary_delineation_large` | 22,059,603 | 512 | `9a0bd4a58c9eeaf8` |
| `crop_boundary_delineation_mega` | 60,451,267 | 512 | `7556a3341f655f9c` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("crop_boundary_delineation_base")  # verified starter weights
```

## Training

Required reference data: Field polygons rasterised into interiors and boundaries.

```bash
unbihexium train crop_boundary_delineation_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
