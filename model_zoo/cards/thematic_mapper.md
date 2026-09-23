# Thematic Mapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `thematic_mapper` |
| Task | segmentation |
| Domain | imaging |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Eight-class thematic mapping for general land cover mapping projects.

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

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `artificial` | - |
| 1 | `agricultural` | - |
| 2 | `forest` | - |
| 3 | `grassland` | - |
| 4 | `wetland` | - |
| 5 | `water` | - |
| 6 | `bare` | - |
| 7 | `snow_ice` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `thematic_mapper_tiny` | 734,056 | 256 | `f9196c950cc627da` |
| `thematic_mapper_base` | 7,060,680 | 256 | `ebf52dc3a51492c3` |
| `thematic_mapper_large` | 22,062,440 | 512 | `713b4d1d277799d3` |
| `thematic_mapper_mega` | 60,455,048 | 512 | `638d4ffd4b0f4dc7` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("thematic_mapper_base")  # verified starter weights
```

## Training

Required reference data: Thematic masks.

```bash
unbihexium train thematic_mapper_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
