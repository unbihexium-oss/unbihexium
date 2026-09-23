# Landform Mapper

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `topography_mapper` |
| Task | segmentation |
| Domain | imaging |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Classifies landforms from a digital elevation model.

## Inputs

1 channels, float32, shape (N, 1, H, W):

1. `elevation`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `flat` | - |
| 1 | `slope` | - |
| 2 | `ridge` | - |
| 3 | `valley` | - |
| 4 | `peak` | - |
| 5 | `pit` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `topography_mapper_tiny` | 732,726 | 256 | `7e422fb7baf19a03` |
| `topography_mapper_base` | 7,058,022 | 256 | `90ebeaf10cf291cc` |
| `topography_mapper_large` | 22,058,454 | 512 | `b8add6e7c679dff5` |
| `topography_mapper_mega` | 60,449,734 | 512 | `4901c2acf1e64283` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("topography_mapper_base")  # verified starter weights
```

## Training

Required reference data: Landform masks, for example geomorphons.

```bash
unbihexium train topography_mapper_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Copernicus DEM GLO-30
- national DEMs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
