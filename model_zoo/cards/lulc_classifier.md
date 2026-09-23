# Land Use and Land Cover Classifier

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `lulc_classifier` |
| Task | segmentation |
| Domain | environment |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps land cover in the eleven classes of ESA WorldCover.

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
| 0 | `tree_cover` | - |
| 1 | `shrubland` | - |
| 2 | `grassland` | - |
| 3 | `cropland` | - |
| 4 | `built_up` | - |
| 5 | `bare_sparse` | - |
| 6 | `snow_ice` | - |
| 7 | `water` | - |
| 8 | `herbaceous_wetland` | - |
| 9 | `mangroves` | - |
| 10 | `moss_lichen` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `lulc_classifier_tiny` | 734,107 | 256 | `9916f3d82d5fd01a` |
| `lulc_classifier_base` | 7,060,779 | 256 | `68ecbf0013071527` |
| `lulc_classifier_large` | 22,062,587 | 512 | `9b0c44f549e12c09` |
| `lulc_classifier_mega` | 60,455,243 | 512 | `1089868497fb863e` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("lulc_classifier_base")  # verified starter weights
```

## Training

Required reference data: Land cover masks, for example ESA WorldCover.

```bash
unbihexium train lulc_classifier_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
