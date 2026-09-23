# Crop Type Classifier

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `crop_classifier` |
| Task | segmentation |
| Domain | agriculture |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps crop types per pixel.

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
| 0 | `background` | - |
| 1 | `wheat` | - |
| 2 | `maize` | - |
| 3 | `rice` | - |
| 4 | `soybean` | - |
| 5 | `sunflower` | - |
| 6 | `other_crop` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `crop_classifier_tiny` | 734,039 | 256 | `be881d63284ec626` |
| `crop_classifier_base` | 7,060,647 | 256 | `ce5e923880427215` |
| `crop_classifier_large` | 22,062,391 | 512 | `4c90d94df816ee5a` |
| `crop_classifier_mega` | 60,454,983 | 512 | `51bd920a13e07b68` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("crop_classifier_base")  # verified starter weights
```

## Training

Required reference data: Crop type masks, for example from farmer declarations.

```bash
unbihexium train crop_classifier_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
