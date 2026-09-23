# Tourist Destination Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `tourist_destination_monitor` |
| Task | segmentation |
| Domain | tourism |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps beaches, built-up areas, vegetation and water around tourist destinations.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `background` | - |
| 1 | `beach` | - |
| 2 | `built_up` | - |
| 3 | `vegetation` | - |
| 4 | `water` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `tourist_destination_monitor_tiny` | 732,997 | 256 | `4398c70719469d39` |
| `tourist_destination_monitor_base` | 7,058,565 | 256 | `b5fbadd6aacd1d0f` |
| `tourist_destination_monitor_large` | 22,059,269 | 512 | `c33acbb1162e7b59` |
| `tourist_destination_monitor_mega` | 60,450,821 | 512 | `67bef88070797b6b` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("tourist_destination_monitor_base")  # verified starter weights
```

## Training

Required reference data: Masks of the land cover classes.

```bash
unbihexium train tourist_destination_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
