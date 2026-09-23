# Erosion Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `erosion_detector` |
| Task | segmentation |
| Domain | environment |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Maps eroded soil surfaces and gullies using spectral and terrain inputs.

## Inputs

5 channels, float32, shape (N, 5, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`
5. `elevation`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `stable` | - |
| 1 | `sheet_erosion` | - |
| 2 | `gully` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `erosion_detector_tiny` | 733,251 | 256 | `99be45a181e4b02d` |
| `erosion_detector_base` | 7,059,075 | 256 | `fa163570d60c4a32` |
| `erosion_detector_large` | 22,060,035 | 512 | `6f9db95b3ae8186c` |
| `erosion_detector_mega` | 60,451,843 | 512 | `062a9bf17258b759` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("erosion_detector_base")  # verified starter weights
```

## Training

Required reference data: Erosion masks.

```bash
unbihexium train erosion_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Multispectral imagery with a co-registered DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
