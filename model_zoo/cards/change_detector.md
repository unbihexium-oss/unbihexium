# Change Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `change_detector` |
| Task | change_detection |
| Domain | ai |
| Architecture | unet_early_fusion |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Binary change detection between two dates.

## Inputs

6 channels, float32, shape (N, 6, H, W):

1. `red_t1`
2. `green_t1`
3. `blue_t1`
4. `red_t2`
5. `green_t2`
6. `blue_t2`

## Outputs

Tensor (N, K, H, W) of change class logits; softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `no_change` | - |
| 1 | `change` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `change_detector_tiny` | 733,378 | 256 | `ea42cf2f02884363` |
| `change_detector_base` | 7,059,330 | 256 | `4ade34ac830d494f` |
| `change_detector_large` | 22,060,418 | 512 | `d48eb6e31d478d0c` |
| `change_detector_mega` | 60,452,354 | 512 | `f2141f08b980fcce` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("change_detector_base")  # verified starter weights
```

## Training

Required reference data: Binary change masks.

```bash
unbihexium train change_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Co-registered RGB image pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
