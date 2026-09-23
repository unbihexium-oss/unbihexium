# Protected Area Change Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `protected_area_change_detector` |
| Task | change_detection |
| Domain | environment |
| Architecture | unet_early_fusion |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects land cover change inside protected areas.

## Inputs

20 channels, float32, shape (N, 20, H, W):

1. `B02_t1`
2. `B03_t1`
3. `B04_t1`
4. `B05_t1`
5. `B06_t1`
6. `B07_t1`
7. `B08_t1`
8. `B8A_t1`
9. `B11_t1`
10. `B12_t1`
11. `B02_t2`
12. `B03_t2`
13. `B04_t2`
14. `B05_t2`
15. `B06_t2`
16. `B07_t2`
17. `B08_t2`
18. `B8A_t2`
19. `B11_t2`
20. `B12_t2`

## Outputs

Tensor (N, K, H, W) of change class logits; softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `no_change` | - |
| 1 | `vegetation_loss` | - |
| 2 | `new_structure` | - |
| 3 | `other_change` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `protected_area_change_detector_tiny` | 735,428 | 256 | `8568cafa531fcb74` |
| `protected_area_change_detector_base` | 7,063,428 | 256 | `bf3e30686589fd8e` |
| `protected_area_change_detector_large` | 22,066,564 | 512 | `ecb5a1034d94366b` |
| `protected_area_change_detector_mega` | 60,460,548 | 512 | `12a10e75a1b2ab62` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("protected_area_change_detector_base")  # verified starter weights
```

## Training

Required reference data: Change masks by change type.

```bash
unbihexium train protected_area_change_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A image pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
