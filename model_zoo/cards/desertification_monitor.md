# Desertification Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `desertification_monitor` |
| Task | segmentation |
| Domain | environment |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Classifies land degradation severity in drylands.

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
| 0 | `not_degraded` | - |
| 1 | `low` | - |
| 2 | `moderate` | - |
| 3 | `severe` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `desertification_monitor_tiny` | 733,988 | 256 | `bc701283d74c7973` |
| `desertification_monitor_base` | 7,060,548 | 256 | `4955b155404bc6b0` |
| `desertification_monitor_large` | 22,062,244 | 512 | `71e4f0ce5dca0da8` |
| `desertification_monitor_mega` | 60,454,788 | 512 | `07c3a075b4875cb3` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("desertification_monitor_base")  # verified starter weights
```

## Training

Required reference data: Degradation severity masks from field surveys.

```bash
unbihexium train desertification_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Landsat 8/9

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
