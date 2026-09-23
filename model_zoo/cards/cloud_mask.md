# Cloud and Shadow Mask

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `cloud_mask` |
| Task | segmentation |
| Domain | imaging |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Masks clear sky, thick cloud, thin cloud and cloud shadow in Sentinel-2 imagery.

## Inputs

13 channels, float32, shape (N, 13, H, W):

1. `B01`
2. `B02`
3. `B03`
4. `B04`
5. `B05`
6. `B06`
7. `B07`
8. `B08`
9. `B8A`
10. `B09`
11. `B10`
12. `B11`
13. `B12`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `clear` | - |
| 1 | `thick_cloud` | - |
| 2 | `thin_cloud` | - |
| 3 | `cloud_shadow` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `cloud_mask_tiny` | 734,420 | 256 | `85edd633f829bf55` |
| `cloud_mask_base` | 7,061,412 | 256 | `b773e6718fc5f2c4` |
| `cloud_mask_large` | 22,063,540 | 512 | `a2af64f7e0d48e6f` |
| `cloud_mask_mega` | 60,456,516 | 512 | `34a798e7f197f30a` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("cloud_mask_base")  # verified starter weights
```

## Training

Required reference data: Cloud and shadow masks.

```bash
unbihexium train cloud_mask_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L1C

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
