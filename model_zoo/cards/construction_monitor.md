# Construction Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `construction_monitor` |
| Task | change_detection |
| Domain | urban |
| Architecture | unet_early_fusion |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects new construction and demolition between two dates.

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
| 1 | `new_construction` | - |
| 2 | `demolition` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `construction_monitor_tiny` | 733,395 | 256 | `2cbff6ade8444126` |
| `construction_monitor_base` | 7,059,363 | 256 | `2f09047f118a0d93` |
| `construction_monitor_large` | 22,060,467 | 512 | `e61ec305b12f9035` |
| `construction_monitor_mega` | 60,452,419 | 512 | `bedea571ec59aff6` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("construction_monitor_base")  # verified starter weights
```

## Training

Required reference data: Construction change masks.

```bash
unbihexium train construction_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Co-registered very high resolution RGB image pairs

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
