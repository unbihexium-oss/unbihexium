# Security Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `security_monitor` |
| Task | detection |
| Domain | defense |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects vehicles, vessels and temporary structures around critical sites.

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `red`
2. `green`
3. `blue`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `vehicle` | - |
| 1 | `vessel` | - |
| 2 | `temporary_structure` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `security_monitor_tiny` | 730,647 | 256 | `459bc17bb672c6a5` |
| `security_monitor_base` | 7,049,255 | 256 | `21594348dcc33eb7` |
| `security_monitor_large` | 22,038,391 | 512 | `f122c21d13eeaa85` |
| `security_monitor_mega` | 60,413,767 | 512 | `600261af77a1cf52` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("security_monitor_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of the three object classes.

```bash
unbihexium train security_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Very high resolution satellite or aerial RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
