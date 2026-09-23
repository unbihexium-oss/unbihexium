# Leakage Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `leakage_detector` |
| Task | detection |
| Domain | assets |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects vegetation stress and surface anomalies that indicate pipeline or canal leakage.

## Inputs

5 channels, float32, shape (N, 5, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`
5. `swir1`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `leak_signature` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `leakage_detector_tiny` | 730,869 | 256 | `25bec0a1be5a702c` |
| `leakage_detector_base` | 7,049,701 | 256 | `4fa44cc09a8248f1` |
| `leakage_detector_large` | 22,039,061 | 512 | `681e0badf37ab9e1` |
| `leakage_detector_mega` | 60,414,661 | 512 | `a32785ced2048631` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("leakage_detector_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of confirmed leak locations.

```bash
unbihexium train leakage_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Multispectral satellite or aerial imagery with a short-wave infrared band

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
