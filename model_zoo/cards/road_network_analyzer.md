# Road Network Extractor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `road_network_analyzer` |
| Task | segmentation |
| Domain | urban |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Extracts road surfaces for road network mapping.

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
| 1 | `road` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `road_network_analyzer_tiny` | 732,946 | 256 | `e0dea8276f123cbd` |
| `road_network_analyzer_base` | 7,058,466 | 256 | `47105799b542f89c` |
| `road_network_analyzer_large` | 22,059,122 | 512 | `cf1411cd301bd5e4` |
| `road_network_analyzer_mega` | 60,450,626 | 512 | `5e520b0cf771820f` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("road_network_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Road masks.

```bash
unbihexium train road_network_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Aerial or satellite RGB imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
