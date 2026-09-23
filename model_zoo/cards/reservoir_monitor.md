# Reservoir Monitor

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `reservoir_monitor` |
| Task | segmentation |
| Domain | water |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments the water surface of reservoirs to track storage.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K, H, W) of class logits; apply softmax over K.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `background` | - |
| 1 | `water` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `reservoir_monitor_tiny` | 733,090 | 256 | `9ae5b63bed069ccb` |
| `reservoir_monitor_base` | 7,058,754 | 256 | `2541942cf603157d` |
| `reservoir_monitor_large` | 22,059,554 | 512 | `9bf9eb5f0098affb` |
| `reservoir_monitor_mega` | 60,451,202 | 512 | `7af5cd2e35628365` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("reservoir_monitor_base")  # verified starter weights
```

## Training

Required reference data: Water masks.

```bash
unbihexium train reservoir_monitor_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- Landsat 8/9

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
