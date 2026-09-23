# Water Surface Detector

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `water_surface_detector` |
| Task | segmentation |
| Domain | water |
| Architecture | unet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Segments open water surfaces.

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
| `water_surface_detector_tiny` | 733,090 | 256 | `37ab16434f2af56a` |
| `water_surface_detector_base` | 7,058,754 | 256 | `701c53fadd7c3134` |
| `water_surface_detector_large` | 22,059,554 | 512 | `ef7c5a613a8f8bb2` |
| `water_surface_detector_mega` | 60,451,202 | 512 | `313a064e3539f706` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("water_surface_detector_base")  # verified starter weights
```

## Training

Required reference data: Water masks.

```bash
unbihexium train water_surface_detector_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- Landsat 8/9
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
