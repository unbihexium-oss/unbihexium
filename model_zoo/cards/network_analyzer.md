# Road Network Density Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `network_analyzer` |
| Task | dense_regression |
| Domain | urban |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates road network density as road length per square kilometre.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `road_density` | km km-2 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `network_analyzer_tiny` | 733,073 | 256 | `816c023e8c6bb087` |
| `network_analyzer_base` | 7,058,721 | 256 | `b296239c0cea7267` |
| `network_analyzer_large` | 22,059,505 | 512 | `566196cdf2c4f8b3` |
| `network_analyzer_mega` | 60,451,137 | 512 | `969bcbac67a239b0` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("network_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Road density rasters derived from road vector data.

```bash
unbihexium train network_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
