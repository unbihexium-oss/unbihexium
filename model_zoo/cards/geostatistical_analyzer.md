# Geostatistical Surface Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `geostatistical_analyzer` |
| Task | dense_regression |
| Domain | analysis |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Learns a continuous surface of a sampled variable from covariates, as a learned alternative to kriging.

## Inputs

12 channels, float32, shape (N, 12, H, W):

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
11. `elevation`
12. `slope`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `value` | user defined |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `geostatistical_analyzer_tiny` | 734,225 | 256 | `e7a0ec11cd9caafc` |
| `geostatistical_analyzer_base` | 7,061,025 | 256 | `36f274f5a36520ad` |
| `geostatistical_analyzer_large` | 22,062,961 | 512 | `69a037c12b438bea` |
| `geostatistical_analyzer_mega` | 60,455,745 | 512 | `709c9d23559cdc92` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("geostatistical_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Point samples rasterised as sparse targets.

```bash
unbihexium train geostatistical_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
