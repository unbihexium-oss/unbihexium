# Phenology Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `timeseries_analyzer` |
| Task | scene_regression |
| Domain | agriculture |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates start, peak and end of season from a stack of six NDVI observations.

## Inputs

6 channels, float32, shape (N, 6, H, W):

1. `ndvi_t1`
2. `ndvi_t2`
3. `ndvi_t3`
4. `ndvi_t4`
5. `ndvi_t5`
6. `ndvi_t6`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `start_of_season` | day of year |
| 1 | `peak_of_season` | day of year |
| 2 | `end_of_season` | day of year |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `timeseries_analyzer_tiny` | 494,979 | 256 | `bdf4c862bab85231` |
| `timeseries_analyzer_base` | 3,746,051 | 256 | `5b2ab8fad4fa526c` |
| `timeseries_analyzer_large` | 14,607,555 | 512 | `307f224e34847200` |
| `timeseries_analyzer_mega` | 37,764,355 | 512 | `1f905f9e06be4eaf` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("timeseries_analyzer_base")  # verified starter weights
```

## Training

Required reference data: Phenology dates from field observation or reference products.

```bash
unbihexium train timeseries_analyzer_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A time series
- Landsat 8/9 time series

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
