# Bathymetry Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `offshore_survey` |
| Task | dense_regression |
| Domain | water |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates shallow water depth (satellite-derived bathymetry).

## Inputs

10 channels, float32, shape (N, 10, H, W):

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

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `depth` | m |

Outputs are bounded to [0.0, 30.0].

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `offshore_survey_tiny` | 733,937 | 256 | `56cd1baba6f97771` |
| `offshore_survey_base` | 7,060,449 | 256 | `815ed22de4baf1ff` |
| `offshore_survey_large` | 22,062,097 | 512 | `02c4a7a08d7840a7` |
| `offshore_survey_mega` | 60,454,593 | 512 | `07093f18fb75b082` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("offshore_survey_base")  # verified starter weights
```

## Training

Required reference data: Echo sounding or LiDAR bathymetry.

```bash
unbihexium train offshore_survey_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
