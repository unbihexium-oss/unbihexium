# Emergency Needs Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `emergency_disaster_manager` |
| Task | scene_regression |
| Domain | risk |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates affected population and shelter needs after a disaster.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K) with one value per target and chip.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `affected_population` | persons |
| 1 | `shelter_need` | persons |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `emergency_disaster_manager_tiny` | 494,658 | 256 | `5d2be318a827ce4e` |
| `emergency_disaster_manager_base` | 3,745,410 | 256 | `59fc3c20ce3756fb` |
| `emergency_disaster_manager_large` | 14,606,594 | 512 | `01790988c32f37e4` |
| `emergency_disaster_manager_mega` | 37,763,074 | 512 | `1d9c9d34ed85ea1b` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("emergency_disaster_manager_base")  # verified starter weights
```

## Training

Required reference data: Affected population and shelter statistics per chip.

```bash
unbihexium train emergency_disaster_manager_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- very high resolution imagery

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
