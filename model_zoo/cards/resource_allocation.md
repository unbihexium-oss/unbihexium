# Service Demand Estimator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `resource_allocation` |
| Task | scene_regression |
| Domain | analysis |
| Architecture | encoder_regressor |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates population and service demand of an area for resource allocation.

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
| 0 | `population` | persons |
| 1 | `service_demand` | 1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `resource_allocation_tiny` | 494,658 | 256 | `274e26bb8a1c519b` |
| `resource_allocation_base` | 3,745,410 | 256 | `1cf7b3d45b5ee5dd` |
| `resource_allocation_large` | 14,606,594 | 512 | `b70dd7f03ed659c5` |
| `resource_allocation_mega` | 37,763,074 | 512 | `34024b9c08a8c3fc` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("resource_allocation_base")  # verified starter weights
```

## Training

Required reference data: Census population and service statistics per chip.

```bash
unbihexium train resource_allocation_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
