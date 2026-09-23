# Route Cost Surface

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `route_planner` |
| Task | dense_regression |
| Domain | tourism |
| Architecture | unet_regression |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Estimates a travel cost surface for least-cost routing.

## Inputs

10 channels, float32, shape (N, 10, H, W):

1. `B02`
2. `B03`
3. `B04`
4. `B08`
5. `B11`
6. `B12`
7. `elevation`
8. `slope`
9. `population`
10. `road_distance`

## Outputs

Tensor (N, K, H, W) of target values in the listed units.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `travel_cost` | s m-1 |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `route_planner_tiny` | 733,937 | 256 | `4965bcbf980b5c74` |
| `route_planner_base` | 7,060,449 | 256 | `ea38aec6437c110f` |
| `route_planner_large` | 22,062,097 | 512 | `27e2cb7d2f03a34e` |
| `route_planner_mega` | 60,454,593 | 512 | `8d3388c4ea666e5d` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("route_planner_base")  # verified starter weights
```

## Training

Required reference data: Travel cost rasters from speed models.

```bash
unbihexium train route_planner_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 L2A
- Copernicus DEM
- road distance rasters

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
