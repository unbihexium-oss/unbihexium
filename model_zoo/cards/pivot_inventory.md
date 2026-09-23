# Centre Pivot Inventory

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `pivot_inventory` |
| Task | detection |
| Domain | agriculture |
| Architecture | centernet |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Detects centre pivot irrigation fields for irrigation inventories.

## Inputs

4 channels, float32, shape (N, 4, H, W):

1. `blue`
2. `green`
3. `red`
4. `nir`

## Outputs

Tensor (N, K + 4, H/4, W/4): K class heatmap logits, box width and height in output-stride pixels, and the x and y centre offsets.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `centre_pivot` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `pivot_inventory_tiny` | 730,725 | 256 | `53d7ae9907fb0393` |
| `pivot_inventory_base` | 7,049,413 | 256 | `bc1a26ae96644349` |
| `pivot_inventory_large` | 22,038,629 | 512 | `46b590a011de8813` |
| `pivot_inventory_mega` | 60,414,085 | 512 | `4ae259625f2cdb23` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("pivot_inventory_base")  # verified starter weights
```

## Training

Required reference data: Bounding boxes of centre pivot fields.

```bash
unbihexium train pivot_inventory_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-2 10 m bands
- Landsat 8/9
- PlanetScope

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
