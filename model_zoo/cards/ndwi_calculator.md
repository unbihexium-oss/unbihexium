# NDWI Calculator

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Reference implementation of a published formula; no training needed.

## Overview

| Property | Value |
| --- | --- |
| Family | `ndwi_calculator` |
| Task | spectral_index |
| Domain | indices |
| Architecture | spectral_formula |
| Licence | MPL-2.0 |
| Trained on Earth observation data | Not applicable |

Normalized Difference Water Index, (GREEN - NIR) / (GREEN + NIR) (McFeeters, 1996).

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `green`
2. `nir`

## Outputs

Tensor (N, 1, H, W) with the index value; NaN where undefined.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `ndwi` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `ndwi_calculator_tiny` | 0 | 256 | `e3b0c44298fc1c14` |
| `ndwi_calculator_base` | 0 | 256 | `e3b0c44298fc1c14` |
| `ndwi_calculator_large` | 0 | 512 | `e3b0c44298fc1c14` |
| `ndwi_calculator_mega` | 0 | 512 | `e3b0c44298fc1c14` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("ndwi_calculator_base")  # verified starter weights
```

## Suitable data

- Surface reflectance from Sentinel-2
- Landsat or similar sensors

## Limitations and responsible use

The index is only meaningful for reflectance of the listed bands.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
