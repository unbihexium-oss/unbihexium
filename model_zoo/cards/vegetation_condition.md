# Vegetation Condition Index

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Reference implementation of a published formula; no training needed.

## Overview

| Property | Value |
| --- | --- |
| Family | `vegetation_condition` |
| Task | spectral_index |
| Domain | indices |
| Architecture | spectral_formula |
| Licence | MPL-2.0 |
| Trained on Earth observation data | Not applicable |

Vegetation Condition Index, (NDVI - NDVImin) / (NDVImax - NDVImin) (Kogan, 1995).

## Inputs

3 channels, float32, shape (N, 3, H, W):

1. `ndvi`
2. `ndvi_min`
3. `ndvi_max`

## Outputs

Tensor (N, 1, H, W) with the index value; NaN where undefined.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `vci` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `vegetation_condition_tiny` | 0 | 256 | `e3b0c44298fc1c14` |
| `vegetation_condition_base` | 0 | 256 | `e3b0c44298fc1c14` |
| `vegetation_condition_large` | 0 | 512 | `e3b0c44298fc1c14` |
| `vegetation_condition_mega` | 0 | 512 | `e3b0c44298fc1c14` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("vegetation_condition_base")  # verified starter weights
```

## Suitable data

- NDVI with its multi-year per-pixel minimum and maximum

## Limitations and responsible use

The index is only meaningful for reflectance of the listed bands.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
