# SAR Despeckler

<!-- Generated from src/unbihexium/zoo/catalog.yaml by `python -m unbihexium.zoo.sync`. -->

> Starter model: complete architecture with deterministic starter weights, **not trained**. Train or fine-tune it on labelled data before use.

## Overview

| Property | Value |
| --- | --- |
| Family | `sar_mapping_workflow` |
| Task | enhancement |
| Domain | sar |
| Architecture | unet_image_to_image |
| Licence | MPL-2.0 |
| Trained on Earth observation data | No |

Removes speckle from dual-polarisation SAR backscatter.

## Inputs

2 channels, float32, shape (N, 2, H, W):

1. `VV`
2. `VH`

## Outputs

Tensor (N, K, H, W) of output bands or displacement components.

| Index | Name | Unit |
| --- | --- | --- |
| 0 | `VV` | - |
| 1 | `VH` | - |

## Variants

| Model id | Parameters | Tile size | Weights digest (SHA-256, first 16) |
| --- | --- | --- | --- |
| `sar_mapping_workflow_tiny` | 732,802 | 256 | `437bc2908c429846` |
| `sar_mapping_workflow_base` | 7,058,178 | 256 | `8790ce677c9b721c` |
| `sar_mapping_workflow_large` | 22,058,690 | 512 | `e206e2a5492aff37` |
| `sar_mapping_workflow_mega` | 60,450,050 | 512 | `26392f26d6c7e309` |

## Usage

```python
from unbihexium.zoo import load_model

model = load_model("sar_mapping_workflow_base")  # verified starter weights
```

## Training

Required reference data: Multi-temporal averages as speckle-free references.

```bash
unbihexium train sar_mapping_workflow_base --data path/to/dataset --epochs 50
```

See docs/model_zoo/training.md for the dataset layout.

## Suitable data

- Sentinel-1 GRD

## Limitations and responsible use

The starter weights produce meaningless predictions until the model is trained. After training, validate the model on independent reference data for your area, sensor and season, and report its accuracy with the results.

Follow RESPONSIBLE_USE.md, in particular for uses that affect people or property.
