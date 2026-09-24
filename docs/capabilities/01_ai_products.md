<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/01_ai_products.md
Title       : Capability Domain 01: AI Products
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 01: AI Products

| Field | Value |
| --- | --- |
| Document | UBX-DOC-601 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium (declared version 1.0.1, model catalogue 2.0.0) |

## Abstract

This document describes capability domain 01, "AI products": the general-purpose learned models of the Unbihexium model zoo (object detection, semantic segmentation, change detection and SAR-to-optical translation) and the library capabilities that build, run, train, evaluate and serve every model of the zoo. It is written for users who want to know which model families exist in this domain and how to run them, for contributors who extend the task interfaces, and for reviewers who need to check what the code actually provides. It maps the domain to the registry (`unbihexium.registry`, domain value `ai`) and to the `domain` field of `src/unbihexium/zoo/catalog.yaml`, lists the seven model families in a table generated from the catalogue, explains the network output layouts and the decoding formulas that the code implements, and gives executed examples. All seven families are untrained starter models; the domain describes intended applications, not validated products.

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Place in the registry and the catalogue](#2-place-in-the-registry-and-the-catalogue)
3. [Model families](#3-model-families)
4. [Library functions](#4-library-functions)
5. [Examples](#5-examples)
6. [Limitations and responsible use](#6-limitations-and-responsible-use)
7. [Related documents](#7-related-documents)
8. [References](#references)

## 1. Scope and status

### 1.1 What the domain covers

Domain 01 groups two kinds of capabilities:

- the model families whose catalogue entry has `domain: ai`: generic detectors of aircraft, ships, vehicles and common objects in overhead RGB imagery, a six-class semantic segmentation model, a two-date change detector and a network that synthesises an RGB image from Sentinel-1 backscatter;
- the library capabilities that the registry files under the same domain: the model zoo itself (`model_zoo`), training and evaluation (`model_training`) and the REST service (`model_serving`). These serve all 130 families of the zoo, not only the seven of this domain.

Thematic models that also detect or segment objects (buildings, water, crops, oil spills and so on) belong to the domains of their application and are described in the other capability documents; see the [capability index](index.md).

### 1.2 Status of the models

Every model of this domain is an **untrained starter model**: a complete, trainable network with the input and output layout of its task and deterministic initial weights whose SHA-256 digest is published in `model_zoo/manifests/`. The weights have not been fitted to Earth observation data, so predictions are not meaningful until the model has been trained or fine-tuned on labelled data for the sensor and area of interest (see [docs/model_zoo/training.md](../model_zoo/training.md)). The only exception in the whole zoo are the seven spectral index families (28 models) of [domain 03](03_indices_flood_water.md), which compute exact formulas; none of them belongs to this domain. No accuracy figures are published for any model, because there are no trained weights to measure. Section 5.2 shows what the output of an untrained model looks like.

### 1.3 Changes from the previous version

Version 1 of this document listed thirteen "production" models, accuracy and throughput tables, an SRCNN super-resolution network, a Siamese change detection network and commands such as `unbihexium infer`. None of these exists in the current code: the models are untrained, no benchmarks were measured, super-resolution and pansharpening families belong to the imaging domain ([domain 04](04_environment_forestry_image_processing.md)), change detection uses an early-fusion U-Net (Section 3.3), and models are run with `unbihexium predict`. The content was replaced by a description of the code.

## 2. Place in the registry and the catalogue

The registry enumeration `unbihexium.registry.CapabilityDomain` has the member `AI = "ai"`. The capability registry loads one model capability per catalogue family and a fixed list of library capabilities (`src/unbihexium/registry/capabilities.py`). For the value `ai` it holds ten capabilities:

| Capability id | Kind | Maturity | Entry points |
| --- | --- | --- | --- |
| `aircraft_detector`, `object_detector`, `ship_detector`, `vehicle_detector`, `multi_solution_segmentation`, `change_detector`, `synthetic_imagery` | model family | beta (starter model, `requires_training` tag `true`) | `unbihexium.ai.predict.predict` |
| `model_zoo` | library | stable | `unbihexium.zoo` |
| `model_training` | library | stable | `unbihexium.ai.training`, `unbihexium.ai.evaluation` |
| `model_serving` | library | stable | `unbihexium.serving` |

The maturity follows a fixed rule in `catalogue_capabilities()`: spectral index families are `stable`, every other family is `beta` until it is trained. Two of the seven families have a registered pipeline (`PIPELINES_BY_FAMILY`): `ship_detector` runs in the pipeline `ship_detection` and `change_detector` in `change_detection`. The pipelines `building_detection`, `water_detection` and `super_resolution` also list `ai` among their domains, but their model families belong to the urban, water and imaging domains.

Every capability of a model family carries the command line `unbihexium predict <family>_base INPUT OUTPUT`, the family's task, its input bands and the four model ids of its size variants (`tiny`, `base`, `large`, `mega`).

## 3. Model families

### 3.1 Inventory

The two tables below were generated from the catalogue and the model registry of the installed package with the script of [index.md, Section 4](index.md#4-regenerating-the-family-tables), run with the argument `ai`. The parameter counts are those of the deterministic starter networks (`unbihexium.zoo.get_model(model_id).num_parameters`).

| Family | Domain | Task | Input bands | Outputs | Parameters (tiny / base / large / mega) |
| --- | --- | --- | --- | --- | --- |
| `aircraft_detector` | ai | detection | red, green, blue | aircraft | 730,581 / 7,049,125 / 22,038,197 / 60,413,509 |
| `object_detector` | ai | detection | red, green, blue | building, vehicle, ship, aircraft, storage_tank, bridge | 730,746 / 7,049,450 / 22,038,682 / 60,414,154 |
| `ship_detector` | ai | detection | red, green, blue | ship | 730,581 / 7,049,125 / 22,038,197 / 60,413,509 |
| `vehicle_detector` | ai | detection | red, green, blue | car, truck, bus | 730,647 / 7,049,255 / 22,038,391 / 60,413,767 |
| `multi_solution_segmentation` | ai | segmentation | red, green, blue | background, building, road, water, vegetation, bare_ground | 733,014 / 7,058,598 / 22,059,318 / 60,450,886 |
| `change_detector` | ai | change_detection | red, green, blue (x 2 dates) | no_change, change | 733,378 / 7,059,330 / 22,060,418 / 60,452,354 |
| `synthetic_imagery` | ai | enhancement | VV, VH | red, green, blue | 732,819 / 7,058,211 / 22,058,739 / 60,450,115 |

| Family | Name | Intended application | Reference data needed for training |
| --- | --- | --- | --- |
| `aircraft_detector` | Aircraft Detector | Detects aircraft on aprons, runways and in flight in very high resolution optical imagery. | Bounding boxes of aircraft. |
| `object_detector` | Generic Object Detector | Multi-class detector for common objects in overhead imagery. | Bounding boxes of the six object classes. |
| `ship_detector` | Ship Detector | Detects ships in optical imagery. | Bounding boxes of ships. |
| `vehicle_detector` | Vehicle Detector | Detects cars, trucks and buses. | Bounding boxes of vehicles by type. |
| `multi_solution_segmentation` | General Semantic Segmentation | General-purpose semantic segmentation with six common classes. | Masks of the six classes. |
| `change_detector` | Change Detector | Binary change detection between two dates. | Binary change masks. |
| `synthetic_imagery` | SAR to Optical Translator | Synthesises an optical RGB image from SAR backscatter, for example to fill cloud gaps. | Co-located cloud-free optical images. |

The column "Intended application" is the catalogue description of what the model does **once it has been trained**. The catalogue also records suitable input data for each family (field `sources`), which `unbihexium zoo info <model id>` prints.

### 3.2 Size variants

Each family exists in four variants that differ in width, depth and tile size (`unbihexium.zoo.get_variant`): `tiny` (16 base channels, depth 3, tile 256 pixels), `base` (32, 4, 256), `large` (48, 4, 512) and `mega` (64, 5, 512). The `base` variant is the default wherever a family name is given without a variant.

### 3.3 Architectures and output layouts

The networks are defined in `src/unbihexium/ai/models/`. All take a float32 tensor $(N, C, H, W)$ with the input bands on the channel axis; the encoder is a residual convolutional network [1] with group normalisation [2] and the decoder a U-Net decoder with skip connections [3].

| Task (families) | Architecture id | Output tensor |
| --- | --- | --- |
| detection (`aircraft_detector`, `object_detector`, `ship_detector`, `vehicle_detector`) | `centernet` | $(N, K + 4, H/4, W/4)$: $K$ class heat map logits, box width and height in output-stride pixels, and the $x$ and $y$ centre offsets [4] |
| segmentation (`multi_solution_segmentation`) | `unet` | $(N, K, H, W)$ class logits; softmax over $K$ |
| change detection (`change_detector`) | `unet_early_fusion` | $(N, K, H, W)$ change class logits; the two dates are stacked on the channel axis (6 input channels: `red_t1`, `green_t1`, `blue_t1`, `red_t2`, `green_t2`, `blue_t2`) |
| enhancement (`synthetic_imagery`) | `unet_image_to_image` | $(N, K, H, W)$ output bands (here red, green and blue) |

The initial weights are drawn from a NumPy `RandomState` stream seeded with the first four bytes of SHA-256 of `"unbihexium:" + model id`, with He normal initialisation [5] of convolution and linear layers (`src/unbihexium/ai/models/init.py`), so the same model id yields bit-identical weights on every platform.

## 4. Library functions

### 4.1 Task interfaces

`unbihexium.ai` exposes one class per task. Each accepts a family name, a model id, a checkpoint, an ONNX file or a loaded model, plus the options `variant`, `weights`, `device`, `backend` (`auto`, `torch` or `onnx`), `tile_size`, `overlap` (default 0.25) and `batch_size`. Inputs are a `Raster`, a file path or a NumPy array of shape $(C, H, W)$; arrays without georeferencing are given the identity transform and `EPSG:4326`.

| Class | Default family | Method | Result |
| --- | --- | --- | --- |
| `ObjectDetector` | `object_detector` | `predict(image)` | `DetectionResult`: list of `Detection` (pixel box, map box, confidence, class id and name); options `threshold` (0.5), `iou_threshold` (0.5), `max_detections` (1000) |
| `ShipDetector`, `AircraftDetector`, `VehicleDetector` | `ship_detector`, `aircraft_detector`, `vehicle_detector` | `predict(image)` | `DetectionResult` |
| `SemanticSegmenter` | `lulc_classifier` (pass `multi_solution_segmentation` explicitly) | `predict(image)` | `SegmentationResult`: `mask` (uint8 class map, 255 for rejected pixels), `classes`, `class_fractions()`, `class_areas()`, optional `probabilities` |
| `ChangeDetector` | `change_detector` | `predict_pair(before, after)` | `SegmentationResult` of the classes `no_change` and `change` |
| `Enhancer` | `pansharpening` (pass `synthetic_imagery` explicitly) | `predict(image)` | `EnhancementResult` with an output raster |

The generic functions `unbihexium.ai.predict(model, image, **options)` and `unbihexium.ai.write_result(result, path)` pick the class that matches the task of any model and write detections as a GeoJSON FeatureCollection, class maps as single-band GeoTIFF and image outputs as multi-band float32 GeoTIFF. The command `unbihexium predict` is a thin wrapper around these two functions.

For two-class models, `SemanticSegmenter` labels a pixel as class 1 when $p_1 \ge t$ (threshold $t$, default 0.5); for more classes it takes the arg max and rejects pixels whose largest probability is below $t$ (label 255).

### 4.2 Decoding of detector outputs

`unbihexium.ai.decode` turns the raw detector tensor into boxes (CenterNet decoding [4]). With $K$ classes, stride $s = 4$ and output $O$ of shape $(K + 4, h, w)$:

1. heat map scores $\hat{Y}_{k,i,j} = \sigma(O_{k,i,j})$ with the logistic function $\sigma$;
2. peaks are cells equal to the maximum of their $3 \times 3$ neighbourhood with $\hat{Y} \ge$ `threshold` (default 0.3 in `decode_centernet`);
3. a peak at row $i$, column $j$ becomes the box with centre and size

$$
c_x = (j + O_{K+2,i,j})\,s, \quad c_y = (i + O_{K+3,i,j})\,s, \quad w = \max(O_{K,i,j}, 0)\,s, \quad h = \max(O_{K+1,i,j}, 0)\,s .
$$

Overlapping boxes are then removed per class by greedy non-maximum suppression (`nms`) with the intersection over union

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} ,
$$

keeping the box with the higher score when the IoU exceeds `iou_threshold`.

### 4.3 Training and evaluation (`model_training`)

`unbihexium.ai.training` trains any family of the zoo on a dataset folder or on synthetic samples: AdamW with linear warm-up and cosine decay, gradient clipping, optional mixed precision, validation after every epoch, best and last checkpoints and early stopping. The per-band normalisation statistics of the training data are stored with the checkpoint so that inference and ONNX exports apply the same scaling. The command line equivalents are `unbihexium train` and `unbihexium evaluate`.

`unbihexium.ai.evaluation` computes the measures reported by training and evaluation: average precision per class and its mean at IoU 0.5 and averaged over IoU 0.5 to 0.95 [6], [7] for detectors; overall accuracy, per-class and mean IoU, F1, precision, recall and Cohen's kappa [8] for segmentation and change detection; MAE, RMSE, bias and $R^2$ for regression; PSNR and SSIM [9] for image outputs. The training procedure and the dataset layout are described in [docs/model_zoo/training.md](../model_zoo/training.md).

### 4.4 Model zoo and serving (`model_zoo`, `model_serving`)

`unbihexium.zoo` holds the catalogue, builds the deterministic starter weights into the local store (`UNBIHEXIUM_CACHE`), verifies them against the published digests and exports models to ONNX (`unbihexium zoo build|verify|export|info|list|where|clear`). `unbihexium.serving` is a FastAPI application with the routes `/health`, `/capabilities`, `/capabilities/{capability_id}`, `/models`, `/models/{model_id}`, `/pipelines`, `/predict/{model_id}`, `/infer/{model_id}`, `/detect/{model_id}` and `/segment/{model_id}`. See [docs/model_zoo/inference.md](../model_zoo/inference.md), [docs/model_zoo/download_and_verify.md](../model_zoo/download_and_verify.md) and [docs/operations/docker.md](../operations/docker.md).

## 5. Examples

The examples were executed on 24 September 2026 against the main branch with CPython 3.13, PyTorch (CPU) and `UNBIHEXIUM_CACHE` set to a temporary directory. They use the `tiny` variants and small synthetic arrays, so they run in a few seconds on a CPU.

### 5.1 Querying the registry

```python
from unbihexium.registry import CapabilityRegistry

for cap in CapabilityRegistry.by_domain("ai"):
    print(f"{cap.capability_id:28} {cap.maturity.value:7} {cap.task or '-'}")
```

```text
aircraft_detector            beta    detection
change_detector              beta    change_detection
model_serving                stable  -
model_training               stable  -
model_zoo                    stable  -
multi_solution_segmentation  beta    segmentation
object_detector              beta    detection
ship_detector                beta    detection
synthetic_imagery            beta    enhancement
vehicle_detector             beta    detection
```

### 5.2 Running a tiny starter model

The change detector compares two acquisitions on the same grid. Here only a 20 by 20 pixel square (about 9.8 % of the image) differs between the dates, yet the untrained model flags most of the image as change. This is the expected behaviour of a starter model and the reason why every model outside the spectral index families must be trained before use.

```python
import numpy as np
from unbihexium.ai import ChangeDetector

rng = np.random.default_rng(0)
before = rng.random((3, 64, 64), dtype=np.float32)
after = before.copy()
after[:, 20:40, 20:40] = 1.0

result = ChangeDetector("change_detector_tiny").predict_pair(before, after)
print(result.mask.shape, result.mask.dtype, result.classes)
print({k: round(v, 3) for k, v in result.class_fractions().items()})
```

```text
(64, 64) uint8 ['no_change', 'change']
{'no_change': 0.271, 'change': 0.729}
```

The model registry validates inputs before a model is run, and names the expected bands:

```python
from unbihexium.registry import ModelRegistry

try:
    ModelRegistry.check_input("change_detector_tiny", (3, 64, 64))
except ValueError as error:
    print(error)
```

```text
change_detector_tiny expects 6 bands (red_t1, green_t1, blue_t1, red_t2, green_t2, blue_t2), got 3
```

### 5.3 Decoding a detector output and suppressing duplicates

A synthetic CenterNet output with one class and a single peak at row 8, column 8 decodes to one box of 16 by 8 pixels centred at (34, 34); non-maximum suppression then removes the second of two boxes whose IoU is 0.681.

```python
import numpy as np
from unbihexium.ai.decode import box_iou, decode_centernet, nms

out = np.zeros((5, 16, 16), dtype=np.float32)
out[0] = -10.0                         # heat map logits of the only class
out[0, 8, 8] = 5.0                     # one peak at row 8, column 8
out[1, 8, 8], out[2, 8, 8] = 4.0, 2.0  # width and height in stride units
out[3, 8, 8], out[4, 8, 8] = 0.5, 0.5  # sub-pixel centre offset
boxes, scores, classes = decode_centernet(out, threshold=0.3)
print(boxes, scores.round(4), classes)

b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
print(box_iou(b[:1], b).round(3))
print(nms(b, np.array([0.9, 0.8, 0.7]), iou_threshold=0.5))
```

```text
[[26. 30. 42. 38.]] [0.9933] [0]
[[1.    0.681 0.   ]]
[0 2]
```

### 5.4 Command line

The following commands write a small georeferenced test image, list the tiny models of the domain and run the ship detector. The untrained detector finds no ship, so the FeatureCollection is empty.

```python
import numpy as np
from unbihexium.core.raster import Raster

rng = np.random.default_rng(0)
Raster.from_array(
    rng.random((3, 64, 64)).astype("float32"),
    crs="EPSG:32633",
    transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 6700000.0),
).to_file("rgb.tif")
```

```bash
unbihexium zoo list --domain ai --variant tiny
unbihexium predict ship_detector_tiny rgb.tif ships.geojson
cat ships.geojson
```

```text
                             Model zoo (7 models)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model ID                         ┃ Task             ┃ Domain ┃ Parameters ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ aircraft_detector_tiny           │ detection        │ ai     │    730,581 │
│ object_detector_tiny             │ detection        │ ai     │    730,746 │
│ ship_detector_tiny               │ detection        │ ai     │    730,581 │
│ vehicle_detector_tiny            │ detection        │ ai     │    730,647 │
│ multi_solution_segmentation_tiny │ segmentation     │ ai     │    733,014 │
│ change_detector_tiny             │ change_detection │ ai     │    733,378 │
│ synthetic_imagery_tiny           │ enhancement      │ ai     │    732,819 │
└──────────────────────────────────┴──────────────────┴────────┴────────────┘
Wrote: ships.geojson (ship_detector_tiny)
{
  "type": "FeatureCollection",
  "model_id": "ship_detector_tiny",
  "crs": "EPSG:32633",
  "features": []
}
```

The same model runs through the registered pipeline with `unbihexium pipeline run ship_detection -i rgb.tif -o ships.geojson -p variant=tiny`.

## 6. Limitations and responsible use

### 6.1 Conventions

The key words MUST, SHOULD and MAY in this section are to be interpreted as described in RFC 2119 and RFC 8174 [11], [12] when, and only when, they appear in capitals.

### 6.2 Limitations

- The models of this domain are untrained. Their outputs MUST NOT be used for decisions before the model has been trained and validated on independent reference data for the intended sensor, resolution and area. The measures of Section 4.3 and of `unbihexium.metrics` SHOULD be used for that validation.
- The family descriptions state intended applications. They are not claims that a trained model reaches any particular accuracy.
- Detections of aircraft, ships and vehicles can concern people and property. Users MUST read [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) before deploying a trained model, in particular for monitoring or security uses.
- Detection results keep the coordinate reference system of the input raster, which is recorded in the `crs` member of the GeoJSON output. Users who need a strictly RFC 7946 conformant file [10] SHOULD reproject the boxes to WGS 84.

## 7. Related documents

- [Capability index](index.md) and the other capability domain documents.
- [Model catalogue](../model_zoo/model_catalog.md), [training](../model_zoo/training.md), [inference](../model_zoo/inference.md).
- [Command line reference](../reference/cli.md) and [API reference](../reference/api.md).
- [Capability registry architecture](../architecture/capability_registry.md).
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) and [README.md](../../README.md).

## References

[1] He, K., Zhang, X., Ren, S., Sun, J. Deep residual learning for image recognition. CVPR 2016. 2016. <https://doi.org/10.1109/CVPR.2016.90>

[2] Wu, Y., He, K. Group normalization. ECCV 2018. 2018. <https://doi.org/10.1007/978-3-030-01261-8_1>

[3] Ronneberger, O., Fischer, P., Brox, T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015. 2015. <https://doi.org/10.1007/978-3-319-24574-4_28>

[4] Zhou, X., Wang, D., Kraehenbuehl, P. Objects as points. arXiv:1904.07850. 2019. <https://arxiv.org/abs/1904.07850>

[5] He, K., Zhang, X., Ren, S., Sun, J. Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. ICCV 2015. 2015. <https://doi.org/10.1109/ICCV.2015.123>

[6] Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., Zisserman, A. The PASCAL visual object classes (VOC) challenge. International Journal of Computer Vision 88, 303-338. 2010. <https://doi.org/10.1007/s11263-009-0275-4>

[7] Lin, T.-Y., et al. Microsoft COCO: common objects in context. ECCV 2014. 2014. <https://doi.org/10.1007/978-3-319-10602-1_48>

[8] Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[9] Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600-612. 2004. <https://doi.org/10.1109/TIP.2003.819861>

[10] Butler, H., et al. The GeoJSON format. RFC 7946. 2016. <https://www.rfc-editor.org/rfc/rfc7946>

[11] Bradner, S. Key words for use in RFCs to indicate requirement levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[12] Leiba, B. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/capabilities/01_ai_products.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
