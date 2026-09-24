<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/index.md
Title       : Capability Domains
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domains

| Field | Value |
| --- | --- |
| Document | UBX-DOC-600 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium (declared version 1.0.1, model catalogue 2.0.0) |

## Abstract

The capability documentation of Unbihexium is organised in twelve numbered domain documents. This index explains how these twelve documents relate to the two classifications that exist in the code: the capability domains of the registry (`unbihexium.registry.CapabilityDomain`) and the `domain` field of the 130 model families in `src/unbihexium/zoo/catalog.yaml`. It gives the counts of families, models and library capabilities per domain, generated from the installed package, states the status of the models, and provides the script that generated the family tables of the domain documents. It is written for users looking for the right document, for contributors who add a family or a capability, and for reviewers who check that the documentation matches the code. The domain documents describe intended applications; apart from the spectral index families, every model is an untrained starter model.

## Contents

1. [How the capabilities are classified](#1-how-the-capabilities-are-classified)
2. [The twelve domain documents](#2-the-twelve-domain-documents)
3. [Status of the models](#3-status-of-the-models)
4. [Regenerating the family tables](#4-regenerating-the-family-tables)
5. [Querying capabilities](#5-querying-capabilities)
6. [Related documents](#6-related-documents)
7. [References](#references)

## 1. How the capabilities are classified

A **capability** is something the library can do, described by an id, a name, a domain, a maturity and the entry points that implement it (`unbihexium.registry.Capability`). The capability registry (`CapabilityRegistry`, in `src/unbihexium/registry/capabilities.py`) loads two kinds of built-in capabilities on first access:

- **model capabilities**, one per model family of the catalogue. The capability id is the family id, its domain is the family's `domain` field, its models are the four size variants `<family>_tiny`, `_base`, `_large` and `_mega`, and it runs through `unbihexium.ai.predict.predict` or `unbihexium predict`;
- **library capabilities**, a fixed list of seventeen algorithms and formats implemented directly in the library (input and output, spectral indices, SAR processing, terrain, geostatistics, spatial analysis, preprocessing, postprocessing, metrics, visualisation, the model zoo, training and serving).

The enumeration `CapabilityDomain` has sixteen values. Fifteen of them are used by the `domain` field of the catalogue; the sixteenth, `io`, holds only library capabilities. The registry currently contains 147 capabilities: 130 model capabilities and 17 library capabilities. The table below was generated from the installed package with the script of Section 5.2.

| Registry domain | Families | Models | Library capabilities |
| --- | --- | --- | --- |
| `ai` | 7 | 28 | model_serving, model_training, model_zoo |
| `tourism` | 4 | 16 | - |
| `analysis` | 8 | 32 | accuracy_metrics, geostatistics, spatial_analysis, terrain_analysis |
| `indices` | 7 | 28 | spectral_indices |
| `water` | 8 | 32 | - |
| `environment` | 11 | 44 | - |
| `forestry` | 4 | 16 | - |
| `imaging` | 20 | 80 | image_preprocessing, prediction_postprocessing, visualization |
| `assets` | 7 | 28 | - |
| `energy` | 5 | 20 | - |
| `urban` | 10 | 40 | - |
| `agriculture` | 16 | 64 | - |
| `risk` | 10 | 40 | - |
| `defense` | 5 | 20 | - |
| `sar` | 8 | 32 | sar_processing |
| `io` | 0 | 0 | io_geojson, io_geoparquet, io_geotiff, io_stac, io_zarr |

## 2. The twelve domain documents

The numbered documents group the registry domains by theme. Nine documents correspond to one or more registry domains and list every family of those domains; three documents (08, 10 and 11) have no catalogue domain of their own and describe products built from families and functions of other domains, mostly of the domain `imaging`, whose families are all listed in document 04. Document 09 is a narrative document without models. Each family is listed with its full description in exactly one document, the one of its catalogue domain.

| No. | Document | Registry domains | Families | Models |
| --- | --- | --- | --- | --- |
| 01 | [AI products](01_ai_products.md) | `ai` | 7 | 28 |
| 02 | [Tourism and data processing](02_tourism_data_processing.md) | `tourism`, `analysis` | 12 | 48 |
| 03 | [Spectral indices, floods and water](03_indices_flood_water.md) | `indices`, `water` | 15 | 60 |
| 04 | [Environment, forestry and image processing](04_environment_forestry_image_processing.md) | `environment`, `forestry`, `imaging` | 35 | 140 |
| 05 | [Asset management and energy](05_asset_management_energy.md) | `assets`, `energy` | 12 | 48 |
| 06 | [Urban planning and agriculture](06_urban_agriculture.md) | `urban`, `agriculture` | 26 | 104 |
| 07 | [Risk and defence (neutral monitoring)](07_risk_defense_neutral.md) | `risk`, `defense` | 15 | 60 |
| 08 | [Value-added imagery](08_value_added_imagery.md) | none of its own (products from `imaging` families) | - | - |
| 09 | [Benefits narrative](09_benefits_narrative.md) | none | - | - |
| 10 | [Satellite imagery features](10_satellite_imagery_features.md) | none of its own (products from `imaging` families) | - | - |
| 11 | [Resolution, metadata and quality assurance](11_resolution_metadata_qa.md) | none of its own (products from `imaging` families) | - | - |
| 12 | [Radar and SAR](12_radar_sar.md) | `sar` | 8 | 32 |
| | Total | | 130 | 520 |

The library capabilities are described in the document of their registry domain: `model_zoo`, `model_training` and `model_serving` in 01; `terrain_analysis`, `geostatistics`, `spatial_analysis` and `accuracy_metrics` in 02 (the hydrological part of `terrain_analysis` in 03); `spectral_indices` in 03; `image_preprocessing`, `prediction_postprocessing` and `visualization` in 04; `sar_processing` in 12. The input and output capabilities of the domain `io` have no numbered document; they are covered by the [API reference](../reference/api.md).

Registered pipelines (`unbihexium pipeline list`) belong to the families they run: `ship_detection` and `change_detection` (01), `water_detection` (03), `super_resolution` (04) and `building_detection` (06).

## 3. Status of the models

The model zoo contains 520 models: 130 families in the four variants `tiny`, `base`, `large` and `mega`. **Apart from the seven spectral index families of document 03 (28 models), which compute exact formulas and have no trainable parameters, every model is an untrained starter model**: a complete, trainable network with the input and output layout of its task and deterministic initial weights whose SHA-256 digest is published in `model_zoo/manifests/`. These weights have not been fitted to Earth observation data, so predictions are meaningless until the model has been trained or fine-tuned on labelled data (see [training](../model_zoo/training.md)). The registry expresses this in the maturity of each model capability: `stable` for the spectral index families, `beta` for all others, with the tag `requires_training`. No accuracy, throughput or latency figures are published for any model.

The family descriptions in the catalogue and in the domain documents state what a model is designed to do once trained. They describe intended applications, not validated products. Before a trained model is used for decisions that affect people, property or the environment, read [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

| Variant | Base channels | Encoder depth | Tile size (pixels) |
| --- | --- | --- | --- |
| `tiny` | 16 | 3 | 256 |
| `base` | 32 | 4 | 256 |
| `large` | 48 | 4 | 512 |
| `mega` | 64 | 5 | 512 |

The variant parameters are those of `unbihexium.zoo.get_variant`; the number of parameters of each model depends on its task and bands and is listed in the family tables of the domain documents.

## 4. Regenerating the family tables

The family tables of the domain documents are generated from the catalogue and the model registry of the installed package, so that they cannot drift from the code. The script below prints the two tables for the catalogue domains given as arguments (for example `python family_tables.py indices water` for document 03); without arguments it prints the tables of the domain `ai`. It builds no network: `get_model` reads the parameter counts from the packaged file `src/unbihexium/zoo/digests.json`.

```python
"""Print Markdown tables of the model families of capability domains."""

import sys

from unbihexium.zoo import get_model, list_specs

VARIANTS = ("tiny", "base", "large", "mega")
DOMAINS = sys.argv[1:] or ["ai"]

specs = [s for d in DOMAINS for s in list_specs(domain=d)]

print("| Family | Domain | Task | Input bands | Outputs | Parameters (tiny / base / large / mega) |")
print("| --- | --- | --- | --- | --- | --- |")
for s in specs:
    bands = ", ".join(s.bands) + (f" (x {s.dates} dates)" if s.dates > 1 else "")
    outputs = ", ".join(s.outputs)
    if s.units:
        outputs += " [" + ", ".join(s.units) + "]"
    params = " / ".join(f"{get_model(f'{s.family}_{v}').num_parameters:,}" for v in VARIANTS)
    print(f"| `{s.family}` | {s.domain} | {s.task.value} | {bands} | {outputs} | {params} |")

print()
print("| Family | Name | Intended application | Reference data needed for training |")
print("| --- | --- | --- | --- |")
for s in specs:
    print(f"| `{s.family}` | {s.name} | {s.description} | {s.labels} |")
```

Run without arguments, it prints the tables of document 01:

```text
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
```

When a family is added to or changed in `catalog.yaml`, the tables of the affected document SHOULD be regenerated with this script, and the counts of Sections 1 and 2 with the script of Section 5.2. The key word SHOULD is used as described in RFC 2119 and RFC 8174 [1], [2].

## 5. Querying capabilities

### 5.1 In Python

The registry answers queries by domain, maturity, task, model id and text. The examples were executed on 24 September 2026 against the main branch with CPython 3.13.

```python
from unbihexium.registry import CapabilityRegistry, list_capabilities

print(len(CapabilityRegistry.list_all()), len(CapabilityRegistry.by_maturity("stable")))
print([c.capability_id for c in list_capabilities("indices")])
print(CapabilityRegistry.for_model("ship_detector_tiny").capability_id)
cap = CapabilityRegistry.get("flood_risk")
print(cap.domain.value, cap.maturity.value, cap.task, cap.tags, cap.cli_command)
```

```text
147 24
['evi_calculator', 'msi_calculator', 'nbr_calculator', 'ndvi_calculator', 'ndwi_calculator', 'savi_calculator', 'spectral_indices', 'vegetation_condition']
ship_detector
water beta dense_regression {'requires_training': 'true'} unbihexium predict flood_risk_base INPUT OUTPUT
```

The 24 stable capabilities are the 17 library capabilities and the 7 spectral index families.

### 5.2 Counts per domain

This script generated the table of Section 1:

```python
from unbihexium.registry import CapabilityDomain, CapabilityRegistry

print("| Registry domain | Families | Models | Library capabilities |")
print("| --- | --- | --- | --- |")
for domain in CapabilityDomain:
    caps = CapabilityRegistry.by_domain(domain)
    families = [c for c in caps if c.model_family]
    library = [c.capability_id for c in caps if not c.model_family]
    print(f"| `{domain.value}` | {len(families)} | {4 * len(families)} | {', '.join(library) or '-'} |")
```

### 5.3 On the command line and over REST

`unbihexium zoo list --domain <domain>` lists the models of a catalogue domain (add `--variant tiny` for one row per family, `--json` for machine-readable output), and `unbihexium zoo info <model id>` prints the bands, outputs, labels and sources of a model. The REST service (`unbihexium.serving`) exposes the registry at `/capabilities` and `/capabilities/{capability_id}`; see the [command line reference](../reference/cli.md) and [docs/model_zoo/inference.md](../model_zoo/inference.md).

## 6. Related documents

- [Model catalogue](../model_zoo/model_catalog.md) and [training](../model_zoo/training.md).
- [Capability registry architecture](../architecture/capability_registry.md) and [pipeline framework](../architecture/pipeline_framework.md).
- [API reference](../reference/api.md) and [command line reference](../reference/cli.md).
- [README.md](../../README.md) and [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md).

## References

[1] Bradner, S. Key words for use in RFCs to indicate requirement levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/capabilities/index.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
