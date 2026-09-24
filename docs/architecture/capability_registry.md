<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/architecture/capability_registry.md
Title       : Capability Registry
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Registry

| Field | Value |
| --- | --- |
| Document | UBX-DOC-502 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch, model catalogue 2.0.0 |

## Abstract

This document describes the package `unbihexium.registry` as it is implemented in [src/unbihexium/registry/](../../src/unbihexium/registry/): the capability registry, which describes what the library can do, the model registry, which gives a flat and validating view of the model zoo, and the pipeline registry, which holds the pipeline factories that the command line runs. For each registry it specifies the record type, how the built-in entries are created, the registration and lookup rules, and which parts of the command line interface and the REST service consume it. It is written for contributors who add capabilities, models or pipelines, for integrators who query the registries programmatically or over HTTP, and for reviewers who need to know what the registries do and do not guarantee. All examples were run against the current code, and all counts were measured.

## Contents

- [1. Introduction](#1-introduction)
- [2. Package structure](#2-package-structure)
- [3. The capability record](#3-the-capability-record)
- [4. Built-in capabilities](#4-built-in-capabilities)
- [5. Capability registry operations](#5-capability-registry-operations)
- [6. Model registry](#6-model-registry)
- [7. Pipeline registry](#7-pipeline-registry)
- [8. Consumers of the registries](#8-consumers-of-the-registries)
- [9. Examples](#9-examples)
- [10. Limitations](#10-limitations)
- [11. Related documents](#11-related-documents)

## 1. Introduction

### 1.1 Purpose

The registries answer three questions at run time:

1. Which capabilities does the library offer, in which domain, at which maturity, and which code implements them? (`CapabilityRegistry`)
2. Which models exist, what input do they expect, and is a given input compatible? (`ModelRegistry`)
3. Which named pipelines can be run on files? (`PipelineRegistry`)

The registries are descriptive and dispatching aids. They do not load data, they do not download anything, and, apart from `ModelRegistry.load`, they do not load models.

### 1.2 Status of the models

Every model capability in the registry corresponds to a model family of the model zoo. The zoo has 520 models (130 families in the variants tiny, base, large and mega). Apart from the 28 models of the 7 spectral index families, which compute exact formulas, all of them are untrained starter models with deterministic weights; the registry marks them with the maturity `beta` and the tag `requires_training: "true"` (Section 4.2).

## 2. Package structure

| Module | Contents |
| --- | --- |
| [registry/\_\_init\_\_.py](../../src/unbihexium/registry/__init__.py) | Re-exports and the shortcuts `get_capability`, `list_capabilities` and `register_capability` |
| [registry/capabilities.py](../../src/unbihexium/registry/capabilities.py) | `CapabilityDomain`, `CapabilityMaturity`, `Capability`, the table of library capabilities, `CapabilityRegistry` |
| [registry/models.py](../../src/unbihexium/registry/models.py) | `ModelEntry`, `ModelRegistry` |
| [registry/pipelines.py](../../src/unbihexium/registry/pipelines.py) | `PipelineEntry`, `PipelineRegistry` |

Importing `unbihexium.registry` loads no other `unbihexium` subpackage and does not read the model catalogue. The catalogue (`unbihexium.zoo`) is imported inside the functions that need it, on first access to the capability or model registry.

All three registries are implemented as classes with class-level dictionaries and class methods. There is one instance of each registry per Python process; entries registered by one part of a program are visible to every other part.

## 3. The capability record

### 3.1 Fields

`Capability` is a dataclass with the following fields.

| Field | Type | Meaning |
| --- | --- | --- |
| `capability_id` | `str` | Unique identifier; for model capabilities, the family identifier (for example `ship_detector`) |
| `name` | `str` | Human-readable name |
| `domain` | `CapabilityDomain` | Domain (Section 3.2) |
| `description` | `str` | What the capability does |
| `maturity` | `CapabilityMaturity` | Maturity (Section 3.3), default `stable` |
| `entry_points` | `list[str]` | Import paths of the implementing modules or functions |
| `pipeline_id` | `str` or `None` | Registered pipeline that runs the capability |
| `cli_command` | `str` or `None` | Command line invocation |
| `example_path`, `test_path`, `docs_path` | `str` or `None` | Paths relative to the repository |
| `model_family` | `str` or `None` | Model zoo family, for model capabilities |
| `task` | `str` or `None` | Model task, for example `detection` |
| `bands` | `list[str]` | Input bands of one acquisition, for model capabilities |
| `tags` | `dict[str, str]` | Free-form metadata |

The read-only property `models` returns the four model identifiers of the family (`<family>_tiny`, `_base`, `_large`, `_mega`), or an empty list for library capabilities. `to_dict()` returns a JSON-serialisable dictionary of all fields except `example_path` and `test_path`, with `models` added.

### 3.2 Domains

`CapabilityDomain` is a string enumeration with 16 members. The model catalogue uses the same names in its `domain` field, and `Capability(domain=...)` of a catalogue family is created with `CapabilityDomain(spec.domain)`, so a catalogue entry with an unknown domain fails when the capability registry is loaded.

| Value | Meaning | Value | Meaning |
| --- | --- | --- | --- |
| `ai` | Generic AI models | `imaging` | Image processing |
| `tourism` | Tourism and destination analysis | `assets` | Asset management |
| `analysis` | Spatial analysis | `energy` | Energy infrastructure |
| `indices` | Spectral indices | `urban` | Urban planning |
| `water` | Water and floods | `agriculture` | Agriculture |
| `environment` | Environment monitoring | `risk` | Risk and insurance |
| `forestry` | Forestry | `defense` | Defence and security (neutral monitoring) |
| `sar` | Synthetic aperture radar | `io` | Data input and output |

### 3.3 Maturity levels

| Value | Meaning in the code | Used by the built-in capabilities for |
| --- | --- | --- |
| `stable` | Tested and ready for production use | The 17 library capabilities and the 7 spectral index families |
| `beta` | Complete but needs training or wider validation | The 123 model families that are starter models |
| `research` | Experimental | Not used by built-in capabilities |
| `deprecated` | Scheduled for removal | Not used by built-in capabilities |

`stable` states the maturity of the implementation. It is not a certification, and it does not replace the validation that a user must perform for a particular purpose (see [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md)).

## 4. Built-in capabilities

The built-in capabilities are loaded on first access to the registry (any lookup, listing or registration), in two groups. `unbihexium info` reports the total: `Registered capabilities: 147`.

### 4.1 Library capabilities

Seventeen capabilities describe algorithms and formats implemented directly in the library. They are defined in the tuple `LIBRARY_CAPABILITIES` and all have the maturity `stable`.

| Identifier | Domain | Entry points |
| --- | --- | --- |
| `io_geotiff` | io | `unbihexium.io.geotiff` |
| `io_zarr` | io | `unbihexium.io.zarr_io` |
| `io_geojson` | io | `unbihexium.io.geojson` |
| `io_geoparquet` | io | `unbihexium.io.parquet` |
| `io_stac` | io | `unbihexium.io.stac` |
| `spectral_indices` | indices | `unbihexium.indices` |
| `sar_processing` | sar | `unbihexium.sar` |
| `terrain_analysis` | analysis | `unbihexium.terrain` |
| `geostatistics` | analysis | `unbihexium.geostat` |
| `spatial_analysis` | analysis | `unbihexium.analysis` |
| `image_preprocessing` | imaging | `unbihexium.preprocessing` |
| `prediction_postprocessing` | imaging | `unbihexium.postprocessing` |
| `accuracy_metrics` | analysis | `unbihexium.metrics`, `unbihexium.ai.evaluation` |
| `visualization` | imaging | `unbihexium.visualization` |
| `model_zoo` | ai | `unbihexium.zoo` |
| `model_training` | ai | `unbihexium.ai.training`, `unbihexium.ai.evaluation` |
| `model_serving` | ai | `unbihexium.serving` |

### 4.2 Model capabilities

`catalogue_capabilities()` creates one capability per family of [catalog.yaml](../../src/unbihexium/zoo/catalog.yaml), 130 in total, with these rules:

| Field | Value |
| --- | --- |
| `capability_id`, `model_family` | The family identifier |
| `name`, `description`, `domain`, `task`, `bands` | Copied from the catalogue entry |
| `maturity` | `stable` for the task `spectral_index`, `beta` otherwise |
| `tags` | `{"requires_training": "false"}` for spectral index families, `{"requires_training": "true"}` otherwise |
| `entry_points` | `["unbihexium.ai.predict.predict"]`, the generic runner |
| `pipeline_id` | From `PIPELINES_BY_FAMILY` for five families (Section 7.3), otherwise `None` |
| `cli_command` | `unbihexium predict <family>_base INPUT OUTPUT` |
| `docs_path` | `docs/model_zoo/model_catalog.md` |

Measured distribution of the 147 capabilities:

| Domain | Library | Model families | Domain | Library | Model families |
| --- | --- | --- | --- | --- | --- |
| imaging | 3 | 20 | ai | 3 | 7 |
| agriculture | 0 | 16 | assets | 0 | 7 |
| environment | 0 | 11 | indices | 1 | 7 |
| urban | 0 | 10 | defense | 0 | 5 |
| risk | 0 | 10 | energy | 0 | 5 |
| analysis | 4 | 8 | tourism | 0 | 4 |
| water | 0 | 8 | forestry | 0 | 4 |
| sar | 1 | 8 | io | 5 | 0 |

By task, the 130 model capabilities are 49 dense regression, 26 segmentation, 19 detection, 11 scene regression, 11 enhancement, 7 spectral index, 6 change detection and 1 super-resolution family.

## 5. Capability registry operations

### 5.1 Loading

`CapabilityRegistry._ensure_loaded()` runs once per process. It sets the loaded flag first (so that registration during loading cannot recurse) and inserts the library capabilities followed by the model capabilities with `dict.setdefault`, so a user capability registered before the first access keeps its place and is not overwritten by a built-in capability of the same identifier.

### 5.2 Registration rules

- `register(capability, replace=False)` loads the built-ins first, then raises `ValueError("capability '<id>' is already registered")` if the identifier exists and `replace` is false. With `replace=True` the existing entry is replaced.
- `unregister(capability_id)` removes an entry and returns whether it existed. Built-in capabilities can be removed this way.
- `reset()` forgets every capability, including user registrations; the built-ins are reloaded on the next access.
- Registrations exist only in memory. They are not written to disk and must be repeated in every process.

### 5.3 Lookups

| Method | Returns |
| --- | --- |
| `get(capability_id)` | The capability or `None` |
| `require(capability_id)` | The capability; `KeyError` for unknown identifiers |
| `list_all()`, `list_capabilities()` | All capabilities sorted by identifier |
| `ids()` | All identifiers, sorted |
| `by_domain(domain)` | Capabilities of a domain given as a member or its value; `ValueError` for an unknown domain |
| `by_maturity(maturity)` | Capabilities of a maturity level |
| `by_task(task)` | Capabilities whose `task` equals the given task name or enumeration value |
| `for_model(model_id)` | The model capability that provides a model identifier (the variant suffix is removed with `parse_model_id`), or `None` |
| `search(text)` | Capabilities whose identifier, name or description contains the text, case-insensitively |
| `domain_counts()` | Number of capabilities per domain value |

The shortcuts in `unbihexium.registry` are `get_capability(id)` (same as `get`), `list_capabilities(domain=None)` (same as `by_domain` or `list_all`) and `register_capability(capability, replace=False)`.

## 6. Model registry

### 6.1 Purpose

`ModelRegistry` presents the model zoo as flat `ModelEntry` records, adds input validation and a cache of loaded models, and allows descriptions of models that are not part of the zoo. The source of the zoo entries is `unbihexium.zoo`; the model registry does not copy them but converts them on every call with `ModelEntry.from_zoo`.

### 6.2 The model entry

`ModelEntry` has the fields `model_id`, `name`, `task`, `family`, `domain`, `variant`, `channels` (input channel names, including the `_t1`, `_t2` suffixes of multi-date models), `outputs`, `units`, `sha256` (the published weights digest), `download_url`, `num_parameters`, `license` (default `MPL-2.0`), `source` (default `build`), `requires_training`, `tile_size` and `tags`. The property `in_channels` is the number of channel names.

### 6.3 Operations

| Method | Behaviour |
| --- | --- |
| `register(entry, replace=False)` | Adds a description. Raises `ValueError` when the identifier is a model zoo identifier (zoo models cannot be shadowed) or is already registered without `replace` |
| `unregister(model_id)` | Removes a registered description |
| `get(model_id)` | Registered description first, else the zoo entry, else `None`. A family name without variant resolves to the base variant (`ship_detector` gives `ship_detector_base`) |
| `require(model_id)` | As `get`, but `KeyError` for unknown identifiers |
| `list_all(task, domain, variant)` | Zoo models (filtered by `unbihexium.zoo.list_models`), then registered descriptions that match the same filters |
| `ids()` | Identifiers of `list_all()` |
| `check_input(model_id, shape)` | Validates a `(bands, height, width)` or `(height, width)` shape: the number of bands must equal `in_channels` and both spatial sizes must be at least 1; raises `ValueError` otherwise and returns the entry |
| `load(model_id, variant=None, verify=True)` | Loads a zoo model with `unbihexium.zoo.load_model` and caches it per `(model_id, variant)`; requires the `torch` extra |
| `clear_cache()` | Empties the cache of loaded models |

A description registered with `ModelRegistry.register` is metadata only. It appears in listings and in `GET /models` of the REST service (unless its `source` is `external`), but `ModelRegistry.load` and the prediction routes resolve models through `unbihexium.zoo`, which does not know it. To make a trained checkpoint loadable under its own identifier, register it with `unbihexium.zoo.register_model` instead ([model_zoo_architecture.md](model_zoo_architecture.md), Section 6.4).

## 7. Pipeline registry

### 7.1 The pipeline entry

`PipelineEntry` has the fields `pipeline_id`, `name`, `description`, `config_class` (unused by the built-in pipelines), `factory` (a callable that returns a pipeline), `domains` and `tags`. `to_dict()` returns the identifier, name, description, domains and tags.

### 7.2 Operations

| Method | Behaviour |
| --- | --- |
| `register(pipeline_id, name, description="", domains=None, tags=None)` | Returns a decorator that stores the decorated factory under the identifier and returns the factory unchanged. The identifier must be non-empty and consist of letters, digits, `_` and `-`, otherwise `ValueError`. An existing entry with the same identifier is replaced without warning |
| `get(pipeline_id)`, `require(pipeline_id)` | Entry or `None`; `require` raises `KeyError` listing the known identifiers |
| `create(pipeline_id, **kwargs)` | Calls the factory with the keyword arguments and returns its result; returns `None` for an unknown identifier or an entry without factory |
| `list_all()`, `list_pipelines()`, `ids()` | Entries or identifiers sorted by identifier |
| `by_domain(domain)` | Entries whose `domains` contain the domain |
| `search(text)` | Entries whose identifier, name or description contains the text |
| `unregister(pipeline_id)` | Removes an entry |

### 7.3 Built-in pipelines

The five built-in pipelines are registered by `unbihexium.ai.base.register_task_pipeline` when the task API modules are imported, that is, when `unbihexium.ai` is imported. Each wraps one task API (Section 3 of [pipeline_framework.md](pipeline_framework.md)).

| Pipeline | Domains | Task API | Default model family | Inputs |
| --- | --- | --- | --- | --- |
| `ship_detection` | ai, maritime | `ShipDetector` | `ship_detector` | `input` |
| `building_detection` | ai, urban | `BuildingDetector` | `building_detector` | `input` |
| `water_detection` | ai, water | `WaterDetector` | `water_surface_detector` | `input` |
| `change_detection` | ai, change | `ChangeDetector` (`predict_pair`) | `change_detector` | `input1`, `input2` |
| `super_resolution` | ai, imaging | `SuperResolution` (`enhance`) | `super_resolution` | `input` |

The domains `maritime` and `change` are not members of `CapabilityDomain`; pipeline domains are free-form strings.

## 8. Consumers of the registries

| Consumer | Registry use |
| --- | --- |
| `unbihexium info` | Number of capabilities (`CapabilityRegistry.ids()`) and of pipelines (after importing `unbihexium.ai`) |
| `unbihexium pipeline list [--domain D]` | `PipelineRegistry.list_all()` or `by_domain(D)` |
| `unbihexium pipeline run` | `PipelineRegistry.create(pipeline_id, **params)` |
| `GET /capabilities[?domain=D]`, `GET /capabilities/{capability_id}` | `CapabilityRegistry.list_all()`, `by_domain`, `get`; an unknown domain gives HTTP 422, an unknown identifier 404 |
| `GET /models`, `GET /models/{model_id}` | `ModelRegistry.list_all` and `get` through `ModelInferenceService` |
| `POST /predict/{model_id}` and the earlier routes | `ModelRegistry.get` and `check_input` before any model is opened |
| `GET /pipelines` | `PipelineRegistry.list_all()` after importing `unbihexium.ai` |

The REST responses expose a subset of the capability fields: `id`, `name`, `domain`, `maturity`, `description`, `task`, `models` and `pipeline_id`.

## 9. Examples

### 9.1 Querying and extending the registries

The following script needs only the core installation: it neither imports PyTorch nor builds a model.

```python
from unbihexium.registry import (
    Capability,
    CapabilityDomain,
    CapabilityMaturity,
    CapabilityRegistry,
    ModelRegistry,
    PipelineRegistry,
    get_capability,
    list_capabilities,
    register_capability,
)

# Built-in capabilities are loaded on first access.
print(len(CapabilityRegistry.ids()), CapabilityRegistry.domain_counts()["io"])
print(len(CapabilityRegistry.by_maturity("stable")), len(CapabilityRegistry.by_maturity("beta")))

ships = get_capability("ship_detector")
print(ships.task, ships.maturity.value, ships.pipeline_id, ships.models[0], ships.tags)
print(CapabilityRegistry.for_model("ndvi_calculator_mega").maturity.value)
print([c.capability_id for c in list_capabilities("sar")][:3])

# A user capability; registering the same id again raises ValueError.
register_capability(
    Capability(
        capability_id="my_ndvi_composite",
        name="Monthly NDVI composite",
        domain=CapabilityDomain.INDICES,
        maturity=CapabilityMaturity.RESEARCH,
        entry_points=["my_project.composites.monthly_ndvi"],
    )
)
print([c.capability_id for c in CapabilityRegistry.search("ndvi composite")])

# The model registry is a flat, validating view of the model zoo.
entry = ModelRegistry.check_input("ndvi_calculator_tiny", (2, 64, 64))
print(entry.channels, entry.requires_training)
try:
    ModelRegistry.check_input("ship_detector_tiny", (4, 64, 64))
except ValueError as error:
    print(error)

# Pipelines are registered when unbihexium.ai is imported.
import unbihexium.ai  # noqa: E402,F401

print(PipelineRegistry.ids())
```

Output:

```text
147 5
24 123
detection beta ship_detection ship_detector_tiny {'requires_training': 'true'}
stable
['ground_displacement', 'sar_amplitude', 'sar_flood_detector']
['my_ndvi_composite']
['red', 'nir'] False
ship_detector_tiny expects 3 bands (red, green, blue), got 4
['building_detection', 'change_detection', 'ship_detection', 'super_resolution', 'water_detection']
```

### 9.2 Registering a pipeline

```python
import numpy as np

from unbihexium.core.pipeline import Pipeline, PipelineConfig
from unbihexium.registry import PipelineRegistry


@PipelineRegistry.register("scale_values", "Scale values", "Multiply an array by a factor", ["analysis"])
def create_scale_pipeline(factor: float = 2.0) -> Pipeline:
    pipeline = Pipeline(PipelineConfig("scale_values", "Scale values", parameters={"factor": factor}))
    pipeline.add_step(lambda values: {"array": values["array"] * factor}, name="scale")
    return pipeline


pipeline = PipelineRegistry.create("scale_values", factor=3.0)
run = pipeline.run({"array": np.ones(3)})
print(run.status.value, run.results["array"], PipelineRegistry.get("scale_values").domains)
print(PipelineRegistry.create("unknown_pipeline"))
```

Output:

```text
completed [3. 3. 3.] ['analysis']
None
```

A pipeline registered this way is visible to `unbihexium pipeline list` only if the module that registers it is imported in the same process; the CLI imports `unbihexium.ai` but no user modules.

### 9.3 Command line

```bash
unbihexium info
unbihexium pipeline list --domain water
```

The first command prints the version, `Registered capabilities: 147`, `Model zoo models: 520 (catalogue 2.0.0)` and `Registered pipelines: 5`; the second prints a table with the single row `water_detection`.

## 10. Limitations

- **No persistence and no discovery.** All registrations live in process memory. There is no plug-in discovery through entry points; user capabilities and pipelines exist only in processes that import the registering code.
- **No locking.** `CapabilityRegistry`, `ModelRegistry` and `PipelineRegistry` do not lock their dictionaries. Register entries at start-up, before threads or the REST service's worker threads use them.
- **Silent replacement of pipelines.** Unlike the capability and model registries, `PipelineRegistry.register` replaces an existing entry of the same identifier without an error.
- **Unbounded model cache.** `ModelRegistry.load` keeps every loaded model until `clear_cache()` is called. The REST service does not use this cache; it keeps its own LRU cache of at most `UNBIHEXIUM_SERVING__MODEL_CACHE_SIZE` opened models (default 4).
- **Descriptive metadata only.** `entry_points`, `cli_command` and `docs_path` are strings; the registry does not check that they resolve. `maturity` is set by the rules in Section 4 and is not derived from test results.

## 11. Related documents

- [overview.md](overview.md): package layout and dependencies.
- [pipeline_framework.md](pipeline_framework.md): pipelines, run records and the `unbihexium pipeline` commands.
- [model_zoo_architecture.md](model_zoo_architecture.md): the catalogue that the model capabilities are derived from.
- [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md): the model families.
- [docs/reference/api.md](../reference/api.md) and [docs/reference/cli.md](../reference/cli.md): API and CLI reference.

<!--
=============================================================================
End of file docs/architecture/capability_registry.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
