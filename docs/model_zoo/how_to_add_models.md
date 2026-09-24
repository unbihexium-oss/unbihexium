<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/model_zoo/how_to_add_models.md
Title       : Adding Models to the Model Zoo
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Adding Models to the Model Zoo

| Field | Value |
| --- | --- |
| Document | UBX-DOC-705 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The main branch of Unbihexium, model zoo catalogue version 2.0.0 (not part of release 1.0.1) |

## Abstract

This document describes how a model family is added to the Unbihexium model zoo, and how a user can make a model of their own available to the library without changing the repository. It is written for contributors who propose catalogue changes and for maintainers who review them. Because the zoo distributes no weight files, adding a model means adding a catalogue entry, and code only when the family needs a new spectral index formula, task or architecture. The document specifies the fields and rules of a catalogue entry, the regeneration of the digests, manifests and model cards with `python -m unbihexium.zoo.sync`, the tests and counts that must be updated, the checks that must pass locally and in continuous integration, the versioning of the catalogue, and a review checklist. The procedure was exercised on a copy of the repository on 2026-09-24 by adding a test family; the observed results are reported where they help. A new learned family is an untrained starter model like every other learned model of the zoo.

## Contents

1. [Introduction](#1-introduction)
2. [Kinds of change](#2-kinds-of-change)
3. [Adding a family with an existing task](#3-adding-a-family-with-an-existing-task)
4. [Adding a spectral index](#4-adding-a-spectral-index)
5. [Adding a task or changing an architecture](#5-adding-a-task-or-changing-an-architecture)
6. [Registering a model without changing the repository](#6-registering-a-model-without-changing-the-repository)
7. [Review checklist](#7-review-checklist)
8. [References](#references)

## 1. Introduction

### 1.1 Scope

The catalogue [src/unbihexium/zoo/catalog.yaml](../../src/unbihexium/zoo/catalog.yaml) is the single source of truth of the model zoo. Everything else that describes a model, the published digests in `src/unbihexium/zoo/digests.json` and the files under [model_zoo/](../../model_zoo/README.md), is generated from it together with the architecture code. The zoo contains no weight files and does not use Git LFS ([distribution.md](distribution.md)); a contributed family therefore consists of its catalogue entry, the generated files and, where needed, code and tests. Trained weights are not added to the repository by this procedure.

The general contribution rules (development environment, code and configuration style, commit messages, Developer Certificate of Origin, review) are in [CONTRIBUTING.md](../../CONTRIBUTING.md). This document adds what is specific to the model zoo.

### 1.2 Status of new models

A new learned family is a starter model: its weights are initialised deterministically from its model identifiers and it is **not** trained. Its catalogue entry, and hence its model card, describes what the model does once trained. Descriptions MUST NOT claim trained behaviour or accuracy. Only spectral index families compute their output without training.

### 1.3 Conventions

The key words MUST, MUST NOT, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

## 2. Kinds of change

| Change | Files to edit by hand | Section |
| --- | --- | --- |
| New family with an existing task (detection, segmentation, change detection, dense regression, scene regression, enhancement, super-resolution) | `catalog.yaml`, tests with counts, documents with counts, `CHANGELOG.md` | [3](#3-adding-a-family-with-an-existing-task) |
| New spectral index | The above, plus `src/unbihexium/ai/models/spectral.py` and its tests | [4](#4-adding-a-spectral-index) |
| New task or new architecture, or a change to an existing architecture | The above, plus the model, training, evaluation, inference and generator code and `model_zoo/manifest.schema.json` | [5](#5-adding-a-task-or-changing-an-architecture) |
| A model for one's own use, for example a trained checkpoint | None; registration at run time | [6](#6-registering-a-model-without-changing-the-repository) |

A contributor SHOULD open an issue with the model zoo issue form before a larger change, so that the scope of the family and its domain are agreed before review.

## 3. Adding a family with an existing task

### 3.1 Design the entry

Every family has the fields below. `unbihexium.zoo.catalog` validates them when the catalogue is loaded (`CatalogError` on violation), and `tests/unit/test_zoo_catalog.py` checks further rules.

| Field | Required | Rules |
| --- | --- | --- |
| `id` | Yes | Lowercase letters, digits and underscores, starting with a letter and not ending with an underscore (manifest schema pattern `^[a-z][a-z0-9_]*[a-z0-9]$`); unique. It MUST NOT end with `_tiny`, `_base`, `_large` or `_mega`, because those suffixes are parsed as the variant. |
| `name` | Yes | Human-readable name (British spelling as in the rest of the catalogue) |
| `task` | Yes | One of `detection`, `segmentation`, `change_detection`, `dense_regression`, `scene_regression`, `enhancement`, `super_resolution`, `spectral_index` |
| `domain` | Yes | One of the capability domains of `unbihexium.registry.CapabilityDomain`: `ai`, `tourism`, `analysis`, `indices`, `water`, `environment`, `forestry`, `imaging`, `assets`, `energy`, `urban`, `agriculture`, `risk`, `defense`, `sar`, `io`; any other value makes the capability registry fail |
| `description` | Yes | What the model does once trained, one sentence; non-empty |
| `bands` | Yes | Name of a band set under `band_sets`, or an explicit list of band names; a new band set MAY be added when several families share it |
| `dates` | No (1) | Number of acquisitions stacked on the channel axis; MUST be 2 for `change_detection` |
| `outputs` | Yes | Class names (detection, segmentation, change detection), target names (regression) or output bands (enhancement, super-resolution), in channel order; segmentation and change detection need at least two classes, and the first class of a change detection family is the no-change class by convention |
| `units` | For regression | One unit per output; `"1"` for dimensionless values |
| `range` | No | `[min, max]` of regression targets with `min < max`; `[0, 1]` selects a sigmoid output |
| `scale` | For super-resolution | Upscaling factor, at least 2 |
| `formula` | For spectral indices | Formula identifier implemented in `spectral.py` (Section 4) |
| `labels` | Yes | The reference data needed for training; non-empty |
| `sources` | Yes | Suitable input data; non-empty list |

Families in sensitive areas (for example the `defense` domain) MUST also satisfy [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), in particular [its section on contributions in sensitive areas](../../RESPONSIBLE_USE.md#7-contributions-in-sensitive-areas).

### 3.2 Edit the catalogue

Add the entry at the end of the group of its task, so that the generated files keep the catalogue order. The catalogue follows the configuration file documentation style of [CONTRIBUTING.md](../../CONTRIBUTING.md#53-configuration-file-documentation-style): every line that holds content MUST have a comment. The group headers state the number of families of the group and MUST be updated. For example, a segmentation family for snow cover in Sentinel-2 imagery:

```yaml
  - id: snow_cover_mapper  # Models snow_cover_mapper_tiny, _base, _large and _mega.
    name: Snow Cover Mapper  # Display name.
    task: segmentation  # U-Net class map.
    domain: environment  # Capability domain in the registry.
    description: Segments snow-covered ground in Sentinel-2 imagery.  # What the model does once trained.
    bands: s2_10  # Band set s2_10, 10 inputs.
    outputs: [no_snow, snow]  # 2 output channels.
    labels: Snow cover masks.  # Reference data needed for training.
    sources: [Sentinel-2 Level-2A 10 m and 20 m bands]  # Suitable input data.
```

The catalogue version at the top of the file (`version: "2.0.0"`) SHOULD be increased according to [VERSIONING.md](../../VERSIONING.md#62-catalogue-version): the MINOR number when families are added, the MAJOR number when an existing model changes or is removed, the PATCH number for documentation fields only. The default `version` of `ModelZooEntry` in `src/unbihexium/zoo/registry.py` repeats the catalogue version as a literal and MUST be changed with it.

A quick check that the catalogue still loads:

```bash
python -c "from unbihexium.zoo import list_specs; print(len(list_specs()))"
```

### 3.3 Regenerate the model zoo files

```bash
python -m unbihexium.zoo.sync --root .
```

The generator builds every model of the catalogue in every variant, including `mega`, computes the weights digests and parameter counts, and rewrites `src/unbihexium/zoo/digests.json`, `model_zoo/inventory.yaml`, `model_zoo/capability_to_models.yaml`, `model_zoo/checksums.txt`, `model_zoo/MODEL_CARDS.md` and one manifest and one model card per family, writing only the files whose content changed. It needs PyTorch. It has no shortcut for new families: `--skip-digests` reuses `digests.json` and therefore fails with `4 models have no digest` for a family that has none yet.

In the trial on 2026-09-24 (4 vCPUs, CPython 3.13, PyTorch 2.14, CPU only), adding the family above and running the generator took 10 minutes 41 seconds of wall-clock time and reported `267 model zoo files, 7 written.`: the digests, the inventory, the capability map, the checksums, the card index, and the new card and manifest. The digests of the 520 existing models were unchanged.

Before the generator is run, the consistency check reports the missing files:

```text
4 models have no digest; run without --skip-digests/--check
::error file=model_zoo/manifests::no manifest for snow_cover_mapper
::error file=model_zoo/cards::no model card for snow_cover_mapper
3 model zoo problem(s) found.
```

### 3.4 Confirm that no existing model changed

Adding a family MUST NOT change the digest of any existing model. Check that the diff of the digests only adds lines:

```bash
git diff --stat src/unbihexium/zoo/digests.json model_zoo/checksums.txt
git diff src/unbihexium/zoo/digests.json | grep '^-' | grep -v '^---'
```

The second command SHOULD print nothing except the changed `catalog_version` line when the version was increased. Any other removed line means that an existing model changed, which is a MAJOR catalogue change (Section 5.3).

### 3.5 Update the counts

Several tests and documents state the number of families and models. With one new family (131 families, 524 models) the following MUST be updated:

| File | What to change |
| --- | --- |
| `tests/unit/test_zoo_catalog.py` | `EXPECTED_TASKS` (families per task), the family count and the model count |
| `tests/unit/test_registry.py` | The model count of the model registry (and the family count in its comments) |
| `tests/unit/test_serving.py` | `models_available` of the health endpoint |
| `README.md`, `docs/model_zoo/model_catalog.md` and other documents that state the counts | Counts, the task table and the per-task and per-domain tables |
| `CHANGELOG.md` | An entry under `[Unreleased]`, Added |

`git grep -n -E '\b(130|520)\b'` finds most places. Documents MUST keep stating that the new learned models are untrained starter models.

In the trial, `pytest tests/unit/test_zoo_catalog.py` failed as expected before the counts were updated:

```text
>       assert len(specs) == 130
E       AssertionError: assert 131 == 130
```

### 3.6 Run the checks

```bash
python .github/scripts/check_model_zoo.py                        # files, schema, one card and manifest per family
python .github/scripts/check_model_zoo.py --rebuild tiny         # rebuild the tiny variants and compare digests
python scripts/validate_models.py --family snow_cover_mapper --onnx  # forward pass and ONNX export
python -m pytest tests/unit/test_zoo_catalog.py tests/unit/test_registry.py tests/unit/test_serving.py
make check                                                       # everything CI runs locally
```

In the trial, after the generator had run, the results were: `check_model_zoo.py` reported `267 model zoo files checked, 0 out of date.` and `Model zoo check passed.`; the tiny rebuild reported `Rebuilt 131 models: 131 reproduce their published digest.`; and `validate_models.py` reported `4 models validated, 0 failed.` Before the generator had run, the same validation reported `weights digest differs from the published digest` for all four new models.

A quick functional test SHOULD train the new family briefly on synthetic data, which exercises the loss, the metrics and the checkpoint code for its layout:

<!-- doc-example: skip (needs the new family of the previous steps in the catalogue) -->
```bash
unbihexium train snow_cover_mapper_tiny --synthetic 16 --epochs 1 --chip-size 64
```

### 3.7 Open the pull request

The pull request MUST contain the catalogue change, all regenerated files and the updated tests and documents, and SHOULD follow the pull request template (`.github/PULL_REQUEST_TEMPLATE.md`). The workflow `.github/workflows/model-zoo.yml` then checks the generated files against the catalogue and rebuilds the tiny variants; after the merge, the push to `main` rebuilds all 520 (then 524) models, and the weekly run repeats this.

## 4. Adding a spectral index

A spectral index family computes a formula instead of learning one. In addition to the steps of Section 3:

1. Implement the formula in `src/unbihexium/ai/models/spectral.py`: add the formula identifier and its number of input channels to `FORMULA_CHANNELS`, add the computation to `SpectralIndex.forward` using `safe_divide` for divisions (NaN where the denominator is zero), and cite the primary publication in the module header.
2. Add the catalogue entry with `task: spectral_index`, `domain: indices`, the input bands in the order the formula expects, a single output, `formula: <identifier>`, `labels: None, the formula needs no training.` and suitable data sources.
3. Add a unit test that compares the module with the formula on known values, including a zero denominator.
4. Run the generator: the family gets the status `reference` and zero parameters.

If the index also belongs in `unbihexium.core.index.IndexRegistry` (used by `unbihexium index`), that registry is extended separately; the two implementations SHOULD agree and a test SHOULD compare them.

## 5. Adding a task or changing an architecture

### 5.1 A new task

A new task touches every layer that dispatches on `unbihexium.zoo.Task`. The following MUST be extended consistently:

| Location | Change |
| --- | --- |
| `src/unbihexium/zoo/catalog.py` | `Task` enumeration and its properties (`is_dense`, `is_regression`, `is_classification`, `is_trainable`), catalogue validation |
| `src/unbihexium/ai/models/networks.py`, `factory.py` | Network and its construction in `_make_network` |
| `src/unbihexium/ai/models/init.py` | Only if the initialisation needs a special output layer (`init_std`, `init_bias`) |
| `src/unbihexium/ai/losses.py` | Loss in `TaskLoss` |
| `src/unbihexium/ai/evaluation.py` | Metrics in `TaskEvaluator` and the monitored metric in `MONITOR` |
| `src/unbihexium/ai/data.py`, `transforms.py`, `training.py` | Target format on disk, synthetic samples, cropping, padding and augmentation, target encoding in `ChipDataset` |
| `src/unbihexium/ai/inference.py`, `predict.py`, a task API module and `results.py` | Inference, task API, result type and `write_result` |
| `src/unbihexium/zoo/sync.py` | `ARCHITECTURES` and `OUTPUT_LAYOUT` |
| `model_zoo/manifest.schema.json` | The `task` and `architecture` enumerations |
| `tests/` | Unit tests for every layer, and the task in the catalogue tests |
| `docs/model_zoo/` | [training.md](training.md) (dataset layout), [inference.md](inference.md), [model_catalog.md](model_catalog.md) |

### 5.2 Initialisation and export requirements

New layers MUST be initialised by `unbihexium.ai.models.init.initialize`: it covers convolution, linear and group normalisation layers; any other layer with parameters would keep PyTorch's non-deterministic initialisation and break the reproducibility check. A new network MUST export to ONNX with dynamic height and width and MUST pass `scripts/validate_models.py --onnx`.

### 5.3 Changing an existing architecture

A change to the architecture or initialisation code of an existing task changes the digests of every model of that task. It is a MAJOR change of the catalogue version under [VERSIONING.md](../../VERSIONING.md#62-catalogue-version) and MUST be recorded as a breaking change in [CHANGELOG.md](../../CHANGELOG.md). A change to the network code also affects checkpoints that users trained with the old code: checkpoints whose tensor names or shapes differ no longer load, because checkpoints are loaded with `strict=True`, and checkpoints that still load may compute a different function. A change to the initialisation code alone changes only the starter weights. The generator rewrites the digests; the pull request MUST explain why the change is needed.

## 6. Registering a model without changing the repository

A user who wants a model of their own, for example a trained checkpoint, to be available under a model identifier can register it at run time with `unbihexium.zoo.register_model`. The registration lasts for the Python process. The example registers a checkpoint trained as in [training.md](training.md); a short run on synthetic data gives one:

```bash
unbihexium train water_surface_detector_tiny --synthetic 16 --epochs 1 --chip-size 64
```

```python
from unbihexium.zoo import (
    ModelZooEntry,
    ensure_model,
    get_spec,
    get_variant,
    load_model,
    register_model,
    verify_model,
)
from unbihexium.zoo.checkpoint import read_checkpoint

checkpoint = "runs/water_surface_detector_tiny/best.pt"
register_model(
    ModelZooEntry(
        model_id="water_surface_detector_lake",
        spec=get_spec("water_surface_detector"),
        variant=get_variant("tiny"),
        weights_digest=read_checkpoint(checkpoint)["weights_digest"],
        source="local",
        local_path=checkpoint,
    )
)
model = load_model("water_surface_detector_lake")
directory = ensure_model("water_surface_detector_lake")
print(model.model_id, sorted(p.name for p in directory.iterdir()), verify_model("water_surface_detector_lake"))
```

```text
water_surface_detector_tiny ['config.json', 'model.pt', 'model.sha256'] True
```

The loaded model keeps the identifier stored in its checkpoint. A registered entry with `source="url"` and `download_url` is downloaded into the store on first use; its `weights_digest` SHOULD be set so that `verify_model` can compare the download with it ([distribution.md](distribution.md#62-registered-models)). `unregister_model` removes an entry.

## 7. Review checklist

A pull request that adds or changes a family MUST satisfy:

- [ ] The entry follows the rules of Section 3.1 and has a comment on every line; group header counts are updated.
- [ ] The description, labels and sources are accurate and do not claim trained behaviour or accuracy.
- [ ] The catalogue version is increased as required by VERSIONING.md, and `ModelZooEntry.version` matches it.
- [ ] `python -m unbihexium.zoo.sync --root .` was run and all generated files are committed; no generated file was edited by hand.
- [ ] No existing digest changed, or the change is intended, explained and recorded as breaking.
- [ ] Tests and documents that state counts are updated, and new code has tests.
- [ ] `python .github/scripts/check_model_zoo.py --rebuild tiny`, `scripts/validate_models.py --family <family> --onnx` and the test suite pass.
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`.
- [ ] For families in sensitive areas, RESPONSIBLE_USE.md is respected.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/model_zoo/how_to_add_models.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
