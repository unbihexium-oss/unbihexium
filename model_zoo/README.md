<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : model_zoo/README.md
Title       : Model Zoo Metadata
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Model Zoo Metadata

| Field | Value |
| --- | --- |
| Document | UBX-DOC-700 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch, model zoo catalogue version 2.0.0 |

## Abstract

This directory holds the human-readable and machine-readable description of the 520 models of the Unbihexium model zoo: a model card and a manifest per family, an inventory, a mapping from families to model identifiers, the weights digests and the JSON Schema of the manifests. This document is written for users who look up a model, for contributors who change the catalogue, and for auditors who check the published digests. It explains what each file contains, that every file except this README and the schema is generated from `src/unbihexium/zoo/catalog.yaml` by `python -m unbihexium.zoo.sync` and checked by `.github/scripts/check_model_zoo.py`, and that the directory contains no weight files. Apart from the 7 spectral index families, every model described here is an untrained starter model.

## Contents

1. [Status of the models](#1-status-of-the-models)
2. [Contents of the directory](#2-contents-of-the-directory)
3. [How the files are produced](#3-how-the-files-are-produced)
4. [Checks](#4-checks)
5. [Related documents](#5-related-documents)
6. [References](#references)

## 1. Status of the models

The model zoo has 130 model families in four size variants (`tiny`, `base`, `large` and `mega`), 520 models in total. Every model of the 123 learned families is a **starter model**: a complete, trainable network for its task with deterministic starter weights that has **not** been trained on Earth observation data. Its predictions are meaningless until it is trained or fine-tuned on labelled data for the area, sensor and season of use ([docs/model_zoo/training.md](../docs/model_zoo/training.md)). The 28 models of the 7 spectral index families (NDVI, NDWI, EVI, SAVI, MSI, NBR and VCI) compute published formulas exactly and need no training; their status is `reference`.

No weight files are stored here or anywhere else in the project, and the repository does not use Git LFS. The weights of a model are built locally from the catalogue and verified against the digests listed in this directory ([docs/model_zoo/distribution.md](../docs/model_zoo/distribution.md)).

## 2. Contents of the directory

| Path | Generated | Content |
| --- | --- | --- |
| `README.md` | No | This document |
| [manifest.schema.json](manifest.schema.json) | No | JSON Schema (draft 2020-12) [1] of the manifests: required fields, allowed tasks, architectures and statuses, licence fixed to `MPL-2.0` |
| [MODEL_CARDS.md](MODEL_CARDS.md) | Yes | Index of the model cards: family, task, domain, number of inputs and outputs, status |
| [cards/](cards/) `<family>.md` | Yes | Model card per family (130 files): status statement, overview, input channels in order, output table with units, variants with parameters, tile size and digest prefix, usage, training command and required labels, suitable data, limitations |
| [manifests/](manifests/) `<family>.json` | Yes | Machine-readable manifest per family (130 files): catalogue version, task, domain, architecture, licence, `status`, `trained: false`, inputs, outputs with units, range and tensor layout, training labels, data sources, and per variant the model identifier, tile size, base channels, depth, parameters and full weights digest |
| [inventory.yaml](inventory.yaml) | Yes | One summary per family: name, task, domain, architecture, number of input channels, outputs, licence and status |
| [capability_to_models.yaml](capability_to_models.yaml) | Yes | Per family: name, task, the default model (the `base` variant) and the four model identifiers |
| [checksums.txt](checksums.txt) | Yes | One line `<weights digest>  <model id>  # <n> parameters.` per model, sorted by model identifier |

The digests in `checksums.txt` and in the manifests are SHA-256 weights digests over the sorted state dictionary of a model (key, shape and little-endian float32 values of every tensor), the same values as in the packaged file `src/unbihexium/zoo/digests.json`. They identify the weights independently of the file format, so they cannot be checked with `sha256sum -c`; `unbihexium zoo verify` and `unbihexium.zoo.load_model` compare them ([docs/model_zoo/download_and_verify.md](../docs/model_zoo/download_and_verify.md)).

The directory is part of the source distribution but not of the wheel. The wheel carries the catalogue and `digests.json`, which are all the library needs at run time.

## 3. How the files are produced

### 3.1 Conventions

The key words MUST, SHOULD and MAY in this section are to be interpreted as described in RFC 2119 [2] and RFC 8174 [3] when, and only when, they appear in capitals.

### 3.2 Generation

The single source of truth is [src/unbihexium/zoo/catalog.yaml](../src/unbihexium/zoo/catalog.yaml). The module `unbihexium.zoo.sync` builds every model with its starter weights, computes the digests and parameter counts, and writes `src/unbihexium/zoo/digests.json` and every generated file of this directory:

```bash
python -m unbihexium.zoo.sync --root .          # regenerate after a catalogue or architecture change
python -m unbihexium.zoo.sync --root . --check  # verify only; writes nothing
```

- The generated files MUST NOT be edited by hand. A manual change is overwritten by the next run and makes the consistency check fail.
- After any change to the catalogue or to the architecture code, the generator MUST be run and its output committed in the same pull request.
- `manifest.schema.json` and this README are maintained by hand. A change to the manifest layout in `sync.py` MUST be matched by a change to the schema.
- The model cards describe the starter models. Documentation of a trained model MUST NOT be written into them; see [docs/model_zoo/licensing_and_provenance.md](../docs/model_zoo/licensing_and_provenance.md#52-sharing-a-trained-model).

The complete procedure for adding a family is in [docs/model_zoo/how_to_add_models.md](../docs/model_zoo/how_to_add_models.md).

## 4. Checks

| Command | Checks |
| --- | --- |
| `python .github/scripts/check_model_zoo.py` (or `make model-zoo`) | Every generated file matches the catalogue (`sync --check`); every manifest validates against `manifest.schema.json`; exactly one manifest and one card per family and no files of removed families |
| `python .github/scripts/check_model_zoo.py --rebuild tiny` (or `base`, `large`, `mega`, `all`) | The above, plus rebuilding the models of the selected variants and comparing their digests with the published ones |
| `python scripts/validate_models.py --variant tiny --onnx` | Build, digest, forward pass with an output of the expected shape and finite values, and ONNX export compared with ONNX Runtime |

The workflow [.github/workflows/model-zoo.yml](../.github/workflows/model-zoo.yml) runs the consistency check and a rebuild on pull requests and pushes to `main` that touch this directory, `src/unbihexium/zoo/` or `src/unbihexium/ai/models/`, rebuilding the tiny variants on pull requests and all 520 models on pushes, every Monday at 05:00 UTC and on manual request.

## 5. Related documents

- [docs/model_zoo/model_catalog.md](../docs/model_zoo/model_catalog.md): all 130 families grouped by task and domain.
- [docs/model_zoo/download_and_verify.md](../docs/model_zoo/download_and_verify.md): building, verifying and exporting models.
- [docs/model_zoo/training.md](../docs/model_zoo/training.md) and [docs/model_zoo/inference.md](../docs/model_zoo/inference.md): training and running models.
- [docs/model_zoo/licensing_and_provenance.md](../docs/model_zoo/licensing_and_provenance.md): licence (MPL-2.0) and provenance of the models.
- [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md): limits of the starter models and responsible use.

## References

[1] JSON Schema. JSON Schema: A Media Type for Describing JSON Documents, draft 2020-12. 2022. <https://json-schema.org/draft/2020-12>

[2] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[3] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file model_zoo/README.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
