<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/model_zoo/licensing_and_provenance.md
Title       : Model Licensing and Provenance
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Model Licensing and Provenance

| Field | Value |
| --- | --- |
| Document | UBX-DOC-707 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch, model zoo catalogue version 2.0.0 |

## Abstract

This document states under which licence the models of the Unbihexium model zoo are provided and where their weights come from. It is written for users who build, train or redistribute models, for redistributors and packagers, and for auditors who need to trace a model to its origin. It explains that the catalogue, the architecture code, the generated metadata and the starter weights are artefacts of the project licensed under the Mozilla Public License 2.0; that the starter weights are derived deterministically from the model identifier and contain no third-party weights and no third-party data; which third-party names and formulas the catalogue refers to; and that users who train their own weights are responsible for the licences of their data. It lists the provenance records the software keeps and the information that SHOULD accompany a shared trained model. This document is informational and is not legal advice.

## Contents

1. [Introduction](#1-introduction)
2. [Licence of the model zoo](#2-licence-of-the-model-zoo)
3. [Provenance of the starter weights](#3-provenance-of-the-starter-weights)
4. [Third-party references in the catalogue](#4-third-party-references-in-the-catalogue)
5. [Trained weights](#5-trained-weights)
6. [Provenance records](#6-provenance-records)
7. [Related documents](#7-related-documents)
8. [References](#references)

## 1. Introduction

### 1.1 Purpose and status

The model zoo defines 520 models (130 families in the variants `tiny`, `base`, `large` and `mega`). Every learned model is an untrained starter model with deterministic weights; only the 28 models of the 7 spectral index families compute published formulas and need no training. The licensing and provenance statements below concern these starter models and the metadata that describe them. They do not cover weights that users create by training.

This document summarises the position of the project; it is not legal advice. The licence text in [LICENSE.txt](../../LICENSE.txt) prevails over any summary, and decisions that depend on licensing SHOULD be taken with a qualified adviser. The project-wide licence and regulatory information is in [COMPLIANCE.md](../../COMPLIANCE.md), and the licences of third-party components in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

## 2. Licence of the model zoo

### 2.1 What is licensed under the MPL-2.0

| Artefact | Location | Licence |
| --- | --- | --- |
| Catalogue | `src/unbihexium/zoo/catalog.yaml` | MPL-2.0 |
| Architecture, initialisation, training and inference code | `src/unbihexium/ai/`, `src/unbihexium/zoo/` | MPL-2.0 |
| Published digests | `src/unbihexium/zoo/digests.json`, `model_zoo/checksums.txt` | MPL-2.0 |
| Generated metadata | `model_zoo/` (model cards, manifests, inventory, capability map, schema) | MPL-2.0 |
| Starter weights of the 492 learned models | Built locally; not stored anywhere in the project | MPL-2.0, declared by the project |

Every catalogue entry and every manifest records the licence identifier `MPL-2.0` (the constant `MODEL_LICENSE` in `unbihexium.zoo.catalog`), and the manifest schema accepts no other value. The REUSE annotation in [REUSE.toml](../../REUSE.toml) assigns `MPL-2.0` to every file of the repository, which covers the JSON manifests that cannot carry a comment header. The spectral index models have no weights; their formulas are implemented in MPL-2.0 code.

### 2.2 The starter weights as project artefacts

The starter weights are not data that anybody collected or trained. They are the deterministic output of MPL-2.0 code run on the model identifier (Section 3), and the same weights are produced on every machine. The project treats them as artefacts of the project and declares them MPL-2.0 like the code that produces them. The MPL-2.0 is a file-level copyleft licence [3]: files of the project that are modified and distributed remain under the MPL-2.0, while a larger work that uses the library, including a user's own code, may be under another licence. Redistributors of a checkpoint or ONNX export of a starter model SHOULD keep the licence identifier and point to the source of the project, as for any other MPL-2.0 covered file (see [COMPLIANCE.md](../../COMPLIANCE.md#2-project-licence)).

## 3. Provenance of the starter weights

### 3.1 Derivation

The weights of a starter model depend on exactly three inputs, all of which are part of the repository:

1. the catalogue entry of the family and the variant hyperparameters, which fix the architecture and the tensor shapes;
2. the architecture and initialisation code in `src/unbihexium/ai/models/`;
3. the model identifier, from which the seed is computed as the first four bytes of the SHA-256 hash of `unbihexium:<model id>` (details in [distribution.md](distribution.md#3-how-a-model-is-built)).

No Earth observation data, no labels, no pretrained network and no download take part in the derivation. The code does not load pretrained backbones from any source. The weights are therefore not a derivative of any third-party dataset or model, and they carry no information about the Earth; this is also why they must be trained before use.

### 3.2 Verifiability

Because the derivation is deterministic, anyone can rebuild a model and compare its weights digest with the published digest in [src/unbihexium/zoo/digests.json](../../src/unbihexium/zoo/digests.json) or [model_zoo/checksums.txt](../../model_zoo/checksums.txt) ([download_and_verify.md](download_and_verify.md)). The workflow `.github/workflows/model-zoo.yml` does so in continuous integration: the tiny variants on pull requests and all 520 models on pushes to `main` and every week. A digest that matches proves that the weights are the published starter weights and hence that they have the provenance described above.

## 4. Third-party references in the catalogue

The catalogue names third-party products and publications, but it contains no third-party data:

- **Data sources.** The `sources` field names suitable input data, for example Sentinel-1, Sentinel-2, Landsat 8 and 9 or PlanetScope. These are references for the user; no imagery is shipped (see [COMPLIANCE.md](../../COMPLIANCE.md#52-earth-observation-data) for the terms of Copernicus data).
- **Class legends.** Some class lists follow published products; `lulc_classifier` uses the eleven class names of ESA WorldCover [4]. The WorldCover product is distributed under CC BY 4.0; users who train on or display WorldCover data MUST follow its attribution terms ([THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md#33-esa-worldcover-class-legend)).
- **Formulas.** The spectral index families implement formulas from the publications cited in [model_catalog.md](model_catalog.md#38-spectral-indices). Implementing a published method does not reproduce third-party code; the publications SHOULD be cited when results are published, as described in [CITATION.md](../../CITATION.md).
- **Architectures.** The detector, U-Net and super-resolution networks follow published designs (cited in [model_catalog.md](model_catalog.md#22-tasks-and-architectures)); they are implemented in the project's own code.

## 5. Trained weights

### 5.1 Responsibility of the user

Training creates new weights that depend on the training data. The project makes no statement about the licence of such weights. The user who trains a model is responsible for:

- the licences and terms of use of the imagery, for example the source statement required for Copernicus Sentinel data or the redistribution limits of commercial imagery;
- the licences of the reference data used as labels, for example building footprints, land cover maps or field boundaries, some of which impose attribution or share-alike conditions (such as ESA WorldCover under CC BY 4.0 [4] or OpenStreetMap data under the Open Database License [5]);
- the licence of any pretrained network the user adds, since the library itself uses none;
- personal data that may be contained in very high resolution imagery or in labels, which is discussed in [PRIVACY.md](../../PRIVACY.md) and in [COMPLIANCE.md](../../COMPLIANCE.md#73-general-data-protection-regulation).

Whether and how these conditions extend to the trained weights is a legal question that depends on the terms concerned and the jurisdiction; this document does not answer it.

### 5.2 Sharing a trained model

A shared trained model SHOULD be accompanied by a model card written for it, because the generated cards under `model_zoo/cards/` describe the untrained starter models and are overwritten by the generator. That card SHOULD state:

| Item | Where to find it |
| --- | --- |
| Model identifier, family and variant, and whether the layout was customised | `config` in the checkpoint (`model_id`, `family`, `variant`, `customised`) |
| Weights digest of the trained model | `load_checkpoint(path).digest()` or `weights_digest` in the checkpoint |
| Starting point: starter model or an earlier checkpoint with its digest | The command or code used for training |
| Training, validation and test data: sources, licences, required attributions, area, period, sensor and processing level | The user's records |
| Label source and labelling procedure | The user's records |
| Hyperparameters and epoch of the checkpoint | `training` in the checkpoint (`config`, `epoch`, `metrics`) |
| Accuracy on an independent test split | `unbihexium evaluate --split test` ([training.md](training.md#8-evaluation)) |
| Unbihexium version or commit used | `unbihexium --version` and the Git commit |
| Intended use and known limitations | See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) |

The training metadata stored in a checkpoint can be read without building the model. A checkpoint for the example, trained on synthetic data in place of the toy dataset of [training.md](training.md#51-command), is created with:

```bash
unbihexium train water_surface_detector_tiny --synthetic 16 --epochs 3 --chip-size 64
```

```python
from unbihexium.zoo.checkpoint import read_checkpoint

payload = read_checkpoint("runs/water_surface_detector_tiny/best.pt")
print(payload["config"]["model_id"], payload["config"]["customised"], payload["weights_digest"][:16])
print(sorted(payload["training"]), payload["training"]["epoch"], payload["training"]["config"]["epochs"])
```

```text
water_surface_detector_tiny False 7ed5fc4a9ed71ce4
['config', 'epoch', 'metrics'] 3 3
```

The output shown was printed for the checkpoint trained in [training.md](training.md#51-command) on toy data; the digest of a trained model depends on the data, the options and the numerical environment, so another run can print a different value.

## 6. Provenance records

The software records provenance at every stage, so that a model can be traced from its file back to the catalogue:

| Record | Content |
| --- | --- |
| `src/unbihexium/zoo/digests.json` | Catalogue version, digest algorithm, initialisation method, and the weights digest and parameter count of each model |
| `model_zoo/manifests/<family>.json` | Catalogue version, task, architecture, licence, `status` (`starter` or `reference`), `trained: false`, inputs, outputs, training labels, data sources and, per variant, the model identifier, tile size, parameters and weights digest |
| `model_zoo/cards/<family>.md` | Human-readable version of the manifest with the status statement and the limitations |
| Checkpoint `model.pt`, `best.pt`, `last.pt` | Format version, full model configuration (including normalisation statistics after training), weights digest and training metadata |
| `config.json` in the model store | Catalogue fields, source (`build`, `local` or `url`), published digest and SHA-256 of the checkpoint file |
| ONNX export | Model configuration (`unbihexium_config`) and weights digest (`unbihexium_weights_digest`) as metadata |
| `runs/<model id>/history.json` | Metrics of every training epoch and the best epoch |

Release distributions built by `.github/workflows/release.yml` are signed with Sigstore, with SLSA provenance and GitHub artifact attestations, and they contain `catalog.yaml` and `digests.json`; for a release that includes the model zoo, these signatures therefore also cover the published digests. Releases 1.0.0 and 1.0.1 predate both signing and the model zoo. See [SECURITY.md](../../SECURITY.md#7-verifying-released-artefacts) for verifying a release.

## 7. Related documents

- [COMPLIANCE.md](../../COMPLIANCE.md): project licence, dependency licence policy, data licences and regulatory notes.
- [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md): licences of third-party components and material.
- [NOTICE.md](../../NOTICE.md): copyright, warranty and model zoo notices.
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md): limits of the starter models and responsible use.
- [distribution.md](distribution.md) and [download_and_verify.md](download_and_verify.md): how the weights are built and verified.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[4] Zanaga, D. et al. ESA WorldCover 10 m 2021 v200. Zenodo. 2022. <https://doi.org/10.5281/zenodo.7254221>

[5] Open Data Commons. Open Database License (ODbL) v1.0. 2009. <https://opendatacommons.org/licenses/odbl/1-0/>

<!--
=============================================================================
End of file docs/model_zoo/licensing_and_provenance.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
