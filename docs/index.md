<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/index.md
Title       : Unbihexium Documentation
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Unbihexium Documentation

| Field | Value |
| --- | --- |
| Document | UBX-DOC-300 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch (model catalogue 2.0.0) |

## Abstract

This is the entry point of the documentation of Unbihexium, the open source Python library for Earth observation, geospatial analysis, remote sensing and synthetic aperture radar. It is written for every reader of the documentation: users who install the library and run analyses, researchers who need to know exactly which methods are implemented, contributors, operators who deploy the command line or the REST service, and security reviewers and auditors. It states the scope and status of the software that the documentation describes, suggests reading paths for each audience, and lists every document under `docs/` by area with a one-line description of what it covers, followed by the policy documents in the repository root and the other README files. The complete flat list of documents is [docs/toc.md](toc.md).

## Contents

1. [Scope and status](#1-scope-and-status)
2. [Reading paths](#2-reading-paths)
3. [Getting started and tutorials](#3-getting-started-and-tutorials)
4. [Reference](#4-reference)
5. [Architecture](#5-architecture)
6. [Capability domains](#6-capability-domains)
7. [Model zoo](#7-model-zoo)
8. [Security](#8-security)
9. [Operations](#9-operations)
10. [Benchmarks](#10-benchmarks)
11. [General documents](#11-general-documents)
12. [Project documents outside docs/](#12-project-documents-outside-docs)
13. [Documentation conventions](#13-documentation-conventions)
14. [References](#references)

## 1. Scope and status

### 1.1 What the documentation describes

The documentation describes the main branch of the repository <https://github.com/unbihexium-oss/unbihexium>. The latest published release is 2.0.0 on PyPI (<https://pypi.org/project/unbihexium/>), tagged `v2.0.0` on 24 September 2026; the main branch matches it apart from changes listed under `[Unreleased]` in [CHANGELOG.md](../CHANGELOG.md). Differences between 1.0.x and 2.0.0 are listed in [MIGRATION.md](MIGRATION.md) and in the changelog.

### 1.2 What the software is

Unbihexium provides, in one typed package for CPython 3.10 to 3.14: raster and vector input and output (GeoTIFF and Cloud Optimized GeoTIFF, Zarr, GeoJSON, GeoParquet, STAC), radiometric preprocessing, spectral indices, SAR processing, terrain and hydrology, geostatistics, spatial analysis, accuracy and image quality metrics, visualisation, a model zoo with training, evaluation, tiled inference and ONNX export, the `unbihexium` command and a FastAPI-based REST service. It is distributed under the Mozilla Public License 2.0 [1].

### 1.3 Status of the models

The model zoo defines 520 models: 130 families in the variants tiny, base, large and mega. **Apart from the 28 models of the 7 spectral index families, which compute exact formulas, every model is an untrained starter model** with deterministic weights; its predictions are not meaningful until it has been trained on labelled data. No trained weights and no accuracy figures are published. Every document that discusses models repeats this, and [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) sets out what follows from it.

## 2. Reading paths

| Reader | Start with | Continue with |
| --- | --- | --- |
| New user | [Installation](getting_started/installation.md), [Quick start](getting_started/quickstart.md) | [Tutorials](tutorials/index.md), [Command line reference](reference/cli.md), [FAQ](faq.md) |
| User of 1.0.x | [Migration guide](MIGRATION.md) | [Python API reference](reference/api.md), [Configuration](getting_started/configuration.md) |
| Researcher | [Capability domains](capabilities/index.md), [Glossary](glossary.md) | The domain documents of Section 6, [Model zoo architecture](architecture/model_zoo_architecture.md), [CITATION.md](../CITATION.md) |
| Model trainer | [Model catalogue](model_zoo/model_catalog.md), [Training](model_zoo/training.md) | [Inference](model_zoo/inference.md), [Building and verifying models](model_zoo/download_and_verify.md) |
| Contributor | [CONTRIBUTING.md](../CONTRIBUTING.md), [Architecture overview](architecture/overview.md) | [Adding models](model_zoo/how_to_add_models.md), [CI/CD](operations/ci_cd.md) |
| Operator | [Container image and deployment](operations/docker.md), [Configuration](getting_started/configuration.md) | [Security model](architecture/security_model.md), [Secrets and tokens](security/secrets_and_tokens.md) |
| Security reviewer or auditor | [Security self-assessment](security/self_assessment.md), [SECURITY.md](../SECURITY.md), [Supply chain security](security/supply_chain_security.md) | [Model integrity](security/model_integrity.md), [Vulnerability management](security/vulnerability_management.md), [Releasing](operations/releasing.md) |

## 3. Getting started and tutorials

| Document | Covers |
| --- | --- |
| [getting_started/installation.md](getting_started/installation.md) | Requirements, installation from PyPI with extras, from the hashed lock file, as a container image and from source; verifying an installation |
| [getting_started/quickstart.md](getting_started/quickstart.md) | A guided first session: a synthetic scene, GeoTIFF and COG input and output, indices, vectorisation, a tiny model, brief training and the same steps on the command line |
| [getting_started/configuration.md](getting_started/configuration.md) | Environment variables, the layered settings of `unbihexium.config`, the model store, logging and the settings of the REST service |
| [tutorials/index.md](tutorials/index.md) | Five executed end-to-end tutorials: spectral indices from a GeoTIFF, building and running a tiny model, training on synthetic data with ONNX export, serving over HTTP, and kriging |

## 4. Reference

| Document | Covers |
| --- | --- |
| [reference/api.md](reference/api.md) | Every public name exported by the subpackages, with signatures, descriptions and runnable examples |
| [reference/cli.md](reference/cli.md) | Every command, subcommand and option of `unbihexium`, with defaults, outputs, exit status and environment variables |

## 5. Architecture

| Document | Covers |
| --- | --- |
| [architecture/overview.md](architecture/overview.md) | The subpackages and their responsibilities, import dependencies, lazy optional dependencies and data flow |
| [architecture/capability_registry.md](architecture/capability_registry.md) | The capability, model and pipeline registries of `unbihexium.registry` |
| [architecture/model_zoo_architecture.md](architecture/model_zoo_architecture.md) | From `catalog.yaml` to a running model: specifications, variants, networks, starter weights, digests, checkpoints and ONNX export |
| [architecture/pipeline_framework.md](architecture/pipeline_framework.md) | The pipeline framework of `unbihexium.core.pipeline`, the task pipelines, the `pipeline` commands, tiling and mosaicking |
| [architecture/security_model.md](architecture/security_model.md) | Trust boundaries, input validation, model loading and the protections of the REST service |

## 6. Capability domains

The twelve domain documents describe, for one application area each, the model families of the catalogue and the library functions that serve it, with executed examples. [capabilities/index.md](capabilities/index.md) explains how they map to the registry domains and the catalogue.

| Document | Covers |
| --- | --- |
| [capabilities/index.md](capabilities/index.md) | Overview of the twelve domains, their registry and catalogue domains and the counts of families and models |
| [capabilities/01_ai_products.md](capabilities/01_ai_products.md) | General-purpose detection, segmentation, change detection and translation models, and building, running, training and serving models |
| [capabilities/02_tourism_data_processing.md](capabilities/02_tourism_data_processing.md) | Tourism and analysis families; terrain, visibility, geostatistics, suitability, cost surfaces and network analysis |
| [capabilities/03_indices_flood_water.md](capabilities/03_indices_flood_water.md) | Spectral indices, the `unbihexium index` command, water and flood families and hydrological functions |
| [capabilities/04_environment_forestry_image_processing.md](capabilities/04_environment_forestry_image_processing.md) | Environment, forestry and imaging families; preprocessing, postprocessing and image enhancement |
| [capabilities/05_asset_management_energy.md](capabilities/05_asset_management_energy.md) | Asset and energy families; multi-criteria siting, terrain, least-cost routing and hydrology |
| [capabilities/06_urban_agriculture.md](capabilities/06_urban_agriculture.md) | Urban and agriculture families; radiometric scaling, masks, indices, zonal statistics and vectorisation |
| [capabilities/07_risk_defense_neutral.md](capabilities/07_risk_defense_neutral.md) | Risk and neutral monitoring families; burn severity, susceptibility mapping and change analysis |
| [capabilities/08_value_added_imagery.md](capabilities/08_value_added_imagery.md) | Elevation and 3D product families and the functions that derive products from elevation grids |
| [capabilities/09_benefits_narrative.md](capabilities/09_benefits_narrative.md) | The measurable, reportable outputs of the library and how to base statements of benefit on them |
| [capabilities/10_satellite_imagery_features.md](capabilities/10_satellite_imagery_features.md) | Sensor band tables, radiometric conversion, quality layers, pansharpening and enhancement of optical imagery |
| [capabilities/11_resolution_metadata_qa.md](capabilities/11_resolution_metadata_qa.md) | Raster metadata, ground sample distance, resampling, tiling, STAC metadata and quality checks |
| [capabilities/12_radar_sar.md](capabilities/12_radar_sar.md) | SAR calibration, speckle filtering, interferometry, polarimetry and the SAR model families |

## 7. Model zoo

| Document | Covers |
| --- | --- |
| [model_zoo/model_catalog.md](model_zoo/model_catalog.md) | Every family with task, domain, bands and outputs, model identifiers and variants, and links to the model cards |
| [model_zoo/download_and_verify.md](model_zoo/download_and_verify.md) | Building models into the local store and verifying their integrity with the `unbihexium zoo` commands |
| [model_zoo/inference.md](model_zoo/inference.md) | Model sources, PyTorch and ONNX Runtime backends, tiled inference, task APIs and result objects |
| [model_zoo/training.md](model_zoo/training.md) | Dataset layout, training and fine-tuning, synthetic data, and evaluation metrics |
| [model_zoo/how_to_add_models.md](model_zoo/how_to_add_models.md) | Adding a family to the catalogue and registering user models |
| [model_zoo/distribution.md](model_zoo/distribution.md) | How models reach users: local builds instead of weight files, offline and restricted environments |
| [model_zoo/licensing_and_provenance.md](model_zoo/licensing_and_provenance.md) | The licence of the catalogue, code and starter weights, and the provenance of trained models |

## 8. Security

| Document | Covers |
| --- | --- |
| [security/supply_chain_security.md](security/supply_chain_security.md) | Controls from source to artefact: pinning, lock files, audits, static analysis, fuzzing, signing, provenance and SBOMs |
| [security/model_integrity.md](security/model_integrity.md) | Starter weight digests, safe checkpoint loading, ONNX export checks and local checksum files |
| [security/vulnerability_management.md](security/vulnerability_management.md) | How vulnerabilities in the code and its dependencies are found, assessed, fixed and disclosed |
| [security/secrets_and_tokens.md](security/secrets_and_tokens.md) | Repository secrets, workflow token permissions, OIDC tokens and leak detection |
| [security/responsible_use.md](security/responsible_use.md) | Technical guidance for applying [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) with the features of the library |
| [security/self_assessment.md](security/self_assessment.md) | Security self-assessment after the CNCF TAG Security outline: actors, goals, non-goals, controls, development practices and issue resolution |

## 9. Operations

| Document | Covers |
| --- | --- |
| [operations/ci_cd.md](operations/ci_cd.md) | Every GitHub Actions workflow: triggers, jobs, checks and published artefacts |
| [operations/docker.md](operations/docker.md) | The container image, Docker Compose and the Helm and Kubernetes deployments of the REST service |
| [operations/releasing.md](operations/releasing.md) | The release procedure from version bump to signed, attested distributions |

## 10. Benchmarks

| Document | Covers |
| --- | --- |
| [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md) | The benchmark tests, what they measure and assert, how to run them, and the figures measured for this revision with their environment |

## 11. General documents

| Document | Covers |
| --- | --- |
| [faq.md](faq.md) | Answers to frequent questions about installation, data, models, the command line, the REST service and troubleshooting |
| [glossary.md](glossary.md) | Terms and abbreviations used in the code and the documentation, with the formulas of the spectral indices |
| [MIGRATION.md](MIGRATION.md) | Moving code, scripts, clients and deployments from 1.0.x to the main branch (the upcoming 2.0.0) |
| [toc.md](toc.md) | The complete table of contents of `docs/` |
| [document_register.md](document_register.md) | The numbering scheme of the document identifiers and the register of every controlled document, including those outside `docs/` |

## 12. Project documents outside docs/

### 12.1 Root documents

| Document | Covers |
| --- | --- |
| [README.md](../README.md) | Project overview, installation, quick start, features, model zoo, REST service and citation |
| [CHANGELOG.md](../CHANGELOG.md) | Changes of every release and of the unreleased main branch |
| [VERSIONING.md](../VERSIONING.md) | Version numbers, compatibility promises, deprecation, catalogue versions and Python support |
| [ROADMAP.md](../ROADMAP.md) | Planned work |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Development setup, standards, checks and the pull request process |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Expected behaviour in the project |
| [GOVERNANCE.md](../GOVERNANCE.md), [MAINTAINERS.md](../MAINTAINERS.md), [AUTHORS.md](../AUTHORS.md) | Decision making, maintainers and authors |
| [SUPPORT.md](../SUPPORT.md) | Where and how to ask for help |
| [SECURITY.md](../SECURITY.md) | Supported versions, private vulnerability reporting and release verification |
| [RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) | Limits of the starter models, intended and unsupported uses, dual-use considerations |
| [PRIVACY.md](../PRIVACY.md) | Data processed, network access and the absence of telemetry |
| [COMPLIANCE.md](../COMPLIANCE.md) | Licence and regulatory compliance information (not legal advice) |
| [CITATION.md](../CITATION.md), [CITATION.cff](../CITATION.cff) | How to cite the software |
| [NOTICE.md](../NOTICE.md), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), [LICENSE.txt](../LICENSE.txt) | Licence text and attribution notices |

### 12.2 Other README files

| Document | Covers |
| --- | --- |
| [model_zoo/README.md](../model_zoo/README.md) | The model zoo directory: model cards and manifests per family, inventory, family-to-model mapping, weights digests and the manifest schema |
| [model_zoo/MODEL_CARDS.md](../model_zoo/MODEL_CARDS.md) | Index of the 130 family model cards |
| [examples/README.md](../examples/README.md) | The example scripts and the example FastAPI application, with their test status and limitations |

## 13. Documentation conventions

### 13.1 Layout

Every document under `docs/` follows the same layout: a licence and header comment, one title, a document control table (identifier, version, status, review date, owner, scope; the identifiers are numbered as defined in the [Document Register](document_register.md)), an abstract, a list of contents, numbered sections, a numbered list of references for every external standard, publication or tool cited, and a footer comment. Documents are Markdown in CommonMark with the GitHub Flavored Markdown extensions [2], checked with markdownlint using [.markdownlint.yaml](../.markdownlint.yaml), with a link checker, and with the project text policy (`.github/scripts/check_text_policy.py`). Where a document states obligations, the key words MUST, SHOULD and MAY are used as defined in RFC 2119 [3] and RFC 8174 [4].

### 13.2 Accuracy

Documents describe the code of the main branch at their review date. Code examples and commands were executed against that code, and shown outputs are real outputs; the Documentation Examples workflow runs the Python examples and the `unbihexium` commands again on every change, so an example that the code no longer supports fails a check; numbers such as timings state how and where they were measured. Documents do not contain performance, accuracy or compliance claims that the repository cannot support.

### 13.3 Reporting problems

Errors in the documentation are reported as issues on <https://github.com/unbihexium-oss/unbihexium/issues> or fixed in a pull request following [CONTRIBUTING.md](../CONTRIBUTING.md). Security-relevant errors are reported privately as described in [SECURITY.md](../SECURITY.md).

## References

[1] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[2] GitHub. GitHub Flavored Markdown Spec, version 0.29-gfm. 2019. <https://github.github.com/gfm/>

[3] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[4] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/index.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
