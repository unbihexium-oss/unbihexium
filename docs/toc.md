<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/toc.md
Title       : Table of Contents
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Table of Contents

| Field | Value |
| --- | --- |
| Document | UBX-DOC-TOC |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Every document under `docs/` on the main branch of Unbihexium |

## Abstract

This document is the complete table of contents of the `docs/` directory of Unbihexium. It lists every document in the directory exactly once, with its document identifier, grouped in the same areas and order as the documentation hub [docs/index.md](index.md), so that readers, reviewers and maintainers can see at a glance what exists and check that nothing is missing. It does not describe the documents; their purpose and scope are summarised in the hub and stated in the abstract of each document. The list was compiled from `git ls-files docs` and contains 45 documents.

## Contents

1. [Maintenance of this list](#1-maintenance-of-this-list)
2. [Entry points](#2-entry-points)
3. [Getting started and tutorials](#3-getting-started-and-tutorials)
4. [Reference](#4-reference)
5. [Architecture](#5-architecture)
6. [Capability domains](#6-capability-domains)
7. [Model zoo](#7-model-zoo)
8. [Security](#8-security)
9. [Operations](#9-operations)
10. [Benchmarks](#10-benchmarks)
11. [General documents](#11-general-documents)
12. [Summary by directory](#12-summary-by-directory)
13. [References](#references)

## 1. Maintenance of this list

### 1.1 Conventions

The key words MUST and SHOULD in this section are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

### 1.2 Rules

- A pull request that adds, renames or removes a document under `docs/` MUST update this list and [docs/index.md](index.md) in the same pull request.
- The count in the abstract and in Section 12 SHOULD equal the output of `git ls-files docs | wc -l`.
- Link text is the title (first-level heading) of the document; the identifier is the value of the `Document` row of its control table.

## 2. Entry points

| Document | Identifier |
| --- | --- |
| [Unbihexium Documentation](index.md) | UBX-DOC-INDEX |
| [Table of Contents](toc.md) | UBX-DOC-TOC |

## 3. Getting started and tutorials

| Document | Identifier |
| --- | --- |
| [Installation](getting_started/installation.md) | UBX-DOC-GS-INSTALLATION |
| [Quick Start](getting_started/quickstart.md) | UBX-DOC-GS-QUICKSTART |
| [Configuration](getting_started/configuration.md) | UBX-DOC-GS-CONFIGURATION |
| [Tutorials](tutorials/index.md) | UBX-DOC-TUTORIALS |

## 4. Reference

| Document | Identifier |
| --- | --- |
| [Python API Reference](reference/api.md) | UBX-DOC-REF-API |
| [Command Line Reference](reference/cli.md) | UBX-DOC-REF-CLI |

## 5. Architecture

| Document | Identifier |
| --- | --- |
| [Architecture Overview](architecture/overview.md) | UBX-DOC-ARCH-OVERVIEW |
| [Capability Registry](architecture/capability_registry.md) | UBX-DOC-ARCH-CAPABILITY-REGISTRY |
| [Model Zoo Architecture](architecture/model_zoo_architecture.md) | UBX-DOC-ARCH-MODEL-ZOO |
| [Pipeline Framework](architecture/pipeline_framework.md) | UBX-DOC-ARCH-PIPELINE-FRAMEWORK |
| [Security Model](architecture/security_model.md) | UBX-DOC-ARCH-SECURITY-MODEL |

## 6. Capability domains

| Document | Identifier |
| --- | --- |
| [Capability domains](capabilities/index.md) | UBX-DOC-CAP-INDEX |
| [Capability domain 01: AI products](capabilities/01_ai_products.md) | UBX-DOC-CAP-01 |
| [Capability domain 02: tourism and data processing](capabilities/02_tourism_data_processing.md) | UBX-DOC-CAP-02 |
| [Capability domain 03: spectral indices, floods and water](capabilities/03_indices_flood_water.md) | UBX-DOC-CAP-03 |
| [Capability domain 04: environment, forestry and image processing](capabilities/04_environment_forestry_image_processing.md) | UBX-DOC-CAP-04 |
| [Capability Domain 05: Asset Management and Energy](capabilities/05_asset_management_energy.md) | UBX-DOC-CAP-05-ASSETS-ENERGY |
| [Capability Domain 06: Urban Planning and Agriculture](capabilities/06_urban_agriculture.md) | UBX-DOC-CAP-06-URBAN-AGRICULTURE |
| [Capability Domain 07: Risk Assessment and Neutral Monitoring](capabilities/07_risk_defense_neutral.md) | UBX-DOC-CAP-07-RISK-MONITORING |
| [Capability Domain 08: Value-Added Imagery, Elevation and 3D Products](capabilities/08_value_added_imagery.md) | UBX-DOC-CAP-08-VALUE-ADDED-IMAGERY |
| [Capability 09: Benefits Narrative and Reportable Outputs](capabilities/09_benefits_narrative.md) | UBX-DOC-CAP-09-BENEFITS |
| [Capability 10: Satellite Imagery Features](capabilities/10_satellite_imagery_features.md) | UBX-DOC-CAP-10-IMAGERY |
| [Capability 11: Resolution, Metadata and Quality Assurance](capabilities/11_resolution_metadata_qa.md) | UBX-DOC-CAP-11-RESOLUTION-QA |
| [Capability 12: Radar and Synthetic Aperture Radar](capabilities/12_radar_sar.md) | UBX-DOC-CAP-12-SAR |

## 7. Model zoo

| Document | Identifier |
| --- | --- |
| [Model Zoo Catalogue](model_zoo/model_catalog.md) | UBX-DOC-MZ-CATALOG |
| [Building and Verifying Models](model_zoo/download_and_verify.md) | UBX-DOC-MZ-BUILD-VERIFY |
| [Running Models](model_zoo/inference.md) | UBX-DOC-MZ-INFERENCE |
| [Training and Evaluating Models](model_zoo/training.md) | UBX-DOC-MZ-TRAINING |
| [Adding Models to the Model Zoo](model_zoo/how_to_add_models.md) | UBX-DOC-MZ-ADD-MODELS |
| [Model Distribution](model_zoo/distribution.md) | UBX-DOC-MZ-DISTRIBUTION |
| [Model Licensing and Provenance](model_zoo/licensing_and_provenance.md) | UBX-DOC-MZ-LICENSING |

## 8. Security

| Document | Identifier |
| --- | --- |
| [Supply Chain Security](security/supply_chain_security.md) | UBX-DOC-SEC-SUPPLY-CHAIN |
| [Model Integrity](security/model_integrity.md) | UBX-DOC-SEC-MODEL-INTEGRITY |
| [Vulnerability Management](security/vulnerability_management.md) | UBX-DOC-SEC-VULN-MGMT |
| [Secrets and Tokens](security/secrets_and_tokens.md) | UBX-DOC-SEC-SECRETS |
| [Responsible Use: Technical Guidance](security/responsible_use.md) | UBX-DOC-SEC-RESPONSIBLE-USE |

## 9. Operations

| Document | Identifier |
| --- | --- |
| [Continuous Integration and Delivery](operations/ci_cd.md) | UBX-DOC-OPS-CICD |
| [Docker Operations](operations/docker.md) | UBX-DOC-OPS-DOCKER |
| [Release Procedure](operations/releasing.md) | UBX-DOC-OPS-RELEASING |

## 10. Benchmarks

| Document | Identifier |
| --- | --- |
| [Benchmarks](benchmarks/BENCHMARKS.md) | UBX-DOC-BENCHMARKS |

## 11. General documents

| Document | Identifier |
| --- | --- |
| [Frequently Asked Questions](faq.md) | UBX-DOC-FAQ |
| [Glossary](glossary.md) | UBX-DOC-GLOSSARY |
| [Migration Guide from 1.0.x to 2.0.0](MIGRATION.md) | UBX-DOC-MIGRATION |

## 12. Summary by directory

| Directory | Documents |
| --- | --- |
| `docs/` | 5 (`index.md`, `toc.md`, `faq.md`, `glossary.md`, `MIGRATION.md`) |
| `docs/getting_started/` | 3 |
| `docs/tutorials/` | 1 |
| `docs/reference/` | 2 |
| `docs/architecture/` | 5 |
| `docs/capabilities/` | 13 |
| `docs/model_zoo/` | 7 |
| `docs/security/` | 5 |
| `docs/operations/` | 3 |
| `docs/benchmarks/` | 1 |
| **Total** | **45** |

Documents outside `docs/`, such as the policies in the repository root and the README files of `model_zoo/`, `examples/` and `tests/fixtures/`, are listed in [docs/index.md, Section 12](index.md#12-project-documents-outside-docs).

## References

[1] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/toc.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
