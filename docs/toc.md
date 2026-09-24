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
| Document | UBX-DOC-301 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Every document under `docs/` on the main branch of Unbihexium |

## Abstract

This document is the complete table of contents of the `docs/` directory of Unbihexium. It lists every document in the directory exactly once, with its document identifier, grouped in the same areas and order as the documentation hub [docs/index.md](index.md), so that readers, reviewers and maintainers can see at a glance what exists and check that nothing is missing. The identifiers follow the numbering scheme of the [Document Register](document_register.md), which also covers the documents outside `docs/`. It does not describe the documents; their purpose and scope are summarised in the hub and stated in the abstract of each document. The list was compiled from `git ls-files docs` and contains 46 documents.

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
- Link text is the title (first-level heading) of the document; the identifier is the value of the `Document` row of its control table, assigned under the rules of the [Document Register](document_register.md), Section 2.
- `python .github/scripts/check_document_ids.py` (`make doc-ids`, Markdown workflow) fails when an identifier or title in this list differs from the document.

## 2. Entry points

| Document | Identifier |
| --- | --- |
| [Unbihexium Documentation](index.md) | UBX-DOC-300 |
| [Table of Contents](toc.md) | UBX-DOC-301 |
| [Document Register](document_register.md) | UBX-DOC-302 |

## 3. Getting started and tutorials

| Document | Identifier |
| --- | --- |
| [Installation](getting_started/installation.md) | UBX-DOC-303 |
| [Quick Start](getting_started/quickstart.md) | UBX-DOC-304 |
| [Configuration](getting_started/configuration.md) | UBX-DOC-305 |
| [Tutorials](tutorials/index.md) | UBX-DOC-306 |

## 4. Reference

| Document | Identifier |
| --- | --- |
| [Python API Reference](reference/api.md) | UBX-DOC-401 |
| [Command Line Reference](reference/cli.md) | UBX-DOC-402 |

## 5. Architecture

| Document | Identifier |
| --- | --- |
| [Architecture Overview](architecture/overview.md) | UBX-DOC-501 |
| [Capability Registry](architecture/capability_registry.md) | UBX-DOC-502 |
| [Model Zoo Architecture](architecture/model_zoo_architecture.md) | UBX-DOC-503 |
| [Pipeline Framework](architecture/pipeline_framework.md) | UBX-DOC-504 |
| [Security Model](architecture/security_model.md) | UBX-DOC-505 |

## 6. Capability domains

| Document | Identifier |
| --- | --- |
| [Capability Domains](capabilities/index.md) | UBX-DOC-600 |
| [Capability Domain 01: AI Products](capabilities/01_ai_products.md) | UBX-DOC-601 |
| [Capability Domain 02: Tourism and Data Processing](capabilities/02_tourism_data_processing.md) | UBX-DOC-602 |
| [Capability Domain 03: Spectral Indices, Floods and Water](capabilities/03_indices_flood_water.md) | UBX-DOC-603 |
| [Capability Domain 04: Environment, Forestry and Image Processing](capabilities/04_environment_forestry_image_processing.md) | UBX-DOC-604 |
| [Capability Domain 05: Asset Management and Energy](capabilities/05_asset_management_energy.md) | UBX-DOC-605 |
| [Capability Domain 06: Urban Planning and Agriculture](capabilities/06_urban_agriculture.md) | UBX-DOC-606 |
| [Capability Domain 07: Risk Assessment and Neutral Monitoring](capabilities/07_risk_defense_neutral.md) | UBX-DOC-607 |
| [Capability Domain 08: Value-Added Imagery, Elevation and 3D Products](capabilities/08_value_added_imagery.md) | UBX-DOC-608 |
| [Capability Domain 09: Benefits Narrative and Reportable Outputs](capabilities/09_benefits_narrative.md) | UBX-DOC-609 |
| [Capability Domain 10: Satellite Imagery Features](capabilities/10_satellite_imagery_features.md) | UBX-DOC-610 |
| [Capability Domain 11: Resolution, Metadata and Quality Assurance](capabilities/11_resolution_metadata_qa.md) | UBX-DOC-611 |
| [Capability Domain 12: Radar and Synthetic Aperture Radar](capabilities/12_radar_sar.md) | UBX-DOC-612 |

## 7. Model zoo

| Document | Identifier |
| --- | --- |
| [Model Zoo Catalogue](model_zoo/model_catalog.md) | UBX-DOC-701 |
| [Building and Verifying Models](model_zoo/download_and_verify.md) | UBX-DOC-702 |
| [Running Models](model_zoo/inference.md) | UBX-DOC-703 |
| [Training and Evaluating Models](model_zoo/training.md) | UBX-DOC-704 |
| [Adding Models to the Model Zoo](model_zoo/how_to_add_models.md) | UBX-DOC-705 |
| [Model Distribution](model_zoo/distribution.md) | UBX-DOC-706 |
| [Model Licensing and Provenance](model_zoo/licensing_and_provenance.md) | UBX-DOC-707 |

## 8. Security

| Document | Identifier |
| --- | --- |
| [Supply Chain Security](security/supply_chain_security.md) | UBX-DOC-801 |
| [Model Integrity](security/model_integrity.md) | UBX-DOC-802 |
| [Vulnerability Management](security/vulnerability_management.md) | UBX-DOC-803 |
| [Secrets and Tokens](security/secrets_and_tokens.md) | UBX-DOC-804 |
| [Responsible Use: Technical Guidance](security/responsible_use.md) | UBX-DOC-805 |
| [Security Self-Assessment](security/self_assessment.md) | UBX-DOC-806 |

## 9. Operations

| Document | Identifier |
| --- | --- |
| [Continuous Integration and Delivery](operations/ci_cd.md) | UBX-DOC-901 |
| [Container Image and Deployment](operations/docker.md) | UBX-DOC-902 |
| [Release Procedure](operations/releasing.md) | UBX-DOC-903 |

## 10. Benchmarks

| Document | Identifier |
| --- | --- |
| [Benchmarks](benchmarks/BENCHMARKS.md) | UBX-DOC-904 |

## 11. General documents

| Document | Identifier |
| --- | --- |
| [Frequently Asked Questions](faq.md) | UBX-DOC-309 |
| [Glossary](glossary.md) | UBX-DOC-310 |
| [Migration Guide from 1.0.x to 2.0.0](MIGRATION.md) | UBX-DOC-308 |

## 12. Summary by directory

| Directory | Documents |
| --- | --- |
| `docs/` | 6 (`index.md`, `toc.md`, `document_register.md`, `faq.md`, `glossary.md`, `MIGRATION.md`) |
| `docs/getting_started/` | 3 |
| `docs/tutorials/` | 1 |
| `docs/reference/` | 2 |
| `docs/architecture/` | 5 |
| `docs/capabilities/` | 13 |
| `docs/model_zoo/` | 7 |
| `docs/security/` | 6 |
| `docs/operations/` | 3 |
| `docs/benchmarks/` | 1 |
| **Total** | **47** |

Documents outside `docs/`, such as the policies in the repository root and the README files of `model_zoo/` and `examples/`, are listed in [docs/index.md, Section 12](index.md#12-project-documents-outside-docs) and, with their identifiers, in the [Document Register](document_register.md).

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
