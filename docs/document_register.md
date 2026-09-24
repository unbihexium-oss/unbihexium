<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/document_register.md
Title       : Document Register
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Document Register

| Field | Value |
| --- | --- |
| Document | UBX-DOC-302 |
| Version | 1.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | Every controlled Markdown document of the Unbihexium repository |

## Abstract

This register defines how Unbihexium identifies its documents and lists every controlled document with its identifier. An identifier has the form `UBX-DOC-SNN`: a fixed prefix, one series digit for the area of the document and a two-digit sequence number within the series, for example `UBX-DOC-801` for the first security document. Identifiers are permanent: they do not change when a document is edited, retitled, renamed or moved, and the number of a withdrawn document is never assigned again. The register contains 65 documents in nine series. `.github/scripts/check_document_ids.py` compares it with the documents on every change to a Markdown file.

## Contents

1. [Scope and conventions](#1-scope-and-conventions)
2. [Identifier scheme](#2-identifier-scheme)
3. [Register](#3-register)
4. [Withdrawn identifiers](#4-withdrawn-identifiers)
5. [Previous identifiers](#5-previous-identifiers)
6. [Verification](#6-verification)
7. [References](#references)

## 1. Scope and conventions

### 1.1 Scope

Controlled documents are the Markdown documents that carry a document control table: the documents in the repository root, every document under `docs/`, and the README files of `examples/` and `model_zoo/`. Two groups of Markdown files are not controlled and have no identifier:

- the files under `.github/`, such as the pull request template, which GitHub shows as forms or instructions;
- the model cards under `model_zoo/cards/` and their index `model_zoo/MODEL_CARDS.md`, which `python -m unbihexium.zoo.sync` generates from the model catalogue.

### 1.2 Conventions

The key words MUST and MUST NOT in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

## 2. Identifier scheme

### 2.1 Structure

An identifier consists of four parts, shown here for `UBX-DOC-603`:

| Part | Value in the example | Meaning |
| --- | --- | --- |
| Project code | `UBX` | Unbihexium |
| Document class | `DOC` | Controlled documentation |
| Series digit `S` | `6` | Area of the document, from 1 to 9 (Section 2.2) |
| Sequence number `NN` | `03` | Position of the document within its series, from 00 to 99 |

The identifier names a document, not a revision of it. As in the data fields for document headers of ISO 7200 [3], the identification number and the revision are separate fields of the control table: the identifier is in the `Document` row, the revision in the `Version` row (Section 2.4).

### 2.2 Series

| Series | Range | Area | Content | Overview document |
| --- | --- | --- | --- | --- |
| 1 | UBX-DOC-100 to UBX-DOC-199 | Project | Description of the project, its history, plans, citation, people, governance, contribution and support; the root documents that are not policies. | UBX-DOC-100 (README.md) |
| 2 | UBX-DOC-200 to UBX-DOC-299 | Policies and legal notices | Security, privacy, conduct, responsible use, compliance and the licence and third-party notices. | None |
| 3 | UBX-DOC-300 to UBX-DOC-399 | Guides and general documentation | Documentation hub, lists of documents, getting started, tutorials, examples, migration, frequently asked questions and glossary. | UBX-DOC-300 (docs/index.md) |
| 4 | UBX-DOC-400 to UBX-DOC-499 | Reference | Python API and command line reference. | None |
| 5 | UBX-DOC-500 to UBX-DOC-599 | Architecture | Design of the package, its registries, the model zoo, the pipelines and the security model. | None |
| 6 | UBX-DOC-600 to UBX-DOC-699 | Capability domains | The capability domains; UBX-DOC-6NN is domain NN. | UBX-DOC-600 (docs/capabilities/index.md) |
| 7 | UBX-DOC-700 to UBX-DOC-799 | Model zoo | Catalogue, building, verification, inference, training, distribution and licensing of the models. | UBX-DOC-700 (model_zoo/README.md) |
| 8 | UBX-DOC-800 to UBX-DOC-899 | Security | Technical security documents; the security policy itself is in series 2. | None |
| 9 | UBX-DOC-900 to UBX-DOC-999 | Operations and reports | Continuous integration and delivery, the container image, releases and measurement reports. | None |

Series 0 is not used, so that every identifier has three significant digits. Each series holds up to 100 documents.

### 2.3 Assignment rules

- An identifier MUST consist of `UBX-DOC-`, a series digit from 1 to 9 and two further digits, and MUST be unique among the controlled documents. A document MUST have exactly one identifier.
- The sequence number 00 of a series is reserved for the overview document of that series. A series without an overview document starts at 01.
- In series 6 the sequence number MUST equal the number of the capability domain, so that `UBX-DOC-6NN` always identifies domain NN.
- A new document takes the next free number of the series of its area. The pull request that adds it MUST add it to Section 3 and, for a document under `docs/`, to the [Table of Contents](toc.md) (UBX-DOC-301) and the [documentation hub](index.md) (UBX-DOC-300).
- An identifier is permanent. Editing, retitling, renaming or moving a document MUST NOT change its identifier. The series records the area at the time of assignment and is not changed later.
- When a document is removed, its identifier is withdrawn: the pull request that removes the document MUST move the entry from Section 3 to Section 4, with the date and the reason. A withdrawn identifier MUST NOT be assigned again.

### 2.4 Identifiers and versions

The `Version` row of the control table gives the revision of a document as MAJOR.MINOR, the `Last reviewed` row the date of the last review, and the Git history every individual change. A reference to a particular revision gives both values, for example "UBX-DOC-801, version 2.0"; a reference without a version means the current revision on the main branch.

## 3. Register

Each entry gives the identifier, the title (first-level heading) with a link to the document, and the file in the repository. Entries are ordered by identifier.

### 3.1 Series 1: Project

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-100 | [Unbihexium](../README.md) | `README.md` |
| UBX-DOC-101 | [Changelog](../CHANGELOG.md) | `CHANGELOG.md` |
| UBX-DOC-102 | [Versioning and Release Policy](../VERSIONING.md) | `VERSIONING.md` |
| UBX-DOC-103 | [Roadmap](../ROADMAP.md) | `ROADMAP.md` |
| UBX-DOC-104 | [Citing Unbihexium](../CITATION.md) | `CITATION.md` |
| UBX-DOC-105 | [Authors and Contributors](../AUTHORS.md) | `AUTHORS.md` |
| UBX-DOC-106 | [Maintainers](../MAINTAINERS.md) | `MAINTAINERS.md` |
| UBX-DOC-107 | [Project Governance](../GOVERNANCE.md) | `GOVERNANCE.md` |
| UBX-DOC-108 | [Contributing to Unbihexium](../CONTRIBUTING.md) | `CONTRIBUTING.md` |
| UBX-DOC-109 | [Support Policy](../SUPPORT.md) | `SUPPORT.md` |

### 3.2 Series 2: Policies and legal notices

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-201 | [Security Policy](../SECURITY.md) | `SECURITY.md` |
| UBX-DOC-202 | [Privacy Statement](../PRIVACY.md) | `PRIVACY.md` |
| UBX-DOC-203 | [Code of Conduct](../CODE_OF_CONDUCT.md) | `CODE_OF_CONDUCT.md` |
| UBX-DOC-204 | [Responsible Use Policy](../RESPONSIBLE_USE.md) | `RESPONSIBLE_USE.md` |
| UBX-DOC-205 | [Licence and Regulatory Compliance](../COMPLIANCE.md) | `COMPLIANCE.md` |
| UBX-DOC-206 | [Notices](../NOTICE.md) | `NOTICE.md` |
| UBX-DOC-207 | [Third-Party Notices](../THIRD_PARTY_NOTICES.md) | `THIRD_PARTY_NOTICES.md` |

### 3.3 Series 3: Guides and general documentation

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-300 | [Unbihexium Documentation](index.md) | `docs/index.md` |
| UBX-DOC-301 | [Table of Contents](toc.md) | `docs/toc.md` |
| UBX-DOC-302 | [Document Register](document_register.md) | `docs/document_register.md` |
| UBX-DOC-303 | [Installation](getting_started/installation.md) | `docs/getting_started/installation.md` |
| UBX-DOC-304 | [Quick Start](getting_started/quickstart.md) | `docs/getting_started/quickstart.md` |
| UBX-DOC-305 | [Configuration](getting_started/configuration.md) | `docs/getting_started/configuration.md` |
| UBX-DOC-306 | [Tutorials](tutorials/index.md) | `docs/tutorials/index.md` |
| UBX-DOC-307 | [Examples](../examples/README.md) | `examples/README.md` |
| UBX-DOC-308 | [Migration Guide from 1.0.x to 2.0.0](MIGRATION.md) | `docs/MIGRATION.md` |
| UBX-DOC-309 | [Frequently Asked Questions](faq.md) | `docs/faq.md` |
| UBX-DOC-310 | [Glossary](glossary.md) | `docs/glossary.md` |

### 3.4 Series 4: Reference

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-401 | [Python API Reference](reference/api.md) | `docs/reference/api.md` |
| UBX-DOC-402 | [Command Line Reference](reference/cli.md) | `docs/reference/cli.md` |

### 3.5 Series 5: Architecture

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-501 | [Architecture Overview](architecture/overview.md) | `docs/architecture/overview.md` |
| UBX-DOC-502 | [Capability Registry](architecture/capability_registry.md) | `docs/architecture/capability_registry.md` |
| UBX-DOC-503 | [Model Zoo Architecture](architecture/model_zoo_architecture.md) | `docs/architecture/model_zoo_architecture.md` |
| UBX-DOC-504 | [Pipeline Framework](architecture/pipeline_framework.md) | `docs/architecture/pipeline_framework.md` |
| UBX-DOC-505 | [Security Model](architecture/security_model.md) | `docs/architecture/security_model.md` |

### 3.6 Series 6: Capability domains

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-600 | [Capability Domains](capabilities/index.md) | `docs/capabilities/index.md` |
| UBX-DOC-601 | [Capability Domain 01: AI Products](capabilities/01_ai_products.md) | `docs/capabilities/01_ai_products.md` |
| UBX-DOC-602 | [Capability Domain 02: Tourism and Data Processing](capabilities/02_tourism_data_processing.md) | `docs/capabilities/02_tourism_data_processing.md` |
| UBX-DOC-603 | [Capability Domain 03: Spectral Indices, Floods and Water](capabilities/03_indices_flood_water.md) | `docs/capabilities/03_indices_flood_water.md` |
| UBX-DOC-604 | [Capability Domain 04: Environment, Forestry and Image Processing](capabilities/04_environment_forestry_image_processing.md) | `docs/capabilities/04_environment_forestry_image_processing.md` |
| UBX-DOC-605 | [Capability Domain 05: Asset Management and Energy](capabilities/05_asset_management_energy.md) | `docs/capabilities/05_asset_management_energy.md` |
| UBX-DOC-606 | [Capability Domain 06: Urban Planning and Agriculture](capabilities/06_urban_agriculture.md) | `docs/capabilities/06_urban_agriculture.md` |
| UBX-DOC-607 | [Capability Domain 07: Risk Assessment and Neutral Monitoring](capabilities/07_risk_defense_neutral.md) | `docs/capabilities/07_risk_defense_neutral.md` |
| UBX-DOC-608 | [Capability Domain 08: Value-Added Imagery, Elevation and 3D Products](capabilities/08_value_added_imagery.md) | `docs/capabilities/08_value_added_imagery.md` |
| UBX-DOC-609 | [Capability Domain 09: Benefits Narrative and Reportable Outputs](capabilities/09_benefits_narrative.md) | `docs/capabilities/09_benefits_narrative.md` |
| UBX-DOC-610 | [Capability Domain 10: Satellite Imagery Features](capabilities/10_satellite_imagery_features.md) | `docs/capabilities/10_satellite_imagery_features.md` |
| UBX-DOC-611 | [Capability Domain 11: Resolution, Metadata and Quality Assurance](capabilities/11_resolution_metadata_qa.md) | `docs/capabilities/11_resolution_metadata_qa.md` |
| UBX-DOC-612 | [Capability Domain 12: Radar and Synthetic Aperture Radar](capabilities/12_radar_sar.md) | `docs/capabilities/12_radar_sar.md` |

### 3.7 Series 7: Model zoo

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-700 | [Model Zoo Metadata](../model_zoo/README.md) | `model_zoo/README.md` |
| UBX-DOC-701 | [Model Zoo Catalogue](model_zoo/model_catalog.md) | `docs/model_zoo/model_catalog.md` |
| UBX-DOC-702 | [Building and Verifying Models](model_zoo/download_and_verify.md) | `docs/model_zoo/download_and_verify.md` |
| UBX-DOC-703 | [Running Models](model_zoo/inference.md) | `docs/model_zoo/inference.md` |
| UBX-DOC-704 | [Training and Evaluating Models](model_zoo/training.md) | `docs/model_zoo/training.md` |
| UBX-DOC-705 | [Adding Models to the Model Zoo](model_zoo/how_to_add_models.md) | `docs/model_zoo/how_to_add_models.md` |
| UBX-DOC-706 | [Model Distribution](model_zoo/distribution.md) | `docs/model_zoo/distribution.md` |
| UBX-DOC-707 | [Model Licensing and Provenance](model_zoo/licensing_and_provenance.md) | `docs/model_zoo/licensing_and_provenance.md` |

### 3.8 Series 8: Security

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-801 | [Supply Chain Security](security/supply_chain_security.md) | `docs/security/supply_chain_security.md` |
| UBX-DOC-802 | [Model Integrity](security/model_integrity.md) | `docs/security/model_integrity.md` |
| UBX-DOC-803 | [Vulnerability Management](security/vulnerability_management.md) | `docs/security/vulnerability_management.md` |
| UBX-DOC-804 | [Secrets and Tokens](security/secrets_and_tokens.md) | `docs/security/secrets_and_tokens.md` |
| UBX-DOC-805 | [Responsible Use: Technical Guidance](security/responsible_use.md) | `docs/security/responsible_use.md` |

### 3.9 Series 9: Operations and reports

| Identifier | Document | File |
| --- | --- | --- |
| UBX-DOC-901 | [Continuous Integration and Delivery](operations/ci_cd.md) | `docs/operations/ci_cd.md` |
| UBX-DOC-902 | [Container Image and Deployment](operations/docker.md) | `docs/operations/docker.md` |
| UBX-DOC-903 | [Release Procedure](operations/releasing.md) | `docs/operations/releasing.md` |
| UBX-DOC-904 | [Benchmarks](benchmarks/BENCHMARKS.md) | `docs/benchmarks/BENCHMARKS.md` |

## 4. Withdrawn identifiers

No identifier has been withdrawn. A withdrawn identifier is listed here in a table with the columns Identifier, Date, Former document and Reason, and remains reserved.

## 5. Previous identifiers

Until 2026-09-24 the documents carried descriptive identifiers of the form `UBX-DOC-<NAME>`, for example `UBX-DOC-SEC-SUPPLY-CHAIN`. Their structure was not uniform and they did not show the area of a document in a fixed position, so they were replaced by the numbered identifiers of Section 2 on that date. The previous identifiers MUST NOT be used for new references; references in earlier revisions resolve as follows.

| Previous identifier | Identifier |
| --- | --- |
| UBX-DOC-README | UBX-DOC-100 |
| UBX-DOC-CHANGELOG | UBX-DOC-101 |
| UBX-DOC-VERSIONING | UBX-DOC-102 |
| UBX-DOC-ROADMAP | UBX-DOC-103 |
| UBX-DOC-CITATION | UBX-DOC-104 |
| UBX-DOC-AUTHORS | UBX-DOC-105 |
| UBX-DOC-MAINTAINERS | UBX-DOC-106 |
| UBX-DOC-GOVERNANCE | UBX-DOC-107 |
| UBX-DOC-CONTRIBUTING | UBX-DOC-108 |
| UBX-DOC-SUPPORT | UBX-DOC-109 |
| UBX-DOC-SECURITY | UBX-DOC-201 |
| UBX-DOC-PRIVACY | UBX-DOC-202 |
| UBX-DOC-CODE-OF-CONDUCT | UBX-DOC-203 |
| UBX-DOC-RESPONSIBLE-USE | UBX-DOC-204 |
| UBX-DOC-COMPLIANCE | UBX-DOC-205 |
| UBX-DOC-NOTICE | UBX-DOC-206 |
| UBX-DOC-THIRD-PARTY | UBX-DOC-207 |
| UBX-DOC-INDEX | UBX-DOC-300 |
| UBX-DOC-TOC | UBX-DOC-301 |
| UBX-DOC-GS-INSTALLATION | UBX-DOC-303 |
| UBX-DOC-GS-QUICKSTART | UBX-DOC-304 |
| UBX-DOC-GS-CONFIGURATION | UBX-DOC-305 |
| UBX-DOC-TUTORIALS | UBX-DOC-306 |
| UBX-DOC-EXAMPLES | UBX-DOC-307 |
| UBX-DOC-MIGRATION | UBX-DOC-308 |
| UBX-DOC-FAQ | UBX-DOC-309 |
| UBX-DOC-GLOSSARY | UBX-DOC-310 |
| UBX-DOC-REF-API | UBX-DOC-401 |
| UBX-DOC-REF-CLI | UBX-DOC-402 |
| UBX-DOC-ARCH-OVERVIEW | UBX-DOC-501 |
| UBX-DOC-ARCH-CAPABILITY-REGISTRY | UBX-DOC-502 |
| UBX-DOC-ARCH-MODEL-ZOO | UBX-DOC-503 |
| UBX-DOC-ARCH-PIPELINE-FRAMEWORK | UBX-DOC-504 |
| UBX-DOC-ARCH-SECURITY-MODEL | UBX-DOC-505 |
| UBX-DOC-CAP-INDEX | UBX-DOC-600 |
| UBX-DOC-CAP-01 | UBX-DOC-601 |
| UBX-DOC-CAP-02 | UBX-DOC-602 |
| UBX-DOC-CAP-03 | UBX-DOC-603 |
| UBX-DOC-CAP-04 | UBX-DOC-604 |
| UBX-DOC-CAP-05-ASSETS-ENERGY | UBX-DOC-605 |
| UBX-DOC-CAP-06-URBAN-AGRICULTURE | UBX-DOC-606 |
| UBX-DOC-CAP-07-RISK-MONITORING | UBX-DOC-607 |
| UBX-DOC-CAP-08-VALUE-ADDED-IMAGERY | UBX-DOC-608 |
| UBX-DOC-CAP-09-BENEFITS | UBX-DOC-609 |
| UBX-DOC-CAP-10-IMAGERY | UBX-DOC-610 |
| UBX-DOC-CAP-11-RESOLUTION-QA | UBX-DOC-611 |
| UBX-DOC-CAP-12-SAR | UBX-DOC-612 |
| UBX-DOC-MZ-README | UBX-DOC-700 |
| UBX-DOC-MZ-CATALOG | UBX-DOC-701 |
| UBX-DOC-MZ-BUILD-VERIFY | UBX-DOC-702 |
| UBX-DOC-MZ-INFERENCE | UBX-DOC-703 |
| UBX-DOC-MZ-TRAINING | UBX-DOC-704 |
| UBX-DOC-MZ-ADD-MODELS | UBX-DOC-705 |
| UBX-DOC-MZ-DISTRIBUTION | UBX-DOC-706 |
| UBX-DOC-MZ-LICENSING | UBX-DOC-707 |
| UBX-DOC-SEC-SUPPLY-CHAIN | UBX-DOC-801 |
| UBX-DOC-SEC-MODEL-INTEGRITY | UBX-DOC-802 |
| UBX-DOC-SEC-VULN-MGMT | UBX-DOC-803 |
| UBX-DOC-SEC-SECRETS | UBX-DOC-804 |
| UBX-DOC-SEC-RESPONSIBLE-USE | UBX-DOC-805 |
| UBX-DOC-OPS-CICD | UBX-DOC-901 |
| UBX-DOC-OPS-DOCKER | UBX-DOC-902 |
| UBX-DOC-OPS-RELEASING | UBX-DOC-903 |
| UBX-DOC-BENCHMARKS | UBX-DOC-904 |

The Document Register (UBX-DOC-302) was created with the numbered scheme and has no previous identifier.

## 6. Verification

`.github/scripts/check_document_ids.py` checks the rules of this document and fails with an annotation for every problem it finds:

- every tracked Markdown file outside the exclusions of Section 1.1 has a document control table whose `Document` row holds an identifier in the format of Section 2.3;
- the identifiers are unique and none of them is withdrawn;
- Section 3 lists exactly these documents, each under the series of its identifier, with the title and file of the document;
- the `Title` field of the header comment of each document equals its first-level heading;
- the identifiers and titles in the [Table of Contents](toc.md) equal those of the documents.

The check runs in the Markdown workflow (job "Document identifiers") when a Markdown file changes, as the pre-commit hook `document-ids` when a Markdown file is committed, and locally with `make doc-ids`, which is part of `make check`.

## References

[1] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] International Organization for Standardization. ISO 7200:2004, Technical product documentation: Data fields in title blocks and document headers. ISO. 2004. <https://www.iso.org/standard/35446.html>

<!--
=============================================================================
End of file docs/document_register.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
