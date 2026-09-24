<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : COMPLIANCE.md
Title       : Licence and Regulatory Compliance
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Licence and Regulatory Compliance

| Field | Value |
| --- | --- |
| Document | UBX-DOC-205 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This document is written for users, redistributors, auditors and contributors who need to know how Unbihexium handles licensing and which regulations may be relevant when the software is used. It describes the project licence and the obligations it creates for redistribution, the licence metadata and the automated checks that keep every file and every dependency within the project policy, the licensing of the model zoo and of the data it processes, and informational notes on export control and on European Union regulation. The project has not obtained any certification, formal legal review or export classification, and the regulatory notes are general information about the texts cited, not an assessment of any particular use. This document is not legal advice; consult a qualified adviser for decisions that depend on it.

## Contents

- [1. Introduction](#1-introduction)
- [2. Project Licence](#2-project-licence)
- [3. Licence Metadata and REUSE Compliance](#3-licence-metadata-and-reuse-compliance)
- [4. Dependency Licence Policy](#4-dependency-licence-policy)
- [5. Model Zoo and Data Licences](#5-model-zoo-and-data-licences)
- [6. Export Control and Sanctions](#6-export-control-and-sanctions)
- [7. European Union Regulation](#7-european-union-regulation)
- [8. Status of Compliance Activities](#8-status-of-compliance-activities)
- [9. Raising a Compliance Concern](#9-raising-a-compliance-concern)
- [References](#references)

## 1. Introduction

### 1.1 Purpose and Status

Sections 2 to 5 describe what the project does to comply with its own licence and with the licences of its dependencies; these statements are facts about the repository and can be checked with the commands given. Sections 6 and 7 summarise legal texts that are often relevant to Earth observation software. They are provided as information only. They do not state that any use is lawful, they may be incomplete or out of date, and they are not legal advice.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. They describe the project's policy for its repository and releases, and what the licence requires of redistributors. In Sections 6 and 7 they are not used, because those sections describe external law rather than project policy.

## 2. Project Licence

### 2.1 Licence

Unbihexium is licensed under the Mozilla Public License, version 2.0 [3] (SPDX identifier `MPL-2.0`). The full text is in [LICENSE.txt](LICENSE.txt). The licence is declared as a PEP 639 licence expression in [pyproject.toml](pyproject.toml) (`license = "MPL-2.0"`), and [LICENSE.txt](LICENSE.txt), [NOTICE](NOTICE) and [NOTICE.md](NOTICE.md) are included as licence files in the metadata of every distribution.

The MPL-2.0 is a file-level copyleft licence. It permits use, modification and distribution for any purpose, including commercial use, and it allows MPL-covered files to be combined with files under other licences in a larger work. It includes an express patent licence from each contributor and grants no trademark rights.

### 2.2 Obligations of Redistributors

A party that distributes Unbihexium, modified or unmodified, in source or executable form (for example in a wheel, a container image or an application bundle):

- MUST keep the licence notices and the copyright notices in the files, and make the MPL-2.0 text available to recipients (Section 3.2 of the licence);
- MUST make the source code of the MPL-covered files, including any modifications to them, available to recipients of an executable form, and inform them how to obtain it (Section 3.2);
- MUST distribute modified MPL-covered files under the MPL-2.0 (Section 3.1);
- MAY distribute a larger work that combines Unbihexium with other code under different terms, provided the MPL-covered files remain under the MPL-2.0 (Section 3.3).

Each source file carries the Exhibit A notice of the MPL-2.0 at its top. Code that is copied into another project keeps that notice.

### 2.3 Contributions

Contributions are licensed under the MPL-2.0, the licence of the files they change (inbound licence equals outbound licence); the contribution process is described in [CONTRIBUTING.md](CONTRIBUTING.md). New files MUST carry the project header with the MPL-2.0 notice; the checks in Section 3.3 enforce this. Contributors MUST NOT submit code or data that they are not entitled to license under the MPL-2.0.

## 3. Licence Metadata and REUSE Compliance

### 3.1 REUSE

The repository follows the REUSE Specification, version 3.3 [4], which requires copyright and licence information for every file. [REUSE.toml](REUSE.toml) assigns the copyright holder and the licence MPL-2.0 to all files and records the package name, supplier and download location for SPDX documents. Because the repository keeps a single licence text in [LICENSE.txt](LICENSE.txt), the CI job copies it to `LICENSES/MPL-2.0.txt` (the location REUSE expects, ignored by Git) before running `reuse lint`.

### 3.2 Distribution Contents

The source distribution contains the package sources, the model zoo metadata, the tests, [LICENSE.txt](LICENSE.txt), [NOTICE](NOTICE), [NOTICE.md](NOTICE.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [REUSE.toml](REUSE.toml), [README.md](README.md), [CHANGELOG.md](CHANGELOG.md) and [CITATION.cff](CITATION.cff), as configured in [pyproject.toml](pyproject.toml). The wheel contains the package and the licence files in its metadata. The container image records `MPL-2.0` in its `org.opencontainers.image.licenses` label.

### 3.3 Automated Checks

The License Compliance workflow ([.github/workflows/license-compliance.yml](.github/workflows/license-compliance.yml)) runs on pushes to main, on pull requests, weekly on Monday at 04:00 UTC and on demand. It has three jobs:

| Job | Tool | Fails when |
| --- | --- | --- |
| MPL-2.0 notices | [.github/scripts/check_license_headers.py](.github/scripts/check_license_headers.py) | [LICENSE.txt](LICENSE.txt) is not the MPL-2.0 text, or a tracked Python or shell file lacks the Exhibit A notice in its first five lines |
| REUSE compliance | `reuse lint` | a file lacks copyright or licence information |
| Dependency licences | `pip-licenses` and [.github/scripts/check_dependency_licenses.py](.github/scripts/check_dependency_licenses.py) | an installed runtime dependency has a licence denied by the policy in Section 4 |

The same checks can be run locally from the repository root, with `pip-licenses` and `reuse` installed from the hashed lock [.github/requirements/requirements-ci-tools.txt](.github/requirements/requirements-ci-tools.txt):

```bash
python .github/scripts/check_license_headers.py
mkdir -p LICENSES && cp LICENSE.txt LICENSES/MPL-2.0.txt && reuse lint
pip-licenses --from=mixed --format=json --with-urls --output-file licenses.json
python .github/scripts/check_dependency_licenses.py licenses.json
```

Run `pip-licenses` in an environment that contains only Unbihexium and its runtime dependencies (for example a virtual environment created from [requirements.txt](requirements.txt)); in a development environment it also lists the development tools.

## 4. Dependency Licence Policy

### 4.1 Rationale

Unbihexium is distributed under a file-level copyleft licence and is meant to be usable in proprietary and open source applications alike. Dependencies MUST therefore not impose obligations on the whole distribution that go beyond those of the MPL-2.0, and MUST NOT restrict commercial or production use.

### 4.2 Installed Dependencies

[.github/scripts/check_dependency_licenses.py](.github/scripts/check_dependency_licenses.py) reads the JSON report of `pip-licenses` and applies these rules:

- **Denied:** licence names that match GPL (but not LGPL), "GNU General Public License", Affero (AGPL), SSPL or "Server Side Public" and the Commons Clause, compared without regard to case.
- **Accepted:** every other licence, including LGPL, MPL, Apache, BSD, MIT, ISC, PSF and similar permissive licences.
- **Dual and multiple licences:** a package is denied only when every alternative is denied; `MIT OR GPL-2.0` is accepted. Alternatives are separated by `OR` or by the semicolons that `pip-licenses` uses between classifier licences.
- **Missing metadata:** a package without licence metadata is reported as `UNKNOWN` in the job summary and is not denied automatically; such entries are reviewed by hand.
- The project itself is excluded from the check.

The script writes a table of all packages, versions and licences to the GitHub Actions job summary and exits with status 1 when a package is denied.

### 4.3 New Dependencies in Pull Requests

The Dependency Review job of [.github/workflows/security.yml](.github/workflows/security.yml) compares the dependency manifests of a pull request with its base. It fails when a new dependency has a high or critical known vulnerability, or when its SPDX licence is on the deny list of [.github/dependency-review-config.yml](.github/dependency-review-config.yml): AGPL-1.0 and AGPL-3.0, GPL-1.0, GPL-2.0 and GPL-3.0 (each in the `-only` and `-or-later` forms), SSPL-1.0, BUSL-1.1, CC-BY-NC-4.0, CC-BY-NC-SA-4.0 and CC-BY-NC-ND-4.0.

### 4.4 Optional Dependencies

Optional extras (`torch`, `zarr` and `parquet`) are not installed by the Dependency licences job, which audits the locked runtime set in [requirements.txt](requirements.txt), including the `onnx` and `serving` extras. Users who install other extras, or the CUDA builds of PyTorch with their vendor libraries, SHOULD review the licences of what they install. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the main third-party components.

## 5. Model Zoo and Data Licences

### 5.1 Model Zoo

The model zoo contains 520 models (130 families in 4 variants: tiny, base, large and mega). Each catalogue entry declares the licence `MPL-2.0`. The models are untrained starter models: their weights are generated locally and deterministically from the model identifier by code in this repository, so no third-party weights or training data are embedded in them. The exception is the 7 spectral index families (28 models), which compute exact published formulas and have no trainable weights. The published digests in [model_zoo/checksums.txt](model_zoo/checksums.txt) identify the starter weights.

Users who train or fine-tune a model create new weights. The licence and the conditions of the training data and of any pretrained backbone they use apply to those weights; the project makes no statement about them. Model cards in [model_zoo/cards/](model_zoo/cards/) SHOULD be updated with the provenance and licence of the training data when trained models are shared.

### 5.2 Earth Observation Data

Unbihexium does not ship imagery. Users are responsible for complying with the licence of the data they process. For example, Copernicus Sentinel data are available free of charge under the terms of Commission Delegated Regulation (EU) No 1159/2013 [8], which require users who communicate or distribute the data or derived products to state the source, for example "Contains modified Copernicus Sentinel data [year]". Commercial imagery is usually licensed under terms that restrict redistribution and derived products.

## 6. Export Control and Sanctions

This section is information, not legal advice. The maintainers have not obtained a formal export control classification of Unbihexium.

- **European Union.** Regulation (EU) 2021/821 [6] controls the export of dual-use items listed in its Annex I. The General Software Note of Annex I generally excludes software that is "in the public domain", which includes software made available without restrictions on its further dissemination. Unbihexium is published openly on GitHub and PyPI under the MPL-2.0.
- **United States.** Under the Export Administration Regulations [7], published software that is publicly available without restriction is generally not subject to the EAR (15 CFR 734.3(b)(3) and 734.7). Specific rules apply to encryption items. Unbihexium does not implement cryptographic algorithms of its own: it uses SHA-256 digests and constant-time comparison from the Python standard library for integrity checks and API key comparison, and relies on third-party libraries for TLS.
- **Geospatial analysis.** Some jurisdictions have at times introduced controls specific to software that automates the analysis of geospatial imagery. Users are advised to check the current control lists of the countries involved rather than rely on the general exclusions above.
- **Sanctions.** Export control exclusions for public software do not lift sanctions. Restrictive measures, such as Council Regulation (EU) No 833/2014 [9] concerning Russia, can prohibit providing services, technical assistance or software to listed persons, entities or regions. Users are responsible for complying with the sanctions that apply to them.

The model catalogue includes families in the defence domain (for example `military_objects_detector`, `maritime_awareness` and `border_monitor`), shipped as untrained starter architectures. Their dual-use character and the uses the project does not support are described in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## 7. European Union Regulation

This section is information, not legal advice. Application dates and the scope of implementing acts can change; check the current consolidated texts.

### 7.1 Artificial Intelligence Act

Regulation (EU) 2024/1689 [5] (the AI Act) sets rules for AI systems placed on the market or put into service in the Union. Points relevant to Unbihexium:

- Article 2(12) excludes AI systems released under free and open-source licences, unless they are placed on the market or put into service as high-risk AI systems or fall under Article 5 (prohibited practices) or Article 50 (transparency obligations).
- Article 2(3) excludes AI systems used exclusively for military, defence or national security purposes.
- The project publishes a library and untrained starter models; it does not place an AI system for a specific intended purpose on the market. An organisation that builds a system with Unbihexium and places it on the market or puts it into service determines its intended purpose, and therefore its classification (for example under Article 6 and Annex III) and its obligations as provider or deployer.

[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) describes the practices the project does not support, including those prohibited by Article 5.

### 7.2 Cyber Resilience Act

Regulation (EU) 2024/2847 [10] (the Cyber Resilience Act) sets cybersecurity requirements for products with digital elements made available on the market in the course of a commercial activity. Free and open-source software that is developed and supplied outside a commercial activity is not within its scope as a product. Manufacturers that integrate Unbihexium into their products must exercise due diligence over the components they integrate and, under Article 13(6), report vulnerabilities they identify in a component to the person or entity maintaining it. Such reports are welcome through the channels in [SECURITY.md](SECURITY.md). The reporting obligations of manufacturers under Article 14 apply from 11 September 2026, and most other obligations from 11 December 2027.

### 7.3 General Data Protection Regulation

Imagery and location data can be personal data under Regulation (EU) 2016/679 [11]. What the software does with data, and the responsibilities of deployers, are described in [PRIVACY.md](PRIVACY.md).

## 8. Status of Compliance Activities

The table lists what the project does and does not do at the date of review.

| Area | Status |
| --- | --- |
| Project licence | MPL-2.0, text in [LICENSE.txt](LICENSE.txt), SPDX expression in the package metadata |
| File notices | Checked in CI for every tracked Python and shell file |
| REUSE Specification 3.3 | Checked in CI with `reuse lint` |
| Dependency licences | Installed runtime set checked in CI; new dependencies checked in pull requests |
| Software bill of materials | SPDX SBOM for each pushed container image, stored as a workflow artifact; none for the Python distributions |
| Release integrity | Sigstore signatures, SLSA provenance and GitHub artifact attestations for releases built by the current release workflow; see [SECURITY.md](SECURITY.md) |
| Export control classification | Not performed |
| AI Act classification | Not applicable to the library as published; the responsibility of those who build systems with it |
| Certifications and external audits (for example ISO/IEC 27001, SOC 2) | None |
| Formal legal review of this document | None |

## 9. Raising a Compliance Concern

Licensing errors (a missing notice, an incompatible dependency, unattributed third-party material), privacy concerns, AI regulation questions, export control questions and ethics concerns can be raised with the "Compliance, licensing and ethics" issue form ([.github/ISSUE_TEMPLATE/06_compliance.yml](.github/ISSUE_TEMPLATE/06_compliance.yml)) or by e-mail to <yunus.z.imanov@helsinki.fi>. Concerns that are also security vulnerabilities MUST be reported privately as described in [SECURITY.md](SECURITY.md). The maintainers correct confirmed licensing errors in the next release.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://www.mozilla.org/en-US/MPL/2.0/>

[4] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

[5] European Parliament and Council. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). 2024. <https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng>

[6] European Parliament and Council. Regulation (EU) 2021/821 setting up a Union regime for the control of exports, brokering, technical assistance, transit and transfer of dual-use items. 2021. <https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng>

[7] U.S. Department of Commerce, Bureau of Industry and Security. Export Administration Regulations, 15 CFR Parts 730 to 774. 2026. <https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C>

[8] European Commission. Commission Delegated Regulation (EU) No 1159/2013 on registration and licensing conditions for GMES users and access criteria for GMES dedicated data and GMES service information. 2013. <https://eur-lex.europa.eu/eli/reg_del/2013/1159/oj/eng>

[9] Council of the European Union. Council Regulation (EU) No 833/2014 concerning restrictive measures in view of Russia's actions destabilising the situation in Ukraine. 2014. <https://eur-lex.europa.eu/eli/reg/2014/833/oj/eng>

[10] European Parliament and Council. Regulation (EU) 2024/2847 on horizontal cybersecurity requirements for products with digital elements (Cyber Resilience Act). 2024. <https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng>

[11] European Parliament and Council. Regulation (EU) 2016/679 (General Data Protection Regulation). 2016. <https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng>

<!--
=============================================================================
End of file COMPLIANCE.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
