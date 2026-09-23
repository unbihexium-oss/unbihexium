<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : MAINTAINERS.md
Title       : Maintainers
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Maintainers

| Field | Value |
| --- | --- |
| Document | UBX-DOC-MAINTAINERS |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | The unbihexium-oss/unbihexium repository |

## Abstract

This document lists the people who maintain Unbihexium, their areas of responsibility and how to reach them. It is intended for contributors who need to know who reviews their work, for users and security researchers who need a contact, and for auditors who assess the project's review capacity. The project currently has one maintainer. The rules for becoming, remaining and ceasing to be a maintainer are defined in [GOVERNANCE.md](GOVERNANCE.md); this document records the result.

## Contents

- [1. Current maintainers](#1-current-maintainers)
- [2. Areas of responsibility](#2-areas-of-responsibility)
- [3. Review capacity](#3-review-capacity)
- [4. Responsibilities](#4-responsibilities)
- [5. Contact](#5-contact)
- [6. Emeritus maintainers](#6-emeritus-maintainers)
- [7. Becoming a maintainer](#7-becoming-a-maintainer)
- [8. Keeping this list accurate](#8-keeping-this-list-accurate)

## 1. Current maintainers

| Name | GitHub | Affiliation | Role | Since |
| --- | --- | --- | --- | --- |
| Olaf Yunus Laitinen Imanov | [@olaflaitinen](https://github.com/olaflaitinen) | University of Helsinki | Lead maintainer, security contact | Start of the project (2025) |

Olaf Yunus Laitinen Imanov is currently the only maintainer and is also listed as `core-team` member and security champion in `security-insights.yml` and as the contact in [CITATION.cff](CITATION.cff).

## 2. Areas of responsibility

`.github/CODEOWNERS` assigns every path of the repository to the lead maintainer, so that GitHub requests their review on every pull request. The areas below are listed separately so that new maintainers can be assigned to them.

| Area | Paths | Maintainers |
| --- | --- | --- |
| Library source code | `src/` | @olaflaitinen |
| Tests and fuzz targets | `tests/`, `fuzz/` | @olaflaitinen |
| Model zoo | `model_zoo/`, `src/unbihexium/zoo/` | @olaflaitinen |
| Documentation and examples | `docs/`, `examples/`, root Markdown documents | @olaflaitinen |
| CI, workflows and repository configuration | `.github/` | @olaflaitinen |
| Packaging and containers | `pyproject.toml`, lock files, `Dockerfile`, `docker-compose.yml`, `deploy/` | @olaflaitinen |
| Licensing, privacy and compliance documents | `LICENSE.txt`, `NOTICE`, `NOTICE.md`, `THIRD_PARTY_NOTICES.md`, `COMPLIANCE.md`, `PRIVACY.md` | @olaflaitinen |
| Releases | `.github/workflows/release.yml`, tags, PyPI | @olaflaitinen |
| Security response | Private security advisories, [SECURITY.md](SECURITY.md) | @olaflaitinen |
| Code of Conduct enforcement | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | @olaflaitinen |

## 3. Review capacity

With one maintainer, pull requests from contributors are reviewed by that maintainer, and pull requests authored by the maintainer are merged after the automated checks pass without an independent human review. Releases, security response and Code of Conduct enforcement also depend on one person. [GOVERNANCE.md](GOVERNANCE.md) explains the consequences, the review rules that apply once there are two or more maintainers, and the procedure for adding maintainers. Contributors interested in the role are encouraged to read it.

## 4. Responsibilities

Maintainers:

- review and merge pull requests according to [CONTRIBUTING.md](CONTRIBUTING.md) and the checks in `.github/workflows/`;
- triage issues and keep labels and issue forms in order;
- handle vulnerability reports according to [SECURITY.md](SECURITY.md);
- enforce the [Code of Conduct](CODE_OF_CONDUCT.md);
- prepare releases according to [VERSIONING.md](VERSIONING.md) and keep [CHANGELOG.md](CHANGELOG.md) up to date;
- keep the licensing, privacy and compliance documents accurate.

## 5. Contact

| Purpose | Channel |
| --- | --- |
| Bugs, features, questions | The issue forms of the repository; see [SUPPORT.md](SUPPORT.md) |
| Security vulnerabilities | A private security advisory at <https://github.com/unbihexium-oss/unbihexium/security/advisories/new> or e-mail to <yunus.z.imanov@helsinki.fi>, as described in [SECURITY.md](SECURITY.md); never a public issue |
| Code of Conduct reports | E-mail to <yunus.z.imanov@helsinki.fi>, as described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Maintainer nominations and other private matters | E-mail to <yunus.z.imanov@helsinki.fi> |

The project is maintained on a best-effort basis and does not guarantee response times.

## 6. Emeritus maintainers

None.

## 7. Becoming a maintainer

The criteria and the procedure are defined in [GOVERNANCE.md](GOVERNANCE.md), Section 6. In summary, a contributor with sustained, high-quality contributions to at least one area is nominated (or nominates themselves) by e-mail, and after the decision is added to this file, to `.github/CODEOWNERS` and to `security-insights.yml` in one pull request before receiving repository access.

## 8. Keeping this list accurate

This file is updated in the same pull request that changes `.github/CODEOWNERS` or the `core-team` of `security-insights.yml`, and whenever a maintainer joins, changes areas or becomes emeritus. It is reviewed together with [GOVERNANCE.md](GOVERNANCE.md) at least once a year.

<!--
=============================================================================
End of file MAINTAINERS.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
