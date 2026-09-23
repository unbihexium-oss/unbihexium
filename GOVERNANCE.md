<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : GOVERNANCE.md
Title       : Project Governance
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Project Governance

| Field | Value |
| --- | --- |
| Document | UBX-DOC-GOVERNANCE |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | The unbihexium-oss/unbihexium repository and its releases |

## Abstract

This document describes how the Unbihexium project is governed: the roles people hold, how decisions are made, how changes are reviewed, how maintainers are added and retired, and how the project would continue if its maintainer became unavailable. It is written for contributors who want to know how their work is decided on, for people considering taking on a maintainer role, and for users and auditors who assess the project's review practice and continuity. It describes the project as it is today, with a single maintainer, and the rules that apply as the maintainer group grows. The review policy referenced by `security-insights.yml` is Section 5 of this document.

## Contents

- [1. Introduction](#1-introduction)
- [2. Governance model](#2-governance-model)
- [3. Roles](#3-roles)
- [4. Decision making](#4-decision-making)
- [5. Code review policy](#5-code-review-policy)
- [6. Adding maintainers](#6-adding-maintainers)
- [7. Stepping down and removal](#7-stepping-down-and-removal)
- [8. Continuity](#8-continuity)
- [9. Access and security of accounts](#9-access-and-security-of-accounts)
- [10. Releases](#10-releases)
- [11. Conduct and conflicts of interest](#11-conduct-and-conflicts-of-interest)
- [12. Changes to this document](#12-changes-to-this-document)
- [References](#references)

## 1. Introduction

### 1.1 Scope

This document governs the repository `unbihexium-oss/unbihexium` on GitHub, the Python package `unbihexium` published from it, the container image built from it and the model zoo metadata it contains.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals.

### 1.3 Legal and organisational context

The source code is licensed under the Mozilla Public License 2.0 (see [LICENSE.txt](LICENSE.txt)). Copyright notices name "Unbihexium OSS Foundation and contributors", and [CITATION.cff](CITATION.cff) lists the Unbihexium OSS Foundation as the copyright holder. The project does not use a contributor licence agreement: contributors keep the copyright of their contributions and license them under MPL-2.0, certifying the Developer Certificate of Origin as described in [CONTRIBUTING.md](CONTRIBUTING.md). The repository is hosted in the GitHub organisation `unbihexium-oss`. This document describes project practice and is not legal advice.

## 2. Governance model

Unbihexium is a maintainer-led project. The maintainers listed in [MAINTAINERS.md](MAINTAINERS.md) take the decisions about the code, the releases and the project policies, in public and on the basis of the discussion with contributors and users.

At present the project has **one maintainer**, Olaf Yunus Laitinen Imanov (University of Helsinki), who is also the lead maintainer. Consequently:

- every decision described in this document is currently taken by that person;
- changes authored by the maintainer are not reviewed by a second person (Section 5.3);
- the project depends on one person for reviews, releases and security response (Section 8).

The project intends to widen the maintainer group. Section 6 defines how this happens, and the rules in Sections 4 and 5 already describe how decisions and reviews work once there are two or more maintainers.

## 3. Roles

### 3.1 Users

Anyone who uses the software. Users take part by reporting bugs, requesting features and asking questions through the issue forms, and by reporting vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

### 3.2 Contributors

Anyone who has had a contribution merged: code, tests, documentation, model zoo metadata, CI configuration or substantial review. Contributors MAY add themselves to [AUTHORS.md](AUTHORS.md). Contributors have no repository permissions beyond those GitHub grants to every account.

### 3.3 Maintainers

Maintainers have write access to the repository and are listed in [MAINTAINERS.md](MAINTAINERS.md) and in `.github/CODEOWNERS` for the areas they are responsible for. Maintainers:

- review and merge pull requests according to Section 5 and [CONTRIBUTING.md](CONTRIBUTING.md);
- triage issues and keep the labels, milestones and issue forms in order;
- handle vulnerability reports according to [SECURITY.md](SECURITY.md);
- enforce the [Code of Conduct](CODE_OF_CONDUCT.md);
- keep the documentation, the licensing and compliance documents and the model zoo metadata accurate.

### 3.4 Lead maintainer

One maintainer is the lead maintainer. The lead maintainer holds administrator rights on the repository and the organisation and controls the PyPI publishing token (the `PYPI_API_TOKEN` repository secret used by the release workflow), and decides when the maintainers cannot reach agreement (Section 4.3). The lead maintainer is currently Olaf Yunus Laitinen Imanov.

### 3.5 Emeritus maintainers

Former maintainers who stepped down or became inactive (Section 7). They are listed in [MAINTAINERS.md](MAINTAINERS.md) in recognition of their work and hold no repository permissions. There are currently no emeritus maintainers.

## 4. Decision making

### 4.1 Where decisions are made

Decisions about the project are made in public, in GitHub issues and pull requests. Decisions that affect users SHOULD be recorded in an issue or pull request that states the decision and its reasons, so that the history remains traceable. Security matters under embargo are the exception: they are discussed in private security advisories and published when the advisory is disclosed.

### 4.2 Lazy consensus

Most decisions are made by lazy consensus: a maintainer proposes a change in a pull request or issue, and it is accepted if no maintainer objects within a reasonable time. An objection MUST give reasons and SHOULD propose an alternative.

### 4.3 Decisions that need explicit agreement

The following decisions need the explicit agreement of the maintainers, recorded in the relevant issue or pull request, once there are two or more maintainers:

| Decision | Required agreement |
| --- | --- |
| Breaking change to the public API, the command line or the REST API | Majority of the maintainers |
| Adding or removing a runtime dependency | Majority of the maintainers |
| Change to the licence, the licence policy for dependencies or the release process | All maintainers |
| Adding or removing a maintainer | Majority of the maintainers, excluding the person concerned |
| Change to this document | Majority of the maintainers |

If the maintainers cannot reach agreement after discussion, the lead maintainer decides and records the reasons. While the project has a single maintainer, that maintainer takes these decisions alone and records them in the same way.

### 4.4 Input from contributors and users

Contributors and users are invited to comment on any public proposal. Proposals with significant effect on users (breaking changes, removal of features, changes to supported Python versions) SHOULD remain open for comment for at least 14 days before they are merged, except when an urgent security fix requires otherwise.

## 5. Code review policy

### 5.1 General rules

1. Every change to the `main` branch MUST be made through a pull request.
2. A pull request MUST pass the applicable automated checks described in [CONTRIBUTING.md](CONTRIBUTING.md) before it is merged.
3. A pull request from a contributor who is not a maintainer MUST be reviewed and approved by a maintainer before it is merged.
4. Review follows the criteria in [CONTRIBUTING.md](CONTRIBUTING.md), Section 11.1.

### 5.2 Review with two or more maintainers

Once the project has two or more maintainers:

- every pull request SHOULD be approved by at least one maintainer other than its author before it is merged;
- pull requests that change the release process, the signing and provenance configuration, the workflows in `.github/workflows/`, the licence documents or security-relevant code MUST be approved by a maintainer other than the author;
- the repository settings SHOULD be changed so that the default branch requires one approving review from someone other than the last pusher, and CODEOWNERS SHOULD be updated so that each area has at least two owners.

An urgent security fix MAY be merged without a second approval when no other maintainer can be reached in time; it MUST then be reviewed by another maintainer after the fact, and the reason MUST be recorded in the pull request.

### 5.3 Review with a single maintainer

While the project has a single maintainer, rule 5.1.3 is fulfilled by that maintainer. Pull requests authored by the maintainer are merged after the automated checks pass, without an independent human review, because no second reviewer exists. The automated checks (tests on Python 3.10 to 3.14, integration and end-to-end tests, CodeQL, fuzzing, dependency review, licence and secret scanning) partly compensate for this, but do not replace human review. Assessments such as the OpenSSF Scorecard Code-Review check [3] will reflect the missing independent review until further maintainers are added. Reviews of the maintainer's pull requests by other contributors are welcome.

## 6. Adding maintainers

### 6.1 Why the project adds maintainers

Additional maintainers are needed so that changes can be reviewed by someone other than their author (Section 5.2), so that releases and security response do not depend on one person (Section 8), and so that the areas of the project have owners with the relevant expertise.

### 6.2 Criteria

A candidate SHOULD:

1. have contributed to the project over a sustained period, typically at least three months, through merged pull requests, reviews or issue triage;
2. have shown good technical judgement and knowledge of at least one area of the project (for example raster and vector input and output, SAR, terrain, the model zoo, the REST service, packaging or CI);
3. have followed the contribution rules, the text policy and the [Code of Conduct](CODE_OF_CONDUCT.md) consistently;
4. be willing to take on the responsibilities of Section 3.3, including code review and, where agreed, security response;
5. agree to the account security requirements of Section 9.

### 6.3 Procedure

1. **Nomination.** A maintainer nominates the candidate, or a contributor nominates themselves, by e-mail to the lead maintainer at <yunus.z.imanov@helsinki.fi>. The nomination names the proposed areas of responsibility and links to representative contributions. Nominations are handled privately so that a declined nomination does not become public.
2. **Consent.** The candidate confirms that they accept the role and the requirements of Section 9.
3. **Decision.** The maintainers decide according to Section 4.3. While there is a single maintainer, that maintainer decides.
4. **Onboarding.** The new maintainer is added in a single pull request that:
   - adds them to [MAINTAINERS.md](MAINTAINERS.md) with their areas;
   - adds them to `.github/CODEOWNERS` for those areas;
   - adds them to `core-team` in `security-insights.yml` and updates its `last-updated` date;
   - where they will act as security contact, adds them to [SECURITY.md](SECURITY.md) and the `champions` in `security-insights.yml`.
5. **Access.** After the pull request is merged, the lead maintainer grants write access to the repository (or a more limited role where agreed) and, where agreed, access to private security advisories.
6. **Review settings.** When the second maintainer is added, the lead maintainer applies the repository settings described in Section 5.2.

A new maintainer MAY start with triage permissions and receive write access after a period of working together.

## 7. Stepping down and removal

### 7.1 Stepping down

A maintainer MAY step down at any time by informing the other maintainers. They are moved to the emeritus list in [MAINTAINERS.md](MAINTAINERS.md), removed from `.github/CODEOWNERS` and `security-insights.yml`, and their access rights are revoked.

### 7.2 Inactivity

A maintainer who has not taken part in reviews, merges, triage or discussion for six months MAY be asked whether they wish to continue. If they do not answer within 30 days, or decide not to continue, they become emeritus as in Section 7.1. An emeritus maintainer can return through the procedure of Section 6.3.

### 7.3 Removal

A maintainer MAY be removed for a serious or repeated breach of the [Code of Conduct](CODE_OF_CONDUCT.md), of this document or of the security requirements of Section 9, by decision of the other maintainers according to Section 4.3. Access MAY be suspended at once when an account is suspected to be compromised.

## 8. Continuity

With a single maintainer, the project has a bus factor of one: if the maintainer becomes unavailable, reviews, releases and security response stop. The following arrangements limit the effect:

- The source code, the history, the issue tracker and the release artifacts are public, and the licence permits anyone to continue the work in a fork.
- The build, signing and publishing of releases are automated in `.github/workflows/release.yml` and documented in [CONTRIBUTING.md](CONTRIBUTING.md), so that a successor can reproduce them.
- The project data needed to continue (lock files, workflows, model zoo catalogue and checksums) is kept in the repository rather than in personal systems.

Adding maintainers (Section 6) is the main measure to raise the bus factor. When there are two or more maintainers, at least two of them SHOULD hold administrator rights on the organisation and the ability to publish releases.

## 9. Access and security of accounts

Maintainers MUST:

- enable two-factor authentication on their GitHub account;
- keep repository secrets (such as the PyPI publishing token) out of logs, issues, pull requests and local files that are shared;
- use the least privilege needed for a task and not share credentials;
- report a suspected compromise of their account or of a credential to the other maintainers at once and treat it as a security incident under [SECURITY.md](SECURITY.md).

Maintainers SHOULD sign their commits and tags.

## 10. Releases

Releases are prepared by a maintainer. The version in `pyproject.toml` and the changelog are updated in a pull request, and a tag `v<version>` is pushed. `.github/workflows/release.yml` then builds the sdist and the wheel, signs them with Sigstore (`.sigstore.json` bundles), creates GitHub artifact attestations and SLSA provenance (`unbihexium-<tag>.intoto.jsonl`), creates the GitHub release and uploads the distributions to PyPI. The versioning and compatibility policy is described in [VERSIONING.md](VERSIONING.md); notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## 11. Conduct and conflicts of interest

All participants are bound by the [Code of Conduct](CODE_OF_CONDUCT.md). A maintainer MUST disclose a conflict of interest (for example an employer's or client's interest in a decision, or a personal relationship with the author of a pull request) in the relevant issue or pull request and SHOULD abstain from the decision when another maintainer can take it. While there is a single maintainer, the disclosure is still made so that users can take it into account.

## 12. Changes to this document

Changes to this document are proposed in a pull request and decided according to Section 4.3. Substantial changes SHOULD remain open for comment for at least 14 days. The document is reviewed at least once a year and whenever the maintainer group changes.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Open Source Security Foundation. OpenSSF Scorecard checks: Code-Review. 2025. <https://github.com/ossf/scorecard/blob/main/docs/checks.md#code-review>

<!--
=============================================================================
End of file GOVERNANCE.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
