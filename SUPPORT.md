<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : SUPPORT.md
Title       : Support Policy
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Support Policy

| Field | Value |
| --- | --- |
| Document | UBX-DOC-SUPPORT |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This policy explains to users and contributors where to get help with Unbihexium, which channel suits which kind of request, what information a request needs, what response to expect, and which versions and platforms are supported. Unbihexium is a community open source project maintained by one person, Olaf Yunus Laitinen Imanov (University of Helsinki). Support is provided on a best-effort basis through public GitHub channels; there is no service level agreement and no paid or commercial support offered by the project. Security vulnerabilities follow a separate private process described in [SECURITY.md](SECURITY.md).

## Contents

- [1. Introduction](#1-introduction)
- [2. Before Asking for Help](#2-before-asking-for-help)
- [3. Support Channels](#3-support-channels)
- [4. Writing a Good Request](#4-writing-a-good-request)
- [5. Response Expectations](#5-response-expectations)
- [6. Supported Versions and Platforms](#6-supported-versions-and-platforms)
- [7. Scope of Support](#7-scope-of-support)
- [8. Related Documents](#8-related-documents)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

Clear support channels let questions be answered once, in public, where others can find the answer later. This policy directs each kind of request to the channel where it can be handled best and sets realistic expectations for a project with a single maintainer.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. They describe what the maintainers ask of people who request support.

## 2. Before Asking for Help

Many questions are already answered. Before opening a request, please:

1. read the [documentation](docs/index.md), in particular [installation](docs/getting_started/installation.md), the [quick start](docs/getting_started/quickstart.md), [configuration](docs/getting_started/configuration.md) and the [FAQ](docs/faq.md);
2. read the model card of the model you use in [model_zoo/cards/](model_zoo/cards/). The 520 models of the model zoo (130 families in 4 variants) are untrained starter models whose predictions are meaningless until they are trained; only the 7 spectral index families (28 models) compute exact formulas. Unexpected predictions from an untrained model are not a bug;
3. search the existing [issues](https://github.com/unbihexium-oss/unbihexium/issues?q=is%3Aissue) (open and closed) and [discussions](https://github.com/unbihexium-oss/unbihexium/discussions);
4. check that you use the latest release, and that the problem also occurs with it;
5. reduce the problem to a minimal example, using small synthetic arrays or files instead of your own data where possible.

The following commands print the information that most requests need:

```bash
unbihexium --version
unbihexium info
python --version
```

## 3. Support Channels

| Need | Channel |
| --- | --- |
| Usage question, idea or general discussion | [GitHub Discussions](https://github.com/unbihexium-oss/unbihexium/discussions), or the "Question" issue form |
| Defect in the library, CLI, REST service, models or Docker image | "Bug report" issue form |
| New capability, model, data source or improvement | "Feature request" issue form |
| Missing, incorrect or unclear documentation | "Documentation" issue form |
| Model defect, integrity problem or new model request | "Model zoo" issue form |
| Slow execution, high memory use or poor scaling | "Performance" issue form |
| Installation, packaging, container or CI problem | "Build, packaging or CI" issue form |
| Licensing, privacy, AI regulation, export control or ethics concern | "Compliance, licensing and ethics" issue form |
| Security vulnerability | Private report through [GitHub private vulnerability reporting](https://github.com/unbihexium-oss/unbihexium/security/advisories/new), as described in [SECURITY.md](SECURITY.md) |
| Private matter that cannot be discussed in public | E-mail to <yunus.z.imanov@helsinki.fi> |

All issue forms are available from the [new issue page](https://github.com/unbihexium-oss/unbihexium/issues/new/choose); blank issues are disabled so that every report contains the information needed to act on it. Security vulnerabilities MUST NOT be reported in public issues or discussions. E-mail SHOULD be used only when a public channel is not appropriate, because answers given in private help nobody else.

Conduct in all channels is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## 4. Writing a Good Request

A request SHOULD contain:

- the output of the commands in [Section 2](#2-before-asking-for-help), the operating system, and how Unbihexium was installed (pip with which extras, container image tag, or a source checkout with its commit);
- what you did, as a minimal code example or command that others can run;
- what you expected to happen and what happened instead, with the complete error message and traceback as text, not as a screenshot;
- for model questions, the model identifier (for example `ship_detector_base`) and whether the model was trained, and on what data;
- for REST service questions, the request (route, headers without secrets, a small body) and the response status.

Requests MUST NOT contain credentials, API keys, access tokens or data that you are not allowed to share publicly, such as personal data or licensed commercial imagery. Replace them with synthetic examples.

## 5. Response Expectations

### 5.1 Targets

The project is maintained by one person alongside other work. The following are targets, not guarantees:

| Request | Target for a first response |
| --- | --- |
| Security vulnerability | within 7 calendar days, see [SECURITY.md](SECURITY.md) |
| Bug report with a reproducible example | within 14 calendar days |
| Question, feature request, documentation issue | within 30 calendar days |
| Pull request | within 30 calendar days |

A first response can be a question, a label or a pointer to documentation; it is not a promise of a fix. Bugs that affect correctness of results, data integrity or security are prioritised over features. Community members are welcome to answer questions and review pull requests, and often do so faster than the maintainer.

### 5.2 Inactive Items

The Stale workflow ([.github/workflows/stale.yml](.github/workflows/stale.yml)) labels issues without activity for 60 days and pull requests without activity for 45 days as stale, and closes them 14 days later unless there is new activity. Items labelled `security`, `confirmed`, `blocked`, `help wanted`, `good first issue`, `breaking-change` or `compliance` (issues) or `security`, `blocked`, `breaking-change` or `dependencies` (pull requests), and draft pull requests, are never marked stale. A closed item MAY be reopened by commenting with new information.

## 6. Supported Versions and Platforms

| Item | Support |
| --- | --- |
| Unbihexium release | The latest 1.0.x release (1.0.1) and the main branch. Fixes are released in new patch versions; older patch versions are not updated. |
| Python | CPython 3.10 to 3.14, each tested in CI on every pull request. A Python version is dropped after its upstream end of life, as described in [VERSIONING.md](VERSIONING.md). |
| Operating systems | CI runs on Linux (Ubuntu, GitHub-hosted runners). macOS and Windows are expected to work where the dependencies provide wheels; problems specific to them are handled on a best-effort basis. |
| Optional extras | `onnx`, `torch`, `serving`, `zarr` and `parquet`. STAC search needs no extra. GPU use through the CUDA build of PyTorch is handled on a best-effort basis, because CI has no GPU. |
| Container image | Images published to `ghcr.io/unbihexium-oss/unbihexium` from the main branch and from version tags. |

## 7. Scope of Support

The project can help with:

- installing and configuring Unbihexium;
- using its public API, command line interface and REST service;
- defects in the software, the model zoo metadata and the documentation;
- understanding how the starter models are built and how they are trained with the tools in the package ([docs/model_zoo/training.md](docs/model_zoo/training.md)).

The project cannot provide:

- training of models on your data, or accuracy guarantees for models trained by anyone;
- consulting on specific applications, data acquisition or the interpretation of results;
- operation or debugging of deployments run by third parties;
- legal advice on licensing, data protection, AI regulation or export control ([COMPLIANCE.md](COMPLIANCE.md) gives general information only);
- help with uses listed as not supported in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md);
- commercial support, service level agreements or indemnification. Unbihexium is provided under the MPL-2.0 without warranty (Sections 6 and 7 of the licence in [LICENSE.txt](LICENSE.txt)).

Contributions that fix the problem you found are the fastest route to a solution; see [CONTRIBUTING.md](CONTRIBUTING.md).

## 8. Related Documents

- [SECURITY.md](SECURITY.md): private reporting of vulnerabilities.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to contribute code, documentation and models.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): rules for participation.
- [GOVERNANCE.md](GOVERNANCE.md) and [MAINTAINERS.md](MAINTAINERS.md): roles and decision making.
- [VERSIONING.md](VERSIONING.md): versioning and release policy.
- [PRIVACY.md](PRIVACY.md), [COMPLIANCE.md](COMPLIANCE.md) and [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md): data handling, licensing and use policy.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file SUPPORT.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
