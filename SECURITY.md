<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : SECURITY.md
Title       : Security Policy
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Security Policy

| Field | Value |
| --- | --- |
| Document | UBX-DOC-201 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This policy tells security researchers, users and downstream packagers how to report a vulnerability in Unbihexium privately, which versions receive security fixes, how a report is handled from receipt to public advisory, and which supply-chain controls protect the source code, the dependencies and the released artefacts. It covers the Python package, the command line interface, the REST service, the model zoo, the container image, the deployment manifests and the GitHub Actions workflows of the repository. The project is maintained by a single person; the timelines below are therefore targets that one maintainer can honour, not contractual guarantees. The machine-readable counterpart of this policy is [security-insights.yml](security-insights.yml), which follows the OpenSSF Security Insights specification [8].

## Contents

- [1. Introduction](#1-introduction)
- [2. Supported Versions](#2-supported-versions)
- [3. Reporting a Vulnerability](#3-reporting-a-vulnerability)
- [4. Scope](#4-scope)
- [5. Handling and Disclosure Process](#5-handling-and-disclosure-process)
- [6. Supply-Chain Controls](#6-supply-chain-controls)
- [7. Verifying Released Artefacts](#7-verifying-released-artefacts)
- [8. Secure Operation](#8-secure-operation)
- [9. Related Documents](#9-related-documents)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

Unbihexium is an open source Earth observation, geospatial, remote sensing and SAR library for Python, distributed under the Mozilla Public License 2.0. It reads untrusted input (raster files, GeoJSON, STAC documents, model checkpoints, HTTP requests to the REST service), so defects in it can have security consequences for the systems that use it. This policy defines how such defects are reported, assessed, fixed and disclosed.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in all capitals. Obligations written with these words apply to reporters where the sentence addresses the reporter, and to the maintainers otherwise.

### 1.3 Maintainer Capacity

All security reports are received and handled by the sole maintainer, Olaf Yunus Laitinen Imanov (University of Helsinki), as listed in [MAINTAINERS.md](MAINTAINERS.md). There is no dedicated security team, no on-call rotation and no paid bug bounty programme. Holidays, illness or other absence can delay a response; when that happens, the maintainer will say so in the advisory thread as soon as possible.

## 2. Supported Versions

Security fixes are made on the main branch and released as a new patch version of the latest release series. Earlier patch versions are not patched separately; users MUST upgrade to the newest patch release to receive a fix.

| Version | Status | Security fixes |
| --- | --- | --- |
| main branch | Development | Yes, fixes land here first |
| 2.0.x (latest: 2.0.1) | Current release series | Yes, in the next 2.0.x patch release |
| 1.0.x | Previous release series | No; upgrade to 2.0.x ([docs/MIGRATION.md](docs/MIGRATION.md)) |
| Earlier than 1.0.0 | Not released on PyPI | No |

The package supports CPython 3.10 to 3.14. A vulnerability that exists only on a Python version that is past its upstream end of life is out of scope; the Python support policy is described in [VERSIONING.md](VERSIONING.md).

Only the latest release series receives fixes, as stated in [VERSIONING.md](VERSIONING.md). When a new minor or major series is released, this table is updated and the previous series stops receiving security fixes.

## 3. Reporting a Vulnerability

### 3.1 Private Channels

Reporters MUST NOT disclose a suspected vulnerability in a public issue, pull request, discussion or other public forum before a fix is released. Use one of the following private channels:

1. **GitHub private vulnerability reporting (preferred).** Open a draft advisory at <https://github.com/unbihexium-oss/unbihexium/security/advisories/new>. The report is visible only to the reporter and the maintainer, supports private discussion and collaboration on a fix in a temporary private fork, and becomes the published advisory once the issue is fixed [4].
2. **E-mail.** Send the report to <yunus.z.imanov@helsinki.fi> with a subject line that starts with `[SECURITY] unbihexium`. E-mail is not encrypted end to end; the project does not publish an OpenPGP key. Where the details are sensitive, send a short e-mail asking for contact and share the details through a GitHub advisory afterwards.

The issue chooser of the repository ([.github/ISSUE_TEMPLATE/config.yml](.github/ISSUE_TEMPLATE/config.yml)) links to the private reporting form, and blank public issues are disabled.

### 3.2 Content of a Report

A report SHOULD contain:

- the affected component (module, CLI command, REST route, workflow, container image or deployment manifest) and the affected versions or commit;
- the Python version, the operating system and the installed extras (`unbihexium --version` and `pip show unbihexium` print the version);
- a description of the vulnerability and its impact (for example code execution, information disclosure, denial of service, integrity bypass);
- the minimal steps, input files or requests needed to reproduce it, using synthetic data where possible;
- a proposed CVSS vector if the reporter has one, and any known mitigation;
- whether and how the reporter wishes to be credited.

Reports MAY be written in English, which is the working language of the project.

### 3.3 Good-Faith Research

The maintainer will not pursue or support legal action against anyone who researches and reports a vulnerability in good faith and in line with this policy: testing only against their own installations, not accessing or modifying data of others, not degrading services operated by third parties, and giving the maintainer a reasonable opportunity to fix the issue before disclosure. This statement concerns the maintainer only; it cannot authorise testing against systems that belong to other parties.

## 4. Scope

### 4.1 In Scope

The following components of this repository are covered by the policy (see also `vulnerability-reporting.in-scope` in [security-insights.yml](security-insights.yml)):

- the source code of the `unbihexium` Python package under [src/](src/);
- the command line interface (`unbihexium`) and the REST service (`unbihexium.serving`);
- the model zoo catalogue, manifests and checksums under [model_zoo/](model_zoo/) and [src/unbihexium/zoo/](src/unbihexium/zoo/), including the checkpoint loader and the digest verification;
- the container image built from the [Dockerfile](Dockerfile);
- the deployment manifests under [deploy/](deploy/);
- the GitHub Actions workflows and the release pipeline under [.github/](.github/).

### 4.2 Out of Scope

The following are not handled as vulnerabilities of this project:

- publicly known vulnerabilities in third-party dependencies; report them to the upstream project. A vulnerability in the way Unbihexium uses a dependency is in scope;
- deployments operated by third parties;
- findings that require an already compromised host or a malicious local administrator;
- denial of service through intentionally oversized inputs without a practical amplification, where the documented request, pixel and value limits of the REST service are respected;
- the quality of predictions of the model zoo starter models. The 520 models of the zoo (130 families in 4 variants) are untrained starter models with deterministic weights, except the 7 spectral index families (28 models) that compute exact formulas. Their outputs are not meaningful until the models are trained, which is documented behaviour and not a vulnerability.

## 5. Handling and Disclosure Process

### 5.1 Process

The process follows the principles of coordinated vulnerability disclosure described in ISO/IEC 29147 [6] and ISO/IEC 30111 [7], scaled to a single-maintainer project:

1. **Acknowledgement.** The maintainer confirms receipt and, where needed, asks for missing information.
2. **Triage.** The maintainer reproduces the issue, decides whether it is a vulnerability within scope, and assigns a severity (Section 5.3).
3. **Fix.** The fix is developed in the private fork of the GitHub advisory, with the reporter invited to review it. Regression tests are added; crashes found by fuzzing are added to the corpus under [fuzz/corpus/](fuzz/corpus/).
4. **Release.** A patch release is built and published by the release workflow (Section 6.4).
5. **Disclosure.** The GitHub security advisory is published with the affected and fixed versions, the severity, the credit and a CVE identifier where one was assigned. The fix is recorded in [CHANGELOG.md](CHANGELOG.md).

### 5.2 Timelines

| Step | Target |
| --- | --- |
| Acknowledgement of receipt | within 7 calendar days |
| Initial assessment (in scope or not, provisional severity) | within 21 calendar days |
| Status updates to the reporter | at least every 30 days until resolution |
| Fix released for critical and high severity | target within 30 days of confirmation |
| Fix released for medium and low severity | target within 90 days of confirmation, or with the next regular release |
| Public disclosure | when the fix is released, and no later than 90 days after the report unless both parties agree otherwise |

These are best-effort targets. If a target cannot be met, the maintainer will tell the reporter why and agree on a new date. A reporter MAY disclose the vulnerability after 90 days if no fix and no agreed extension exists; the reporter SHOULD notify the maintainer beforehand. If a vulnerability is already being exploited or has been disclosed publicly, the maintainer will prioritise a fix or a documented mitigation and may publish an advisory before a fix exists.

### 5.3 Severity Assessment

Severity is assessed with the Common Vulnerability Scoring System, version 4.0 [3], using the base metrics and the usual deployment of the affected component (for example, the REST service reachable over a network, the library processing files supplied by a user). The qualitative ratings of CVSS (none, low, medium, high, critical) are used in the advisory. The score is a guide for prioritisation, not a measure of risk in a particular deployment.

### 5.4 Advisories and CVE Identifiers

Confirmed vulnerabilities in released versions are published as GitHub security advisories of the repository at <https://github.com/unbihexium-oss/unbihexium/security/advisories>. GitHub is a CVE Numbering Authority; the maintainer requests a CVE identifier through the advisory for every confirmed vulnerability that affects a released version. Published advisories are imported into the GitHub Advisory Database, from which tools such as pip-audit and Dependabot notify users.

### 5.5 Credit

Reporters are credited in the advisory and in the changelog under the name or handle they choose, unless they ask to remain anonymous. There is no monetary reward.

## 6. Supply-Chain Controls

This section lists the controls that exist in the repository at the date of review. Each entry names the file that implements it, so that it can be audited. Controls that are informational only, or that do not cover every artefact, are marked as such.

### 6.1 Source Code and Review

- Every file is owned by the maintainer through [.github/CODEOWNERS](.github/CODEOWNERS), so review requests are assigned automatically.
- Pull request titles are checked against the Conventional Commits format ([.github/workflows/pr-title.yml](.github/workflows/pr-title.yml)).
- Secrets: TruffleHog scans the commits of every push to main and every pull request and fails on verified live credentials ([.github/workflows/secret-scan.yml](.github/workflows/secret-scan.yml)).
- Every third-party action in [.github/workflows/](.github/workflows/) is pinned by full commit SHA, and the workflows declare least-privilege `GITHUB_TOKEN` permissions (read-only by default, widened per job only where needed).
- actionlint, with shellcheck of the run steps, checks every workflow change ([.github/workflows/workflow-lint.yml](.github/workflows/workflow-lint.yml)); CodeQL also analyses the workflows (Section 6.3).
- Configuration files, including [security-insights.yml](security-insights.yml), are validated against their schemas ([.github/workflows/repo-config.yml](.github/workflows/repo-config.yml)).
- Text policy and documentation style checks run on every pull request ([.github/workflows/text-policy.yml](.github/workflows/text-policy.yml), [.github/workflows/markdown.yml](.github/workflows/markdown.yml)).

### 6.2 Dependencies

- Runtime dependencies are locked with exact versions and SHA-256 hashes in [requirements.txt](requirements.txt); the tools used by CI are locked in the same way under [.github/requirements/](.github/requirements/), and workflows install them with `pip install --require-hashes`. The CPU build of PyTorch has its own hashed lock, regenerated by [.github/workflows/torch-lock.yml](.github/workflows/torch-lock.yml).
- Dependabot proposes weekly updates for pip packages and GitHub Actions and monthly updates for the Docker base image ([.github/dependabot.yml](.github/dependabot.yml)).
- Dependency Review blocks pull requests that add dependencies with high or critical known vulnerabilities, or with licences denied by the project ([.github/dependency-review-config.yml](.github/dependency-review-config.yml), run by [.github/workflows/security.yml](.github/workflows/security.yml)).
- pip-audit checks [requirements.txt](requirements.txt) and [requirements-dev.txt](requirements-dev.txt) against the PyPI advisory database on pushes to main, pull requests and weekly. Any known vulnerability fails the job.

### 6.3 Static Analysis and Fuzzing

- CodeQL analyses the Python code and the GitHub Actions workflows with the `security-extended` query suite on pull requests, pushes to main and weekly ([.github/workflows/codeql.yml](.github/workflows/codeql.yml)). Results appear in GitHub code scanning.
- Bandit scans [src/](src/) with the settings in [pyproject.toml](pyproject.toml) on pushes to main, pull requests and weekly. Any finding fails the job.
- Coverage-guided fuzzing with atheris runs the targets in [fuzz/](fuzz/) (GeoJSON validation, bounds and orientation; STAC item and time parsing) for two minutes per target when the parsers change and for twenty minutes per target weekly ([.github/workflows/fuzz.yml](.github/workflows/fuzz.yml)). Crashing inputs become regression seeds that the unit tests replay.

### 6.4 Build and Release

Releases are built only by [.github/workflows/release.yml](.github/workflows/release.yml) when a version tag (`v*`) is pushed. The workflow:

1. builds the source distribution and the wheel with hashed, locked build tools;
2. writes `SHA256SUMS.txt` for the distributions;
3. installs the wheel with the hashed runtime lock in a clean environment, writes its SPDX SBOM `unbihexium-<tag>.spdx.json` and attests it for the distributions (`actions/attest-sbom`);
4. creates GitHub artifact attestations with SLSA build provenance v1 [9] for every distribution (`actions/attest-build-provenance`);
5. signs every distribution with Sigstore [10] in keyless mode, bound to the workflow identity, producing a `.sigstore.json` bundle per file;
6. exports the signed provenance as the release asset `unbihexium-<tag>.intoto.jsonl`;
7. creates the GitHub release with the distributions, the bundles, the provenance, the SBOM and the checksums;
8. uploads the distributions to PyPI with trusted publishing: PyPI accepts the short-lived OIDC token of the job in the environment `pypi`, so no upload token is stored, and PEP 740 attestations are uploaded with the files.

The signing, attestation and provenance steps were added to the workflow after the 1.0.1 release. The existing releases v1.0.0 and v1.0.1 therefore carry only `SHA256SUMS.txt`, and the checksums in the v1.0.1 GitHub release do not match the files on PyPI; verify PyPI downloads against the SHA-256 digests that PyPI publishes. Release 2.0.0 is the first to carry the full set of signed artefacts described above, and every later release does too.

### 6.5 Container Image

- The [Dockerfile](Dockerfile) pins the base image by digest, installs the dependencies from the hashed [requirements-docker.txt](.github/requirements/requirements-docker.txt) and the CPU build of PyTorch from [requirements-ci-torch.txt](.github/requirements/requirements-ci-torch.txt) with binary wheels only, builds the wheel with the hash-pinned backend of [requirements-build.txt](.github/requirements/requirements-build.txt), and runs as the unprivileged user `unbihexium` (UID 1000).
- [.github/workflows/docker.yml](.github/workflows/docker.yml) pushes the image to `ghcr.io/unbihexium-oss/unbihexium` on pushes to main and on version tags. For every pushed digest it stores an SPDX SBOM as a workflow artifact, attests the build provenance and the SBOM in the registry and signs the digest with cosign in keyless mode, bound to the workflow identity. Images pushed before these steps were added are neither signed nor attested.
- Grype scans the image when the Dockerfile or dependencies change, weekly and on demand, fails on critical vulnerabilities that have a released fix, and uploads its SARIF report to code scanning ([.github/workflows/container-scan.yml](.github/workflows/container-scan.yml)).

Each release built by the current workflow carries the SPDX SBOM of the distributions (Section 6.4).

### 6.6 Model Zoo Integrity

- The weights of every catalogue model are generated locally and deterministically from the model identifier, and their SHA-256 weights digest is published in [src/unbihexium/zoo/digests.json](src/unbihexium/zoo/digests.json) and [model_zoo/checksums.txt](model_zoo/checksums.txt). `load_model` and `unbihexium zoo build` compare the built weights with the published digest and raise an error on a mismatch.
- Cached models carry a `model.sha256` file; `unbihexium zoo verify <model_id>` checks the files and the weights digest.
- Checkpoints are loaded with `torch.load(weights_only=True)`, which refuses to unpickle arbitrary Python objects ([src/unbihexium/zoo/checkpoint.py](src/unbihexium/zoo/checkpoint.py)).
- Downloads of user-registered checkpoints are limited to 4 GiB and use HTTPS certificate verification by default.
- [.github/workflows/model-zoo.yml](.github/workflows/model-zoo.yml) checks that the catalogue, manifests, cards and digests are consistent.

### 6.7 Continuous Assessment

The OpenSSF Scorecard [5] runs weekly and on pushes to main ([.github/workflows/scorecard.yml](.github/workflows/scorecard.yml)); its results are published to the public Scorecard API and to GitHub code scanning. No formal external security audit of Unbihexium has been performed.

## 7. Verifying Released Artefacts

For releases built by the current release workflow, a downloaded distribution can be verified in three independent ways. Replace `<tag>` with the release tag (for example `v1.0.2`) and `<file>` with the distribution file name.

Check the SHA-256 digests against the checksum file of the release:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
```

Verify the Sigstore signature, with the `.sigstore.json` bundle in the same directory as the file (requires `pip install sigstore`):

```bash
python -m sigstore verify github \
  --cert-identity https://github.com/unbihexium-oss/unbihexium/.github/workflows/release.yml@refs/tags/<tag> \
  <file>
```

Verify the GitHub artifact attestation and its SLSA provenance (requires the GitHub CLI):

```bash
gh attestation verify <file> --repo unbihexium-oss/unbihexium
```

A container image is verified by its keyless signature and its provenance attestation:

```bash
cosign verify ghcr.io/unbihexium-oss/unbihexium:<tag> \
  --certificate-identity-regexp '^https://github.com/unbihexium-oss/unbihexium/.github/workflows/docker.yml@refs/(heads/main|tags/v.+)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
gh attestation verify oci://ghcr.io/unbihexium-oss/unbihexium:<tag> --repo unbihexium-oss/unbihexium
```

A locally built model zoo model is verified with:

```bash
unbihexium zoo build ship_detector_tiny
unbihexium zoo verify ship_detector_tiny
```

## 8. Secure Operation

These recommendations address operators of Unbihexium. They are not vulnerabilities when ignored, but they reduce exposure.

- **REST service authentication.** The service (`unbihexium serve`) has no API key by default. Operators exposing it beyond the local host SHOULD set one with `UNBIHEXIUM_SERVING__API_KEY`; clients then send it in the `X-API-Key` header, which is required on every route except `/health` and compared in constant time.
- **CORS.** The default CORS origin list is `*`. Operators SHOULD restrict it with `UNBIHEXIUM_SERVING__CORS_ORIGINS` to the origins that need browser access.
- **Rate limiting.** The per-client rate limit is off by default (`0`). Operators SHOULD set `UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE` for public deployments.
- **Transport security.** The service speaks plain HTTP. Operators MUST place it behind a reverse proxy that terminates TLS when it is reachable over an untrusted network.
- **Resource limits.** By default the service answers HTTP 413 to request bodies above 10 MiB, to images with more than 2048 x 2048 pixels (rows times columns) and to input or output arrays with more than 16 x 1024 x 1024 values. Lower these limits with the `UNBIHEXIUM_SERVING__MAX_REQUEST_BYTES`, `UNBIHEXIUM_SERVING__MAX_PIXELS` and `UNBIHEXIUM_SERVING__MAX_VALUES` variables where the hardware requires it.
- **Untrusted models.** Load checkpoints and ONNX files only from sources you trust. The checkpoint loader refuses pickled code, but a malicious model can still produce misleading outputs or consume excessive resources.
- **Container.** Pin the image by release tag or digest rather than `latest`, keep the non-root user and mount the model cache (`UNBIHEXIUM_CACHE`) as a dedicated volume.

## 9. Related Documents

- [security-insights.yml](security-insights.yml): machine-readable security description of the project.
- [PRIVACY.md](PRIVACY.md): what data the software processes, stores and transmits.
- [COMPLIANCE.md](COMPLIANCE.md): licence compliance and regulatory information.
- [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md): intended and prohibited uses.
- [SUPPORT.md](SUPPORT.md): support channels for questions that are not security issues.
- [docs/security/self_assessment.md](docs/security/self_assessment.md): security self-assessment of the project.
- [docs/security/](docs/security/): further notes on model integrity, secrets and supply chain security.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] FIRST. Common Vulnerability Scoring System version 4.0: Specification Document. 2023. <https://www.first.org/cvss/v4.0/specification-document>

[4] GitHub. Privately reporting a security vulnerability. 2026. <https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability>

[5] OpenSSF. OpenSSF Scorecard. 2026. <https://scorecard.dev/>

[6] ISO/IEC. ISO/IEC 29147:2018 Information technology, Security techniques, Vulnerability disclosure. 2018. <https://www.iso.org/standard/72311.html>

[7] ISO/IEC. ISO/IEC 30111:2019 Information technology, Security techniques, Vulnerability handling processes. 2019. <https://www.iso.org/standard/69725.html>

[8] OpenSSF. Security Insights Specification 2.2.0. 2025. <https://github.com/ossf/security-insights>

[9] OpenSSF. Supply-chain Levels for Software Artifacts (SLSA) Specification v1.0: Build Provenance. 2023. <https://slsa.dev/spec/v1.0/provenance>

[10] Sigstore. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

<!--
=============================================================================
End of file SECURITY.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
