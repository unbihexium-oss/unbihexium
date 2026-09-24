<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : ROADMAP.md
Title       : Roadmap
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Roadmap

| Field | Value |
| --- | --- |
| Document | UBX-DOC-ROADMAP |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Planned work on the main branch after Unbihexium 1.0.1 |

## Abstract

This document lists the work the Unbihexium project intends to do next, and why. Every item addresses a gap that can be verified in the repository as of the review date; the evidence is named with each item. The roadmap is written for users deciding whether the library fits their needs, for contributors looking for work that the maintainer will welcome, and for funders and auditors assessing the state of the project. It is a statement of intent, not a commitment: the project has not committed to dates for any item, and none are given, except where a date is fixed by an external schedule such as the Python release calendar.

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Current state](#2-current-state)
- [3. Release and supply chain](#3-release-and-supply-chain)
- [4. Model zoo](#4-model-zoo)
- [5. Examples and documentation](#5-examples-and-documentation)
- [6. Project sustainability and governance](#6-project-sustainability-and-governance)
- [7. Platform support](#7-platform-support)
- [8. Quality assurance](#8-quality-assurance)
- [9. Proposing changes to the roadmap](#9-proposing-changes-to-the-roadmap)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

The roadmap covers the library, the command line interface, the REST service, the model zoo, the examples and documentation, the release pipeline and the governance of the project. Work that is already merged is recorded in [CHANGELOG.md](CHANGELOG.md) and is not repeated here.

### 1.2 Conventions

Items are grouped by area, not by priority. Each item states the gap, the evidence in the repository, and the intended outcome. The words "planned" and "under consideration" distinguish work the maintainer intends to do from ideas that still need a decision. The key words MUST and SHOULD, where they appear, are used as in RFC 2119 [1] and RFC 8174 [2] and refer to requirements stated in other project policies.

## 2. Current state

As of 2026-09-24:

- The latest release is 1.0.1 (2025-12-21), the only version on the Python Package Index. It predates the rewrite of the library, the model zoo and the release pipeline.
- The main branch contains a large set of unreleased changes, including breaking changes, listed under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
- The model zoo defines 520 untrained starter models (130 families in four variants); only the 28 models of the 7 spectral index families compute results without training.
- The project has one maintainer, listed in [MAINTAINERS.md](MAINTAINERS.md).

## 3. Release and supply chain

### 3.1 Release the unreleased changes (planned)

- Gap: users who install from the Python Package Index receive 1.0.1, which does not contain the rewritten packages, the trainable models or the security fixes on `main`.
- Evidence: `version = "1.0.1"` in `pyproject.toml`; the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md); the PyPI release history.
- Outcome: a new release. Because the unreleased changes break compatibility, [VERSIONING.md](VERSIONING.md) requires it to be a new major version (2.0.0). [docs/MIGRATION.md](docs/MIGRATION.md) describes the migration from 1.0.x.

### 3.2 First signed release (planned)

- Gap: the release workflow signs distributions with Sigstore, attests their build provenance and attaches SLSA provenance (`unbihexium-<tag>.intoto.jsonl`), but no release has been published since these steps were added. The GitHub releases v1.0.0 and v1.0.1 have no signatures or provenance.
- Evidence: `.github/workflows/release.yml`; the assets of the existing GitHub releases.
- Outcome: the next release carries `.sigstore.json` bundles, `SHA256SUMS.txt` and SLSA provenance, and the verification steps in [SECURITY.md](SECURITY.md) are checked against a real release.

### 3.3 Trusted publishing to PyPI (under consideration)

- Gap: the release workflow uploads to the Python Package Index with a long-lived API token (`secrets.PYPI_API_TOKEN`).
- Evidence: the "Publish to PyPI" step of `.github/workflows/release.yml`.
- Outcome: replace the token with PyPI trusted publishing through OpenID Connect [3], so that no upload credential is stored in the repository settings.

### 3.4 Security self-assessment (planned)

- Gap: no security self-assessment has been published.
- Evidence: the `security.assessments.self` entry of [security-insights.yml](security-insights.yml).
- Outcome: a self-assessment based on the threat model in `docs/architecture/security_model.md`, referenced from `security-insights.yml`.

## 4. Model zoo

### 4.1 Trained weights (planned)

- Gap: none of the learned models has been trained. Every manifest records `"trained": false` and every model card states that the starter weights produce meaningless predictions.
- Evidence: `model_zoo/manifests/*.json`, `model_zoo/cards/*.md`, `src/unbihexium/zoo/catalog.yaml`.
- Outcome: trained weights for selected families, beginning with families for which openly licensed training data exist. Each trained model is to be published with the licence and citation of its training data, the training configuration, and an evaluation on independent test data with the metrics that `unbihexium evaluate` computes. Until then, the model cards remain the authoritative statement that the models are untrained.

### 4.2 Reproducible benchmark reports (under consideration)

- Gap: `docs/benchmarks/BENCHMARKS.md` reports throughput and memory measurements of the benchmark tests from manual runs on one machine; no job publishes them regularly, and accuracy cannot be reported for untrained models.
- Evidence: `docs/benchmarks/BENCHMARKS.md`; the tests in `tests/benchmarks/`.
- Outcome: a scheduled job that runs the benchmark tests and publishes the results with the hardware and software versions, and accuracy figures for trained models only.

## 5. Examples and documentation

### 5.1 Documentation examples executed in CI (under consideration)

- Gap: the code examples and commands in the documentation were executed against the code when they were written, but no workflow runs them, so a later change of the library can make an example wrong without a failing check.
- Evidence: the workflows under `.github/workflows/`, none of which extracts code blocks from Markdown files.
- Outcome: a job that extracts the Python and shell blocks of the documentation and runs them against the current code.

## 6. Project sustainability and governance

### 6.1 A second maintainer (planned)

- Gap: one person holds all maintainer roles, including releases and security response. Reviews, releases and the handling of vulnerability reports stop when that person is unavailable.
- Evidence: [MAINTAINERS.md](MAINTAINERS.md); `core-team` in [security-insights.yml](security-insights.yml).
- Outcome: at least one further maintainer with write access and a share of the security response, appointed under the rules in [GOVERNANCE.md](GOVERNANCE.md). Contributors interested in the role are invited to start with reviewed pull requests.

### 6.2 Persistent identifiers for citation (under consideration)

- Gap: Unbihexium has no DOI, and the citation metadata record no ORCID identifier for the personal author.
- Evidence: [CITATION.cff](CITATION.cff) and [codemeta.json](codemeta.json).
- Outcome: archive each release in a repository that assigns DOIs, for example through the GitHub integration of Zenodo, and add the DOI and ORCID to the citation metadata and [CITATION.md](CITATION.md).

## 7. Platform support

### 7.1 Python 3.10 end of life (planned)

- Gap: Python 3.10 reaches its upstream end of life in October 2026 [4].
- Evidence: the Python version support policy in [VERSIONING.md](VERSIONING.md).
- Outcome: under that policy, support for Python 3.10 is removed in the first minor release after its end of life, together with the Python 3.10 specific dependency bounds and lock entries.

### 7.2 Python 3.15 (planned)

- Gap: Python 3.15 is scheduled for release in October 2026 [4] and is not yet tested.
- Evidence: the CI test matrix and the classifiers in `pyproject.toml`.
- Outcome: support for Python 3.15 once the runtime dependencies publish wheels for it, following the dependency policy in `pyproject.toml`.

## 8. Quality assurance

### 8.1 Coverage threshold (under consideration)

- Gap: coverage is measured and reported, but no threshold fails a build: the Codecov statuses are informational and no `fail_under` setting exists.
- Evidence: `codecov.yml`; the coverage configuration in `pyproject.toml`.
- Outcome: an enforced minimum for project and patch coverage once the current coverage is known to be stable.

### 8.2 Wider fuzzing (under consideration)

- Gap: the atheris fuzz targets cover the GeoJSON and STAC parsers only; other parsers of untrusted input, such as the request handling of the REST service, are not fuzzed.
- Evidence: `fuzz/fuzz_geojson.py`, `fuzz/fuzz_stac.py`.
- Outcome: further fuzz targets for input parsed from files or network requests.

## 9. Proposing changes to the roadmap

Proposals are made in an issue opened with the feature request form at <https://github.com/unbihexium-oss/unbihexium/issues>. A proposal SHOULD state the gap, the evidence and the intended outcome in the form used above. The maintainer reviews this roadmap when a release is prepared and removes items once they are recorded in [CHANGELOG.md](CHANGELOG.md).

## References

[1] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Python Packaging Authority. Publishing to PyPI with a Trusted Publisher. 2026. <https://docs.pypi.org/trusted-publishers/>

[4] Python Software Foundation. Status of Python versions, Python Developer's Guide. 2026. <https://devguide.python.org/versions/>

<!--
=============================================================================
End of file ROADMAP.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
