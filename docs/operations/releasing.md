<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/operations/releasing.md
Title       : Release Procedure
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Release Procedure

| Field | Value |
| --- | --- |
| Document | UBX-DOC-903 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch, releases built by .github/workflows/release.yml |

## Abstract

This document is the step-by-step procedure for publishing a release of Unbihexium: choosing the version number, updating every place that records it, updating the changelog, tagging, what the release workflow does after the tag is pushed, how the published artefacts are checked, and what to do when something goes wrong. It is written for the maintainer who performs a release and for auditors who want to trace a published file back to its source. The policy behind the procedure (Semantic Versioning, deprecation, supported series) is defined in [VERSIONING.md](../../VERSIONING.md), whose Section 9 this document expands; in case of conflict, VERSIONING.md prevails. Packaging uses the PEP 517 backend hatchling and the `build` frontend; no other packaging tool is involved.

## Contents

- [1. Introduction](#1-introduction)
- [2. Preconditions](#2-preconditions)
- [3. Choosing the Version Number](#3-choosing-the-version-number)
- [4. The Release Pull Request](#4-the-release-pull-request)
- [5. Tagging](#5-tagging)
- [6. The Release Workflow](#6-the-release-workflow)
- [7. Container Image](#7-container-image)
- [8. Post-Release Checks](#8-post-release-checks)
- [9. Handling Failures](#9-handling-failures)
- [10. Release History](#10-release-history)
- [References](#references)

## 1. Introduction

### 1.1 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in all capitals.

### 1.2 Roles

Releases are made by the lead maintainer, who holds administrator rights on the repository and administers the `unbihexium` project on PyPI, including its trusted publisher ([GOVERNANCE.md](../../GOVERNANCE.md)). The project has no fixed release schedule; security fixes are released as described in [vulnerability_management.md](../security/vulnerability_management.md).

### 1.3 Overview

```mermaid
sequenceDiagram
    participant M as Maintainer
    participant GH as GitHub (main)
    participant R as release.yml
    participant D as docker.yml
    participant P as PyPI
    M->>GH: Release pull request (version, changelog)
    GH-->>M: All checks pass, merge
    M->>GH: Push annotated tag vX.Y.Z
    GH->>R: Tag push starts the release job
    R->>R: Build, checksums, SBOM, attestations, Sigstore, provenance
    R->>GH: GitHub release with assets
    R->>P: Upload sdist and wheel (trusted publishing)
    GH->>D: Tag push starts the image build
    D->>GH: Push, sign and attest the image on ghcr.io
    M->>M: Verify the published artefacts
```

## 2. Preconditions

Before starting, the maintainer MUST make sure that:

1. every check on the head of `main` has passed, including the scheduled runs of the past week (Security, Container Scan, Model Zoo, Integration Tests); see [ci_cd.md](ci_cd.md);
2. the `[Unreleased]` section of [CHANGELOG.md](../../CHANGELOG.md) is complete, and every merged pull request with user-visible changes has an entry;
3. no open security advisory is waiting for this release, or, if one is, the fix is included and the advisory is ready to be published with the release;
4. the lock files are current (`make lock-check`), and the Torch Lock workflow reports no difference;
5. the trusted publisher of Section 2.1 is registered on PyPI.

### 2.1 Trusted Publisher (One-Time Set-Up)

The release workflow uploads to PyPI with trusted publishing [8] and holds no PyPI token. Before the first release that uses it, the lead maintainer MUST:

1. open <https://pypi.org/manage/project/unbihexium/settings/publishing/> and add a GitHub publisher with the owner `unbihexium-oss`, the repository `unbihexium`, the workflow name `release.yml` and the environment name `pypi`;
2. delete the repository secret `PYPI_API_TOKEN` (Settings, Secrets and variables, Actions) and revoke that token on PyPI, since no workflow uses it any more.

Both steps were completed on 2026-09-24. They are kept here for the case that the repository, the workflow file or the environment is renamed, since the trusted publisher then has to be registered again with the new names.

The GitHub environment `pypi` is created by the first run of the release job. Required reviewers and a deployment rule that allows only tags matching `v*` MAY be added to it (Settings, Environments), so that every upload waits for an approval and cannot start from a branch.

## 3. Choosing the Version Number

The new version is derived from the changes since the last tag by the rules of [VERSIONING.md](../../VERSIONING.md), Section 4: MAJOR for incompatible changes, MINOR for compatible additions and deprecations, PATCH for compatible fixes; the highest applicable increment wins. At the date of review the `[Unreleased]` section contains breaking changes (for example the relicensing to MPL-2.0 and the new return type of `read_geotiff`), so the next release MUST be 2.0.0. Pre-releases, if used, take the PEP 440 suffixes `aN`, `bN` or `rcN` [3].

The model catalogue has its own version (currently 2.0.0 in [src/unbihexium/zoo/catalog.yaml](../../src/unbihexium/zoo/catalog.yaml)), incremented independently under [VERSIONING.md](../../VERSIONING.md), Section 6. A library release does not change it unless the catalogue changed.

## 4. The Release Pull Request

### 4.1 Version Fields

The release is prepared in a pull request to `main`, from a branch such as `release/v2.0.0`, with a title such as `chore(release): 2.0.0`. It MUST update the following fields, as listed in [VERSIONING.md](../../VERSIONING.md), Section 9.1:

| File | Fields |
| --- | --- |
| [pyproject.toml](../../pyproject.toml) | `version` in `[project]`; this static value is the version of the built distributions |
| [src/unbihexium/_version.py](../../src/unbihexium/_version.py) | `__version__` and `__version_tuple__` |
| [CITATION.cff](../../CITATION.cff) | `version`, `date-released` and the version-specific `identifiers` (release URL) |
| [codemeta.json](../../codemeta.json) | `version`, `softwareVersion` and `dateModified` |
| [deploy/helm/unbihexium/Chart.yaml](../../deploy/helm/unbihexium/Chart.yaml) | `version` and `appVersion` |
| [Dockerfile](../../Dockerfile) | default of the `VERSION` build argument (see [docker.md](docker.md)) |

The first step of the release workflow after the checkout, [.github/scripts/check_release_version.py](../../.github/scripts/check_release_version.py), compares the tag with the version in `pyproject.toml`, `_version.py`, `CITATION.cff` (version and release URL) and `codemeta.json`, and stops the release before the build when any of them differs. Run it before tagging, for example `python .github/scripts/check_release_version.py v1.0.1`. The following check, run from the repository root, prints the version recorded in the four metadata files that define the package and its citation:

<!-- doc-example: skip (reads the metadata files of the repository checkout) -->
```python
import json
import re
import tomllib
from pathlib import Path

found = {
    "pyproject.toml": tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"],
    "src/unbihexium/_version.py": re.search(r'__version__ = "([^"]+)"', Path("src/unbihexium/_version.py").read_text())[1],
    "CITATION.cff": re.search(r"^version: (\S+)", Path("CITATION.cff").read_text(), re.M)[1],
    "codemeta.json": json.loads(Path("codemeta.json").read_text())["version"],
}
for place, version in found.items():
    print(f"{place:28} {version}")
print("consistent" if len(set(found.values())) == 1 else "MISMATCH")
```

On the main branch at the date of review (Python 3.11 or newer, for `tomllib`) it prints:

```text
pyproject.toml               2.0.1
src/unbihexium/_version.py   2.0.1
CITATION.cff                 2.0.1
codemeta.json                2.0.1
consistent
```

The remaining occurrences of the old version, including the Helm chart, the Dockerfile and documents that name the current release (README.md, SUPPORT.md, CITATION.md, SECURITY.md and the issue templates), are found with:

```bash
git grep -n -F "1.0.1"
```

Each hit is then either updated or left deliberately, for example in the changelog history.

### 4.2 Changelog

Following the conventions of [CHANGELOG.md](../../CHANGELOG.md), Section 1, and Keep a Changelog [4]:

1. rename `[Unreleased]` to the new version with the release date in ISO 8601 form, for example `[2.0.0] - 2026-10-01`, and add a short summary paragraph;
2. start a new, empty `[Unreleased]` section above it;
3. update the comparison links at the end of the file: `[Unreleased]` compares the new tag with `HEAD`, and the new version compares the previous tag with the new one;
4. make sure breaking changes, deprecations and security fixes are listed under their headings.

### 4.3 Related Documents

The supported versions table in [SECURITY.md](../../SECURITY.md) and the supported series in [VERSIONING.md](../../VERSIONING.md), Section 10, MUST be updated when a new MINOR or MAJOR series starts, because only the latest series receives fixes.

### 4.4 Local Checks

Before the pull request is merged, the distributions SHOULD be built and checked locally in an environment with the `build` and `twine` packages:

```bash
python -m build
python -m twine check --strict dist/*
```

Run on 2026-09-24 against the main branch, `python -m build` reported `Successfully built unbihexium-2.0.1.tar.gz and unbihexium-2.0.1-py3-none-any.whl`, and `twine check --strict` reported `PASSED` for both files. The pull request MUST pass every required check before it is merged; the Package workflow repeats these checks and installs the wheel on CPython 3.10 and 3.14.

## 5. Tagging

After the release pull request is merged, the maintainer updates the local `main`, confirms that it is the merged commit, and creates and pushes an annotated tag named `v` followed by the version:

```bash
git switch main
git pull --ff-only
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

The tag MAY instead be created on the GitHub releases page (Draft a new release, a new tag `vX.Y.Z` on `main`, Publish release), as for v2.0.0 and v2.0.1; the release workflow then adds its assets and notes to that release. Such a tag is a lightweight tag, which the release workflow accepts.

Tags MUST NOT be moved, deleted and recreated, or reused once pushed ([VERSIONING.md](../../VERSIONING.md), Section 2.3), because the signatures, attestations and PyPI files of a release are bound to the tagged commit. Only tags that match `v*` start the release workflow.

## 6. The Release Workflow

### 6.1 Steps

The tag push starts [.github/workflows/release.yml](../../.github/workflows/release.yml). Its single job `release` runs on `ubuntu-latest` with `contents: write`, `id-token: write` and `attestations: write`, and executes:

| Step | Action | Result |
| --- | --- | --- |
| Check out | `actions/checkout` at the tagged commit | Source tree |
| Set up Python | `actions/setup-python`, CPython 3.14 | Interpreter |
| Check the tag against the version | `.github/scripts/check_release_version.py` | Stops the run when the tag differs from the recorded version |
| Install build tools | `pip install --require-hashes -r .github/requirements/requirements-ci-tools.txt` | Hash-pinned `build` |
| Build package | `python -m build --no-isolation` (hatchling from the hashed tools lock) | `dist/unbihexium-<version>.tar.gz`, `dist/unbihexium-<version>-py3-none-any.whl` |
| Generate checksums | `sha256sum` over `dist/` | `checksums/SHA256SUMS.txt` |
| Install the release into a clean environment | `pip --python sbom-env/bin/python install` of `requirements.txt` (hashed) and of the wheel | Environment described by the SBOM |
| Generate the SBOM | `anchore/sbom-action` (Syft) on the environment | `sbom/unbihexium-<tag>.spdx.json` (SPDX 2.3) |
| Attest the SBOM | `actions/attest-sbom` on `dist/*` | GitHub artifact attestation with the SBOM |
| Generate attestations | `actions/attest-build-provenance` on `dist/*` | GitHub artifact attestation with SLSA provenance v1 [5] |
| Copy and sign | `sigstore/gh-action-sigstore-python` on copies in `signed/` | `<file>.sigstore.json` per distribution [6] |
| Export provenance | `jq -c '.dsseEnvelope'` on the attestation bundle | `signed/unbihexium-<tag>.intoto.jsonl` |
| Create GitHub release | `softprops/action-gh-release` with generated notes | Release with distributions, bundles, provenance, SBOM and checksums |
| Publish to PyPI | `pypa/gh-action-pypi-publish` with trusted publishing in the environment `pypi` | Files and PEP 740 attestations on <https://pypi.org/project/unbihexium/> |

The signing, the attestations and the PyPI upload use the short-lived OIDC identity of the job, so no key or upload token is stored anywhere; see [secrets_and_tokens.md](../security/secrets_and_tokens.md).

### 6.2 Release Notes

The release notes are generated by GitHub from the pull requests merged since the previous tag and grouped by label according to [.github/release.yml](../../.github/release.yml): Breaking changes, Security, New features, Bug fixes, Model zoo, Performance, Compliance and licensing, Documentation, Build, CI and containers, Dependencies, and Other changes. Pull requests labelled `ignore-for-release`, `duplicate`, `invalid` or `wontfix` are omitted. The maintainer SHOULD edit the release on GitHub afterwards to add a link to the changelog section and to highlight breaking changes.

## 7. Container Image

The same tag push starts [.github/workflows/docker.yml](../../.github/workflows/docker.yml), which pushes `ghcr.io/unbihexium-oss/unbihexium` with the tags `MAJOR.MINOR.PATCH`, `MAJOR.MINOR` and `sha-<short commit>`, stores an SPDX SBOM of the pushed image as a workflow artifact, attests its build provenance and SBOM in the registry and signs its digest with cosign in keyless mode. Building, tagging, running and verifying the image are described in [docker.md](docker.md).

## 8. Post-Release Checks

After the workflows have finished, the maintainer SHOULD:

1. confirm that the GitHub release lists the sdist, the wheel, one `.sigstore.json` per distribution, `unbihexium-<tag>.intoto.jsonl`, `unbihexium-<tag>.spdx.json` and `SHA256SUMS.txt`;
2. download the files and verify them as described in [supply_chain_security.md](../security/supply_chain_security.md), Section 9: `sha256sum --check --ignore-missing SHA256SUMS.txt`, `python -m sigstore verify github --cert-identity https://github.com/unbihexium-oss/unbihexium/.github/workflows/release.yml@refs/tags/<tag> <file>` and `gh attestation verify <file> --repo unbihexium-oss/unbihexium`;
3. compare the SHA-256 digests shown on PyPI with `SHA256SUMS.txt`; they MUST be identical, because both come from the same build;
4. install the release in a clean virtual environment with `python -m pip install unbihexium==<version>` and run `unbihexium --version` and `unbihexium info`;
5. publish any security advisory that waited for the release, as described in [vulnerability_management.md](../security/vulnerability_management.md);
6. check that the Docker workflow pushed the version tags of the image and that `cosign verify` and `gh attestation verify` succeed for it ([supply_chain_security.md](../security/supply_chain_security.md), Section 9.5).

## 9. Handling Failures

| Situation | Action |
| --- | --- |
| The workflow fails before "Create GitHub Release" | Nothing was published. Fix the cause on `main` with a pull request. Because a pushed tag must not be moved, release the fix under the next PATCH version with a new tag, and delete the unused tag only if nothing was published from it. |
| The GitHub release exists but "Publish to PyPI" failed | Find the cause in the job log (for example a trusted publisher that is missing or does not match the repository, workflow or environment). Once it is fixed, delete the assets of the existing GitHub release and re-run the job from the Actions tab. A re-run builds and signs the distributions again, so the release assets and the PyPI files then come from the same run; confirm this with item 3 of Section 8. If the cause requires a code change, release a new PATCH version instead. |
| The version in `pyproject.toml` was not bumped | The step "Check the tag against the version" fails and nothing is published. Treat as above: bump the version in a pull request and release it under a new tag. |
| A defect is found after publication | PyPI does not allow a file to be replaced. Release a fixed PATCH version; a broken release MAY be yanked on PyPI [7], which keeps it installable only for exact pins. |
| An unexpected upload appears on PyPI | Check the publishing history of the project on PyPI and the runs of release.yml. Remove or restrict the trusted publisher on PyPI first, then investigate as described in [secrets_and_tokens.md](../security/secrets_and_tokens.md). |

## 10. Release History

| Tag | Date | Published artefacts |
| --- | --- | --- |
| `v1.0.0` | 2025-12-21 | GitHub release with the distributions and `SHA256SUMS.txt`; not on PyPI |
| `v1.0.1` | 2025-12-21 | GitHub release with the distributions and `SHA256SUMS.txt`; PyPI |
| `v2.0.0` | 2026-09-24 | GitHub release with the distributions, `SHA256SUMS.txt`, Sigstore bundles, SLSA provenance and the SPDX SBOM; PyPI with PEP 740 attestations; container image pushed, but not signed or attested (see [CHANGELOG.md](../../CHANGELOG.md), 2.0.1) |
| `v2.0.1` | 2026-09-24 | As v2.0.0, and a container image signed with cosign and attested with SLSA provenance and an SPDX SBOM |

The releases v1.0.0 and v1.0.1 were built before signing and SLSA provenance assets were introduced and have neither. The files attached to the v1.0.1 GitHub release are not byte-identical to the files on PyPI, so the GitHub checksums do not match the PyPI files; PyPI downloads are verified against the digests that PyPI publishes. The first release built by the current workflow will be the first with the full set of signed artefacts. The detailed history is in [CHANGELOG.md](../../CHANGELOG.md).

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Coghlan, N. and Stufft, D. PEP 440: Version Identification and Dependency Specification. 2013. <https://peps.python.org/pep-0440/>

[4] Lacan, O. and contributors. Keep a Changelog, version 1.1.0. 2023. <https://keepachangelog.com/en/1.1.0/>

[5] OpenSSF. SLSA Specification v1.0: Build Provenance. 2023. <https://slsa.dev/spec/v1.0/provenance>

[6] Sigstore. Sigstore documentation. 2026. <https://docs.sigstore.dev/>

[7] Python Packaging Authority. PEP 592: Adding "Yank" Support to the Simple API. 2019. <https://peps.python.org/pep-0592/>

[8] Python Packaging Authority. Publishing to PyPI with a Trusted Publisher. 2026. <https://docs.pypi.org/trusted-publishers/>

<!--
=============================================================================
End of file docs/operations/releasing.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
