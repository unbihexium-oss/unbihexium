<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : VERSIONING.md
Title       : Versioning and Release Policy
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Versioning and Release Policy

| Field | Value |
| --- | --- |
| Document | UBX-DOC-VERSIONING |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x, the main branch and all future releases, including the model catalogue and the container image |

## Abstract

This document defines how Unbihexium numbers its releases, what the version number promises about compatibility, how features are deprecated, how the model catalogue is versioned, which Python versions are supported, and how a release is prepared and published. It is written for users who pin or upgrade the library, for downstream packagers, for contributors who must classify their changes, and for auditors who need to trace a published artefact to its source. The policy follows Semantic Versioning 2.0.0 [1] expressed in the version syntax of PEP 440 [2]; the release history itself is recorded in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Version numbers](#2-version-numbers)
- [3. Public interface](#3-public-interface)
- [4. Rules for incrementing the version](#4-rules-for-incrementing-the-version)
- [5. Deprecation policy](#5-deprecation-policy)
- [6. Model catalogue versioning](#6-model-catalogue-versioning)
- [7. Python version support](#7-python-version-support)
- [8. Dependency versions](#8-dependency-versions)
- [9. Release process](#9-release-process)
- [10. Supported release series](#10-supported-release-series)
- [11. Querying versions](#11-querying-versions)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

This policy covers the `unbihexium` distribution on the Python Package Index, the Git tags and GitHub releases of the repository, the container image `ghcr.io/unbihexium-oss/unbihexium`, the Helm chart under `deploy/helm/unbihexium/`, and the model catalogue `src/unbihexium/zoo/catalog.yaml`.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [3] and RFC 8174 [4] when, and only when, they appear in capitals.

## 2. Version numbers

### 2.1 Format

A release version has the form `MAJOR.MINOR.PATCH`, where each part is a non-negative integer without leading zeros, as defined by Semantic Versioning 2.0.0 [1]. The same string is a valid, normalised PEP 440 [2] version, so pip, uv and the Python Package Index interpret it identically.

### 2.2 Pre-releases

Pre-releases, when they are published, MUST use the normalised PEP 440 [2] suffixes `aN` (alpha), `bN` (beta) and `rcN` (release candidate), for example `2.0.0rc1`. Installers ignore pre-releases unless they are requested explicitly (`pip install --pre`). No pre-release has been published so far.

### 2.3 Tags

Each release MUST have an annotated Git tag named `v` followed by the version, for example `v1.0.1`. Tags MUST NOT be moved or reused once they have been pushed. The existing tags are `v1.0.0` and `v1.0.1`, both created on 2025-12-21.

### 2.4 Development versions

Between releases, the version in `pyproject.toml` stays at the last released version. Changes that are merged but not released are listed under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md). An installation from the main branch therefore reports the previous release number; cite or report such an installation together with its commit hash.

## 3. Public interface

The compatibility promises in [Section 4](#4-rules-for-incrementing-the-version) apply to the public interface, which consists of:

- the names exported in `__all__` of `unbihexium` and of its subpackages, with their documented signatures, return types and units;
- the commands and options of the `unbihexium` command line interface that appear in `unbihexium --help` and in the help of its subcommands;
- the endpoints, request schemas and response schemas of the REST service in `unbihexium.serving`;
- the model identifiers of the catalogue, and the file formats of checkpoints, exported ONNX files, manifests and model cards;
- the environment variables and configuration keys documented in `.env.example` and under `docs/`.

The following are not part of the public interface and MAY change in any release: names that start with an underscore; modules not re-exported by a package; hidden command aliases; the text of log and error messages; performance characteristics; and the numerical output of untrained starter models (see [Section 6](#6-model-catalogue-versioning)).

## 4. Rules for incrementing the version

| Part | MUST be incremented when | Examples |
| --- | --- | --- |
| MAJOR | A change breaks code, commands, requests or files that worked with the previous release | Removing a public function, changing a return type, changing default units, removing a command option, relicensing |
| MINOR | Functionality is added in a backwards compatible way, or a feature is deprecated | New spectral index, new command, new model family, new optional extra, dropping a Python version at its end of life ([Section 7](#7-python-version-support)) |
| PATCH | Only backwards compatible bug fixes and security fixes are made | Correcting a formula, fixing a crash on malformed input, raising a dependency lower bound to exclude a vulnerable release |

When MAJOR is incremented, MINOR and PATCH are reset to zero; when MINOR is incremented, PATCH is reset to zero. A release that contains changes of several kinds takes the highest applicable increment. The `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) currently contains breaking changes (for example the relicensing to MPL-2.0 and the new return type of `read_geotiff`), so the next release MUST be a new major version.

## 5. Deprecation policy

1. A feature SHOULD be deprecated before it is removed. Deprecation happens in a MINOR or MAJOR release and MUST be recorded under Deprecated in [CHANGELOG.md](CHANGELOG.md).
2. A deprecated Python API SHOULD emit a `DeprecationWarning` that names the replacement; a deprecated command SHOULD print a notice to standard error.
3. A deprecated feature SHOULD remain available for at least two MINOR releases and MUST NOT be removed before the next MAJOR release.
4. Security fixes MAY remove or restrict a feature without a deprecation period when no safe alternative exists; the changelog MUST explain why.

At the time of review no public API emits a deprecation warning. The commands `unbihexium zoo download` and `unbihexium infer` are hidden aliases of `zoo build` and `predict`, kept for compatibility with 1.0.x scripts.

## 6. Model catalogue versioning

### 6.1 Identifiers

A model is identified by its family and variant, `<family>_<variant>`, for example `aircraft_detector_base`. The variant is one of `tiny`, `base`, `large` and `mega`. Model identifiers do not contain a version number.

### 6.2 Catalogue version

The catalogue has its own `MAJOR.MINOR.PATCH` version, stored as `version` in `src/unbihexium/zoo/catalog.yaml` (currently 2.0.0) and recorded in every manifest under `model_zoo/manifests/`, in `src/unbihexium/zoo/digests.json` and in every catalogue entry. It is independent of the library version. Changes to the catalogue SHOULD increment it as follows:

- MAJOR: an existing model identifier changes its architecture, inputs, outputs, units or starter weights digest, or is removed;
- MINOR: model families are added;
- PATCH: descriptions, suitable data or other documentation fields change without affecting any model.

### 6.3 Starter weights

The 520 models of the catalogue are untrained starter models: each has a complete architecture and deterministic starter weights, derived from the model identifier, whose SHA-256 digests are published in `digests.json` and verified on load. Only the 28 models of the 7 spectral index families compute their published formulas without training. Their predictions are not versioned results, and the weights of models that users train themselves are outside this policy: users SHOULD record the catalogue version, the model identifier and the checkpoint digest with every result.

## 7. Python version support

### 7.1 Policy

Unbihexium supports CPython versions that are maintained upstream and for which all runtime dependencies publish wheels. Support for a Python version is dropped in the first MINOR release after that version reaches its upstream end of life. A new Python version is added when the runtime dependencies publish wheels for it, and every supported version MUST be tested in CI. Support is declared through `requires-python` and the classifiers in `pyproject.toml`.

### 7.2 Current support

Unbihexium supports CPython 3.10 to 3.14 on Linux, macOS and Windows. The dates below are those of the Python release schedule [5]; end-of-life dates are planned dates set by the Python core developers.

| Python | First release | End of bugfix releases | End of life | Unbihexium status |
| --- | --- | --- | --- | --- |
| 3.14 | 2025-10-07 | 2027-10 | 2030-10 | Supported and tested |
| 3.13 | 2024-10-07 | 2026-10 | 2029-10 | Supported and tested |
| 3.12 | 2023-10-02 | 2025-04 | 2028-10 | Supported and tested |
| 3.11 | 2022-10-24 | 2024-04 | 2027-10 | Supported and tested |
| 3.10 | 2021-10-04 | 2023-04 | 2026-10 | Supported and tested; dropped in the first minor release after its end of life |

Python 3.15 is not yet supported; it will be added under the rule in [Section 7.1](#71-policy).

## 8. Dependency versions

Runtime and optional dependencies are declared in `pyproject.toml` with lower bounds and without upper bounds. The lower bounds follow three rules stated in that file: compiled packages start at the first release with CPython 3.14 wheels, pure Python packages at the release that was current when Python 3.14 was released, and every package at the first release without known vulnerabilities; the highest of these wins. Where a newer release dropped Python 3.10, an environment marker keeps a separate lower bound for Python 3.10.

Raising a lower bound is a PATCH change when it excludes a vulnerable or broken release and a MINOR change otherwise. Adding a required runtime dependency is a MINOR change. Reproducible environments are provided by the lock files `requirements.txt` (runtime with the `onnx` and `serving` extras), `requirements-dev.txt` (all extras) and the hashed CI locks under `.github/requirements/`; they are regenerated with `make lock` and are not part of the compatibility promise.

## 9. Release process

### 9.1 Preparation

A release is prepared in a pull request to `main` that MUST:

1. set the new version in `pyproject.toml` and in `src/unbihexium/_version.py` (`__version__` and `__version_tuple__`);
2. set `version`, `date-released` and the version-specific `identifiers` in `CITATION.cff`, and `version`, `softwareVersion` and `dateModified` in `codemeta.json`;
3. set `version` and `appVersion` in `deploy/helm/unbihexium/Chart.yaml` and the default `VERSION` build argument in the `Dockerfile`;
4. move the `[Unreleased]` entries of [CHANGELOG.md](CHANGELOG.md) into a new version section with the release date and update the comparison links;
5. pass every required CI check.

### 9.2 Publication

After the pull request is merged, the maintainer creates and pushes the annotated tag, for example `git tag -a v2.0.0 -m "Release v2.0.0"` followed by `git push origin v2.0.0`. The tag starts `.github/workflows/release.yml`, which:

1. builds the source distribution and the wheel with `python -m build --no-isolation`, with the build frontend and the backend hatchling from the hashed tools lock;
2. writes `SHA256SUMS.txt` for the distributions;
3. creates GitHub artifact attestations of build provenance for every distribution;
4. signs every distribution with Sigstore and attaches the `.sigstore.json` bundles;
5. attaches the SLSA provenance of the attestation as `unbihexium-<tag>.intoto.jsonl`;
6. creates the GitHub release with notes compiled from the merged pull requests, grouped by the categories in `.github/release.yml`;
7. uploads the distributions to the Python Package Index.

The tag also starts `.github/workflows/docker.yml`, which pushes the container image with the tags `MAJOR.MINOR.PATCH` and `MAJOR.MINOR` and a commit tag, and records an SPDX software bill of materials of the pushed image. Releases v1.0.0 and v1.0.1 were published before signing and SLSA provenance assets were introduced and have neither. How to verify a signed release is described in [SECURITY.md](SECURITY.md); the attestations can be checked with `gh attestation verify <file> --repo unbihexium-oss/unbihexium`.

### 9.3 Release cadence

The project has no fixed release schedule. Releases are made when the maintainer judges the accumulated changes ready; security fixes are released as described in [Section 10](#10-supported-release-series).

## 10. Supported release series

Only the latest release series receives fixes. Fixes, including security fixes, are made on `main` first and released as a new PATCH version of the latest series; earlier patch versions are not patched separately. The latest release is 1.0.1. Vulnerabilities are reported and handled as described in [SECURITY.md](SECURITY.md).

## 11. Querying versions

The installed library version, the catalogue version and the version of a catalogue entry can be read as follows; `get_model` returns catalogue metadata and does not build or download a model.

```python
import unbihexium
from unbihexium.zoo import catalog_version, get_model

print(unbihexium.__version__)        # Library version, for example "1.0.1".
print(unbihexium.__version_tuple__)  # The same version as a tuple, (1, 0, 1).
print(catalog_version())             # Model catalogue version, "2.0.0".

entry = get_model("aircraft_detector_base")  # Catalogue entry of one model.
print(entry.model_id, entry.version, entry.weights_digest[:16])
```

From the command line, `unbihexium --version` prints the library version and `unbihexium info` prints the library version together with the catalogue version.

## References

[1] Tom Preston-Werner. Semantic Versioning 2.0.0. 2013. <https://semver.org/spec/v2.0.0.html>

[2] Nick Coghlan and Donald Stufft. PEP 440: Version Identification and Dependency Specification. Python Software Foundation, 2013. <https://peps.python.org/pep-0440/>

[3] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[4] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[5] Python Software Foundation. Status of Python versions, Python Developer's Guide. 2026. <https://devguide.python.org/versions/>

<!--
=============================================================================
End of file VERSIONING.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
