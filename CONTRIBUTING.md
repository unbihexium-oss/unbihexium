<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : CONTRIBUTING.md
Title       : Contributing to Unbihexium
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Contributing to Unbihexium

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CONTRIBUTING |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This guide describes how to contribute code, documentation, tests and model zoo changes to Unbihexium, and what a contribution must satisfy before it is merged. It is written for first-time and regular contributors and for the maintainers who review their work. It covers the development environment, the coding and documentation styles that the repository checks automatically, the text policy, the test tiers and fuzz targets, the dependency lock files, the commit message and pull request conventions, the Developer Certificate of Origin and the review process. Every rule stated here corresponds to a script, workflow or configuration file in the repository, which is named so that contributors can run the same check locally.

## Contents

- [1. Introduction](#1-introduction)
- [2. Ways to contribute](#2-ways-to-contribute)
- [3. Development environment](#3-development-environment)
- [4. Branches and workflow](#4-branches-and-workflow)
- [5. Code and documentation standards](#5-code-and-documentation-standards)
- [6. Tests](#6-tests)
- [7. Dependencies and lock files](#7-dependencies-and-lock-files)
- [8. Model zoo contributions](#8-model-zoo-contributions)
- [9. Commit messages and pull request titles](#9-commit-messages-and-pull-request-titles)
- [10. Pull requests](#10-pull-requests)
- [11. Review and merge](#11-review-and-merge)
- [12. Changelog and releases](#12-changelog-and-releases)
- [13. Licensing of contributions](#13-licensing-of-contributions)
- [References](#references)

## 1. Introduction

### 1.1 Scope

This guide applies to every change proposed to the repository `unbihexium-oss/unbihexium`: source code under `src/unbihexium/`, tests, fuzz targets, examples, documentation, model zoo metadata, packaging, container and continuous integration (CI) configuration. It does not govern how the software is used; see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) and [COMPLIANCE.md](COMPLIANCE.md) for that.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals. Commands are given for a POSIX shell and are run from the repository root.

### 1.3 Related documents

| Document | Purpose |
| --- | --- |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Standards of behaviour in all project spaces |
| [GOVERNANCE.md](GOVERNANCE.md) | Roles, decision making and how maintainers are added |
| [MAINTAINERS.md](MAINTAINERS.md) | Current maintainers and their areas |
| [SECURITY.md](SECURITY.md) | Private reporting of vulnerabilities |
| [SUPPORT.md](SUPPORT.md) | Support channels |
| [VERSIONING.md](VERSIONING.md) | Versioning and compatibility policy |
| [CHANGELOG.md](CHANGELOG.md) | Record of notable changes |
| [AUTHORS.md](AUTHORS.md) | Authors and contributors |

## 2. Ways to contribute

### 2.1 Issues

Issues are opened through the issue forms in `.github/ISSUE_TEMPLATE/`: bug report, feature request, documentation, model zoo, performance, compliance, build and packaging, and question. Blank issues are disabled (`.github/ISSUE_TEMPLATE/config.yml`). Contributors SHOULD search existing issues before opening a new one and SHOULD open an issue to discuss substantial changes before writing code, so that design questions are settled before review.

### 2.2 Security vulnerabilities

Vulnerabilities MUST NOT be reported in public issues, pull requests or discussions. Report them privately through a GitHub private security advisory at <https://github.com/unbihexium-oss/unbihexium/security/advisories/new> or by e-mail to <yunus.z.imanov@helsinki.fi>, as described in [SECURITY.md](SECURITY.md).

### 2.3 Kinds of contribution

Contributions of every kind are welcome: bug fixes, new processing functions, tests, fuzz targets and corpus entries, documentation, example notebooks and scripts, model zoo metadata, packaging and CI improvements. Issues labelled `good first issue` or `help wanted` are suitable starting points.

## 3. Development environment

### 3.1 Prerequisites

| Tool | Required for | Notes |
| --- | --- | --- |
| Git | All work | Fork and clone the repository |
| Git LFS | Model weights | Needed only for work that loads the model zoo weights (`make validate`) |
| CPython 3.10 to 3.14 | All Python work | The package declares `requires-python = ">=3.10"`; CI tests 3.10, 3.11, 3.12, 3.13 and 3.14 |
| GNU Make | Convenience targets | Every target is a thin wrapper; the commands can be run directly |
| uv | Regenerating lock files | Only for `make lock` and `make lock-check` (Section 7) |
| Node.js with `npx` | Markdown linting | `make md-lint` runs markdownlint-cli 0.49.1 |
| Docker | Container work | Only for the `docker-*` targets |

### 3.2 Setting up

Fork the repository on GitHub, clone the fork and create a virtual environment:

```bash
git clone https://github.com/<your-account>/unbihexium.git
cd unbihexium
python -m venv .venv
. .venv/bin/activate
make install-dev
```

`make install-dev` runs three steps:

1. `pip install -r requirements-dev.txt` installs the locked development environment: the runtime dependencies and the `all` extra (ONNX Runtime, PyTorch, serving, Dask, Ray, Zarr, NetCDF, STAC, Parquet, test and development tools) at exact versions. GPU packages are excluded; install them with `pip install ".[gpu]"` when needed.
2. `pip install --no-deps -e .` installs the package in editable mode without resolving the dependencies again.
3. `pre-commit install` installs the Git hooks defined in `.pre-commit-config.yaml`.

`make verify` prints the installed version and the number of registered models and is a quick check that the installation works.

### 3.3 Make targets

`make help` lists every target. The targets most relevant to contributors are:

| Target | Command it runs |
| --- | --- |
| `make test` | `pytest tests/` |
| `make test-fast` | `pytest tests/unit -n auto -m "not slow and not gpu"` |
| `make test-cov` | `pytest tests/` with branch coverage and an XML report |
| `make lint` | `ruff check src/ tests/` |
| `make format` | `ruff format` and `ruff check --fix` on `src/` and `tests/` |
| `make format-check` | `ruff format --check src/ tests/` |
| `make type-check` | `pyright src/` |
| `make security` | Bandit on `src/` and pip-audit on both lock files |
| `make licence` | REUSE lint and `.github/scripts/check_license_headers.py` |
| `make text-policy` | `.github/scripts/check_text_policy.py` |
| `make md-lint` | markdownlint with `.markdownlint.yaml` |
| `make notebooks` | Notebook format and absence of outputs |
| `make model-zoo` | `.github/scripts/check_model_zoo.py` |
| `make check` | Lint, format check, type check, tests, licence, text policy, YAML lint, notebooks and model zoo |
| `make pre-commit` | Every pre-commit hook on all files |
| `make check-dist` | Builds the sdist and wheel and runs `twine check --strict` |

`tox.ini` provides the same checks in isolated environments: a plain `tox` runs the tests on every installed interpreter from 3.10 to 3.14 and the `lint`, `type` and `licence` environments; `tox -e lowest` runs the tests against the lowest allowed versions of the direct dependencies.

## 4. Branches and workflow

1. Create a topic branch in your fork from the current `main` branch. Branch names are free; a prefix that matches the commit type (for example `fix/geojson-bounds` or `docs/contributing`) is RECOMMENDED.
2. Make focused changes. A pull request SHOULD address one topic; unrelated changes SHOULD be proposed separately.
3. Run the relevant local checks (Sections 5 and 6) before pushing.
4. Open a pull request against `main`. The CI workflows in `.github/workflows/` run on pull requests that target `main`, so a pull request against another branch receives no checks.
5. Keep the branch up to date with `main` while the pull request is open, by merging or rebasing.

The maintainer integrates work on the `dev` branch and merges it into `main` through pull requests; contributors do not need to target `dev`.

## 5. Code and documentation standards

### 5.1 Python code

- Code MUST be compatible with CPython 3.10 to 3.14. Ruff is configured with `target-version = "py310"` and a line length of 100 characters.
- Code MUST pass `ruff check` and `ruff format --check` with the configuration in `pyproject.toml`. CI runs both on `src/`; the Make targets also cover `tests/`.
- Public functions and classes SHOULD carry complete type annotations. Pyright is configured in strict mode in `pyproject.toml`; CI currently runs it at the basic level and reports the result without failing the job, so new code SHOULD NOT add type errors.
- Code SHOULD NOT introduce findings of Bandit, which runs with the configuration in `pyproject.toml`.
- New modules MUST follow the documentation style of Section 5.2.

### 5.2 Python documentation style

Python files are documented with `#` comments instead of docstrings. The layout is enforced by `.github/scripts/check_python_style.py` in the Text Policy workflow (job "Python documentation style") and by the `python-style` pre-commit hook, for the files and directories listed in `STYLE_ROOTS` of that script. The list covers the whole package `src/unbihexium/`, `fuzz/`, `scripts/`, `.github/scripts/`, `examples/scripts/`, `examples/serving/` and the test modules; a new file under one of these paths is checked automatically.

A conforming file has four parts:

1. **Licence notice.** The MPL-2.0 Exhibit A notice in the first three lines.
2. **Header block.** Between two rules of `=` characters, the fields `Project`, `Module`, `Title`, `Author`, `Affiliation`, `Copyright`, `Licence` and `Python`, in the layout `# Name<padding>: value`. The `Module` field MUST be the repository path of the file. The header continues with an `Abstract` section and, where useful, `Usage`, `Method`, `References` and `Exit status` sections.
3. **Comment coverage.** Every line that holds code MUST have a comment on the same line or on the line directly above it. Lines that hold only closing brackets, and continuation lines of a string inside brackets, count as punctuation and need no comment. A comment above the first decorator covers the whole decorator chain and the definition.
4. **Footer.** A closing block that starts with `# End of module <path>`.

Docstrings MUST NOT be used in modules, classes or functions; the checker reports each one. Where a library reads docstrings at runtime, the text is passed explicitly instead: `help=` for command line options, `description=` for FastAPI routes and `json_schema_extra` for Pydantic models, so that the command line help and the OpenAPI document keep their text.

A minimal conforming module, stored as `src/unbihexium/utils/example.py`:

```python
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/example.py
# Title       : Example of the Python documentation style
# Author      : Your Name <you@example.org>
# Affiliation : Your institution
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Rescales an array linearly to the closed interval [0, 1].
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Array computations.
import numpy as np


# Rescale an array to [0, 1]; a constant array becomes all zeros.
def rescale(values: np.ndarray) -> np.ndarray:
    # Smallest and largest finite value.
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    # A constant array has no range to divide by.
    if high == low:
        # Return zeros of the same shape.
        return np.zeros_like(values, dtype=np.float64)
    # Linear rescaling.
    return (values - low) / (high - low)

# =============================================================================
# End of module src/unbihexium/utils/example.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
```

Check one or more files with:

```bash
python .github/scripts/check_python_style.py src/unbihexium/utils/example.py
```

Without arguments the script checks every tracked Python file under `STYLE_ROOTS` and exits with status 1 if it finds a problem.

### 5.3 Configuration file documentation style

`.github/scripts/check_config_style.py` applies the same style to every other tracked text file that can hold `#` comments: YAML (including `CITATION.cff`), TOML, INI, the Dockerfile, the Makefile, shell scripts, pip requirements and lock files, the model checksum list and the Git, Docker and editor configuration files. It runs in the Text Policy workflow and as the `config-style` pre-commit hook. A conforming file has:

1. The MPL-2.0 notice at the top. Only a shebang (`#!`) or a Dockerfile parser directive (`# syntax=`) may precede it.
2. A header block with the fields `Project`, `File`, `Title`, `Author`, `Affiliation`, `Copyright`, `Licence` and `Format`, followed by an `Abstract` section. The `File` field MUST be the repository path of the file.
3. A comment for every line that holds content, on the same line or on the line directly above it.
4. A footer block containing `# End of file <path>` and `# Cite the project as described in CITATION.cff.` within the last six lines.

Same-line comments are accepted only in formats whose parsers strip them: YAML, TOML, shell, pip requirements and `model_zoo/checksums.txt`. In the other formats (INI, Dockerfile, Makefile and the Git, Docker and editor files) a trailing `#` would become part of the value, so the comment MUST stand on the line above. Continuation lines after a trailing backslash, lines with only closing brackets, the contents of YAML block scalars and TOML multi-line strings (except shell scripts under a `run` key in workflows), and INI value continuations are covered by the comment of the line that opens them.

JSON files, NumPy arrays, the empty `py.typed` marker, Markdown and notebooks (which have their own rules), the fuzz corpus in `fuzz/corpus/` and the verbatim licence and notice texts (`LICENSE.txt`, `LICENSES/`, `NOTICE`) are not checked.

A minimal conforming YAML file, stored as `examples/settings.yml`:

```yaml
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# File        : examples/settings.yml
# Title       : Example of the configuration file documentation style
# Author      : Your Name <you@example.org>
# Affiliation : Your institution
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Format      : YAML 1.2, read by PyYAML
# =============================================================================
#
# Abstract
# --------
# Tile size and overlap of an example processing run.
# =============================================================================

tiling:  # Settings of the tiler.
  size: 512  # Tile edge length in pixels.
  # Overlap between neighbouring tiles in pixels.
  overlap: 32

# =============================================================================
# End of file examples/settings.yml
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
```

YAML files are additionally linted with yamllint (`.yamllint.yml`), and workflows with actionlint and shellcheck (Workflow Lint workflow).

### 5.4 Markdown and notebooks

- Markdown MUST pass markdownlint with `.markdownlint.yaml` (`make md-lint`; Markdown workflow). The root documents follow a common layout: an HTML comment with the licence notice and header block, one H1 heading, a document control table, an abstract, a contents list, numbered sections, references and a closing HTML comment.
- Relative links MUST point to files that exist. The Links workflow checks links on a schedule with lychee (`.github/lychee.toml`).
- Example notebooks in `examples/notebooks/` MUST be valid notebooks and MUST NOT contain outputs (`make notebooks`; Notebooks workflow).

### 5.5 Text policy

`.github/scripts/check_text_policy.py` checks every tracked text file (except `LICENSE.txt`) and fails on:

- emojis and pictographic symbols, including the emoji variation selector and the zero width joiner;
- the em dash (U+2014) and the horizontal bar (U+2015);
- letters used only in Turkish (U+011E, U+011F, U+0130, U+0131, U+015E, U+015F), because the repository is written in English only.

Contributions MUST be written in English. Plain ASCII punctuation is RECOMMENDED: use commas, colons, parentheses or the word "to" instead of dashes between words or in ranges. The same character rules apply to commit messages (Section 9). Run the check with `make text-policy`.

### 5.6 Licence notices and REUSE

Every new source file MUST carry the MPL-2.0 Exhibit A notice. `.github/scripts/check_license_headers.py` checks the notices and the REUSE tool [3] checks that every file has licensing information (`REUSE.toml`, `LICENSES/`); both run in the License Compliance workflow and through `make licence`. Code copied from another project MUST keep its original notices, MUST be under a licence compatible with MPL-2.0 and MUST be recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 6. Tests

### 6.1 Test tiers

| Tier | Location | Run locally | Run in CI |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | `make test-fast` or `pytest tests/unit` | CI workflow, job "Test", Python 3.10 to 3.14 |
| Integration | `tests/integration/` | `pytest tests/integration` | Integration Tests workflow, Python 3.10 to 3.14 |
| End-to-end | `tests/e2e/` | `pytest tests/e2e` | Integration Tests workflow, Python 3.14 |
| Benchmarks | `tests/benchmarks/` | `pytest tests/benchmarks` | Part of `pytest tests/` in the CI and Coverage workflows |
| REST API smoke test | `src/unbihexium/serving/` | `make docker-api` or `uvicorn unbihexium.serving.app:app` | Integration Tests workflow, job "API Integration" |

The CI job "Test" runs the whole suite (`pytest tests/`); the Coverage workflow runs it once more under pytest-cov and uploads the report to Codecov. Coverage statuses are informational (`codecov.yml`) and do not block a merge, but a change SHOULD NOT reduce coverage without a reason given in the pull request.

### 6.2 Writing tests

- Bug fixes MUST include a test that fails before the fix and passes after it. New features MUST include tests.
- Tests MUST NOT require network access, credentials or a GPU unless they are marked accordingly. The registered markers are `slow`, `gpu` and `integration` (`pyproject.toml`); pytest runs with `--strict-markers`, so other markers are rejected.
- Tests that depend on an optional extra SHOULD skip with `pytest.mark.skipif` when the extra is not installed, as `tests/unit/test_io.py` does for rasterio, GeoPandas, pyproj and Zarr.
- Test data MUST be synthetic or openly licensed. Small fixtures live in `tests/fixtures/` (see its README); tests SHOULD create larger inputs in temporary directories.
- Tests that touch the model zoo SHOULD set the environment variable `UNBIHEXIUM_CACHE` to a temporary directory and use the `tiny` variants. The 520 models of the zoo (130 families in four variants: tiny, base, large and mega) are untrained starter models with deterministic weights, except the 28 models of the 7 spectral index families, which compute exact formulas. Tests MUST NOT assert accuracy figures for the starter models.

### 6.3 Fuzz targets

`fuzz/` holds atheris [4] fuzz targets for the parsers that read untrusted input:

| Target | Functions under test |
| --- | --- |
| `fuzz/fuzz_geojson.py` | GeoJSON validation, bounds and ring orientation in `unbihexium.io.geojson` |
| `fuzz/fuzz_stac.py` | STAC item and time parsing |

Each target starts from its seed corpus in `fuzz/corpus/<target>/`. The Fuzzing workflow runs each target for two minutes on pull requests and pushes that change `src/unbihexium/io/`, `fuzz/`, the fuzzing lock file or the workflow, and for twenty minutes weekly. A crash fails the job and uploads the crashing input as an artifact. To fix a crash, add the input to `fuzz/corpus/<target>/` as `regression_<name>` in the same pull request as the fix; `tests/unit/test_fuzz_targets.py` replays every corpus file on each test run, so the regression stays covered without atheris.

Run a target locally with a Python version for which atheris publishes wheels (CI uses 3.12):

```bash
python -m pip install --require-hashes -r .github/requirements/requirements-ci-fuzz.txt
python -m pip install --no-deps -e .
cp -r fuzz/corpus/stac /tmp/stac-corpus
python fuzz/fuzz_stac.py /tmp/stac-corpus -max_total_time=60
```

New parsers of untrusted input SHOULD come with a fuzz target, a seed corpus and a matrix entry in `.github/workflows/fuzz.yml`.

## 7. Dependencies and lock files

### 7.1 Declaring dependencies

Dependencies are declared with version ranges in `pyproject.toml`: runtime dependencies under `[project]` and optional ones in the extras `onnx`, `torch`, `gpu`, `serving`, `dask`, `ray`, `zarr`, `netcdf`, `stac`, `parquet`, `test`, `dev` and `all`. A new dependency MUST be justified in the pull request (template questions Q25 and Q26) with its version and SPDX licence identifier. Its licence MUST be allowed by `.github/dependency-review-config.yml`, which denies GPL licences for every dependency scope; the Security and License Compliance workflows enforce this.

### 7.2 Lock files

| File | Content | Hashes |
| --- | --- | --- |
| `requirements.txt` | Runtime dependencies with the `onnx` and `serving` extras | Yes |
| `requirements-dev.txt` | Runtime dependencies with the `all` extra, for development | No |
| `.github/requirements/requirements-ci-test.txt` | Test environment of the CI, from `requirements-ci-test.in` | Yes |
| `.github/requirements/requirements-ci-tools.txt` | Linters and CI tools, from `requirements-ci-tools.in` | Yes |
| `.github/requirements/requirements-ci-fuzz.txt` | atheris and NumPy for fuzzing, from `requirements-ci-fuzz.in` | Yes |
| `.github/requirements/requirements-ci-torch.txt` | CPU build of PyTorch without its dependencies, from `requirements-ci-torch.in` | Yes |

Each lock file covers CPython 3.10 to 3.14 on Linux, macOS and Windows through environment markers. CI workflows install with `pip install --require-hashes`.

### 7.3 Regenerating lock files

Lock files MUST NOT be edited by hand. After changing a version range in `pyproject.toml` or a `.in` file:

```bash
make lock        # regenerate the five lock files compiled with uv
make lock-check  # fail if any of them is out of date
```

`make lock` compiles `requirements.txt`, `requirements-dev.txt` and the test, tools and fuzz locks with `uv pip compile --universal --python-version 3.10`, and `scripts/merge_lock.py` keeps the documented header and footer of each file. Review the diff before committing.

### 7.4 The Torch Lock workflow

The PyTorch lock is not produced by `make lock`, because the index it resolves against (`https://download.pytorch.org/whl/cpu`) is not reachable from every environment. The Torch Lock workflow (`.github/workflows/torch-lock.yml`) compiles `.github/requirements/requirements-ci-torch.txt` from `requirements-ci-torch.in`, prints the complete file in the job log, uploads it as the artifact `requirements-ci-torch` and fails when it differs from the committed file. It runs on demand, weekly on Thursday at 05:00 UTC and on pushes that change the input, the lock or the workflow. To update the lock, run the workflow from the Actions tab, copy the printed file into the repository and commit it.

### 7.5 Automated updates

Dependabot (`.github/dependabot.yml`) proposes weekly updates of the pip dependencies and the GitHub Actions and monthly updates of the Docker base image. All GitHub Actions are pinned to full commit SHAs with the version in a trailing comment; new workflow steps MUST follow the same convention.

## 8. Model zoo contributions

The model zoo is described by `src/unbihexium/zoo/catalog.yaml`. The digests, inventory, capability map, manifests, model cards and checksums in `model_zoo/` are generated from it:

```bash
python -m unbihexium.zoo.sync --check   # report generated files that are out of date
python .github/scripts/check_model_zoo.py
```

A change to the catalogue MUST include the regenerated files, and `make model-zoo` MUST pass. Model cards MUST describe the starter models as untrained (except the spectral index models, which compute exact formulas) and MUST NOT claim accuracy that has not been measured. Weight files are stored in Git LFS (`.gitattributes`). Contributed trained weights MUST come with a statement of the training data, its licence and the training procedure (template questions Q58 to Q63).

## 9. Commit messages and pull request titles

### 9.1 Format

Commit messages and pull request titles MUST follow the Conventional Commits 1.0.0 [5] header format:

```text
type(scope): description
```

- `type` is one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert` and `deps`.
- `scope` is optional and names the affected area, usually a subpackage (`io`, `zoo`, `cli`, `serving`) or a component (`docker`, `release`, `deps`).
- `description` is written in the imperative mood, starts with a lowercase letter and has no final full stop.
- A breaking change is marked with `!` after the type or scope, or with a `BREAKING CHANGE:` footer, and MUST be described in the body.

Examples from the history of the repository:

```text
fix(io): reject malformed GeoJSON and STAC input with ValueError
test(fuzz): add atheris fuzz targets for the GeoJSON and STAC parsers
ci(release): sign the distributions with Sigstore and attach the bundles
build(deps): refresh the development lock (virtualenv 21.11.1)
```

The body, separated from the header by a blank line, SHOULD explain what changed and why.

### 9.2 Automated checks

- **PR Title** (`.github/workflows/pr-title.yml`) validates the pull request title against the allowed types with `amannn/action-semantic-pull-request`. The scope is optional. The title matters because it becomes the commit message when a pull request is squash-merged and it feeds the release notes.
- **Text Policy**, job "Commit messages" (`.github/workflows/text-policy.yml`), scans the message of every commit in the pull request and fails on emojis, the em dash, the horizontal bar and the Turkish-specific letters listed in Section 5.5.

## 10. Pull requests

### 10.1 Template

Every pull request uses `.github/PULL_REQUEST_TEMPLATE.md`. Every question MUST be answered: closed questions are answered by ticking every applicable option, or "Other:" with the answer on the same line; open questions are answered in full, with `N/A` only when a question genuinely does not apply. The template has the following parts:

| Part | Questions | Content |
| --- | --- | --- |
| Summary of the change | Q1 to Q8 | Summary, motivation, closed issues, type of change, modules, capability domains, design notes and alternatives |
| Compatibility | Q9 to Q12 | Backward compatibility under [VERSIONING.md](VERSIONING.md), migration notes, public API changes and version increment |
| Testing | Q13 to Q18 | Tests added, commands and results, Python versions, local checks, coverage and manual testing |
| Impact | Q19 to Q32 | Performance, security, documentation, changelog, licence headers, dependencies, model zoo, data files, CI, commit messages, reviewer notes and contribution terms |
| Submitter and context | Q33 to Q38 | Role, organisation type, jurisdiction and related work, used for triage only |
| Environment | Q39 to Q51 | Versions, installation, platform, accelerator and deployment context |
| Declarations | Q52 to Q68 | Provenance of the code, libraries, models and data, their licences, personal data and geographic sensitivity |
| AI usage declaration | Q69 to Q78 | Whether and how AI tools were used, which parts they produced and how the output was verified |
| Regulatory classification and confirmations | Q79 to Q96 | Classification and confirmations regarding data protection, the EU AI Act, cyber security, export control, geospatial data, other privacy laws and the licence, with links to the articles listed in [COMPLIANCE.md](COMPLIANCE.md) |
| Closing | Q97 and Q98 | Additional context and final confirmation |

The regulatory questions record the contributor's own assessment. They are not legal advice, and neither the template nor this guide replaces advice from a qualified professional.

### 10.2 AI usage declaration

Contributions prepared with the help of AI tools are accepted when they are declared in questions Q69 to Q78. The contributor MUST review, understand and verify all such content, MUST NOT use such tools to fabricate logs, benchmark or test results or citations, and MUST NOT enter personal or confidential data into them without a lawful basis and permission. Pull requests with such content MAY be labelled `ai-assisted`.

### 10.3 Developer Certificate of Origin

By ticking the first statement of question Q32, the contributor certifies that the contribution satisfies the Developer Certificate of Origin (DCO) 1.1 [6]: in short, that they wrote it or otherwise have the right to submit it under the licence of the project, and that they understand the contribution and its record are public and kept indefinitely. Contributors SHOULD also add a `Signed-off-by` trailer to each commit with `git commit -s`, which records the same certification in the history. No workflow currently checks for the trailer; the ticked statement in the pull request is required.

### 10.4 Checks

The following workflows run on pull requests that target `main`. A pull request MUST pass every applicable check before it is merged.

| Workflow | What it checks |
| --- | --- |
| CI | Ruff lint and format check, pyright (informational) and the test suite on Python 3.10 to 3.14 |
| Integration Tests | Integration tests on Python 3.10 to 3.14, end-to-end tests and a REST API smoke test |
| Coverage | Test suite under pytest-cov with upload to Codecov |
| Package | Builds the sdist and wheel, validates the metadata and installs the wheel |
| Text Policy | Text policy of the files, Python and configuration file documentation style, and commit messages |
| PR Title | Conventional Commits format of the title |
| Markdown | markdownlint (when Markdown files change) |
| Notebooks | Notebook format and absence of outputs (when notebooks change) |
| Model Zoo | Consistency of the model zoo (when it changes) |
| Fuzzing | atheris fuzz targets (when the parsers or targets change) |
| License Compliance | MPL-2.0 notices, REUSE and the licences of the dependencies |
| Security | Bandit, pip-audit and dependency review |
| Secret Scan | TruffleHog |
| CodeQL | Static analysis of Python and GitHub Actions |
| Docker | Build of the container image |
| Container Scan | Grype vulnerability scan of the image (when container files change) |
| Repository Config | Issue forms, workflows, Dependabot, CITATION.cff, Codecov, Security Insights and Compose files against their schemas, and YAML style |
| Workflow Lint | actionlint and shellcheck |
| Labeler, Labels | Automatic labels and label definitions |

Scheduled workflows additionally check links (Links), run the OpenSSF Scorecard, mark inactive issues and pull requests (Stale), rerun the integration, security and licence checks weekly and rebuild the model zoo. Most of the pull request checks can be reproduced locally with `make check` and `make pre-commit`.

## 11. Review and merge

### 11.1 Review criteria

Reviewers assess whether a pull request:

1. solves the stated problem and fits the scope and architecture of the project;
2. is correct, including edge cases, error handling and numerical behaviour (units, data types, no-data values, coordinate reference systems);
3. is tested at the appropriate tier, with regression tests for bug fixes and fuzz corpus entries for parser crashes;
4. follows the standards of Section 5 and passes all checks;
5. keeps the public API compatible, or declares and documents the break under [VERSIONING.md](VERSIONING.md);
6. updates the documentation and [CHANGELOG.md](CHANGELOG.md) where users are affected;
7. introduces no dependency, data or model whose licence or provenance is unclear;
8. makes no claim (in code, documentation or model cards) that is not supported by evidence.

### 11.2 Process

1. The Labeler workflow labels the pull request and CODEOWNERS (`.github/CODEOWNERS`) requests a review from the responsible maintainer.
2. The reviewer comments inline. Contributors SHOULD answer every comment, push fixes as new commits during review, and re-request review when ready. Review threads SHOULD be resolved before merge.
3. A maintainer merges the pull request when the checks pass and the review is approved. The maintainer chooses the merge method; when a pull request is squash-merged, its title becomes the commit message.
4. Pull requests without activity are marked `stale` after 45 days and closed 14 days later (Stale workflow). Draft pull requests and pull requests labelled `security`, `blocked`, `breaking-change` or `dependencies` are exempt.

### 11.3 Current review capacity

The project currently has one maintainer (see [MAINTAINERS.md](MAINTAINERS.md)). Pull requests from contributors are reviewed by that maintainer. Changes authored by the maintainer are merged after the automated checks pass, without an independent human review, because no second reviewer exists. This is a known limitation; [GOVERNANCE.md](GOVERNANCE.md) describes how further maintainers are added so that every change can receive review by a person other than its author. Reviews by contributors who are not maintainers are welcome and are taken into account.

## 12. Changelog and releases

User-visible changes SHOULD add an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md), which follows Keep a Changelog 1.1.0 [7]. Versions follow Semantic Versioning 2.0.0 [8] as described in [VERSIONING.md](VERSIONING.md). Releases are made by a maintainer: the version in `pyproject.toml` is updated, a tag `v<version>` is pushed, and `.github/workflows/release.yml` builds the distributions, signs them with Sigstore, attaches the `.sigstore.json` bundles, the SLSA provenance `unbihexium-<tag>.intoto.jsonl` and the checksums to the GitHub release, creates GitHub artifact attestations and uploads the distributions to PyPI. Contributors do not bump versions in pull requests unless asked.

## 13. Licensing of contributions

Unbihexium is licensed under the Mozilla Public License 2.0 [9] (see [LICENSE.txt](LICENSE.txt)). By submitting a contribution, the contributor agrees that it is licensed under MPL-2.0 and certifies the DCO as described in Section 10.3. The project does not use a contributor licence agreement, and contributors keep the copyright of their contributions; copyright notices name "Unbihexium OSS Foundation and contributors". Contributors MAY add themselves to [AUTHORS.md](AUTHORS.md) in the pull request that contains their first contribution. This section describes project practice and is not legal advice.

## References

[1] S. Bradner. Key words for use in RFCs to Indicate Requirement Levels (RFC 2119). 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words (RFC 8174). 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

[4] Google. Atheris: a coverage-guided Python fuzzing engine. 2020. <https://github.com/google/atheris>

[5] Conventional Commits. Conventional Commits 1.0.0. 2019. <https://www.conventionalcommits.org/en/v1.0.0/>

[6] The Linux Foundation and contributors. Developer Certificate of Origin, version 1.1. 2004. <https://developercertificate.org/>

[7] O. Lacan. Keep a Changelog 1.1.0. 2023. <https://keepachangelog.com/en/1.1.0/>

[8] T. Preston-Werner. Semantic Versioning 2.0.0. 2013. <https://semver.org/spec/v2.0.0.html>

[9] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://www.mozilla.org/en-US/MPL/2.0/>

<!--
=============================================================================
End of file CONTRIBUTING.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
