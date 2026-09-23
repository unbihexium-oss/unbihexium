# Changelog

All notable changes to Unbihexium are documented in this file.

## Purpose

This changelog follows [Keep a Changelog](https://keepachangelog.com/) format and adheres to [Semantic Versioning](https://semver.org/).

## Version History

```mermaid
timeline
    title Unbihexium Release Timeline
    2025-01 : v0.1.0 Initial Release
    2025-06 : v0.5.0 Model Zoo
    2025-12 : v1.0.0 Production Release
```

## Semantic Versioning

Version numbers follow the pattern:

$$
\text{MAJOR}.\text{MINOR}.\text{PATCH}
$$

Where:

- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible features
- PATCH: Backwards-compatible fixes

## Releases

| Version | Date | Type |
| ------- | ---------- | ---------- |
| 1.0.0 | 2025-12-19 | Production |
| 0.5.0 | 2025-06-01 | Beta |
| 0.1.0 | 2025-01-15 | Alpha |

---

## [Unreleased]

### Added (Unreleased)

- Official support for Python 3.13 and 3.14; Python 3.10 through 3.14 are now supported and tested in CI
- Python version support policy in VERSIONING.md
- Issue forms for bug reports, feature requests, documentation, model zoo, performance, compliance, build and questions, each with about 95 mandatory questions, an AI usage declaration and regulatory confirmations
- Pull request template with the same declarations, AI usage declaration and regulatory confirmations
- CODEOWNERS, label definitions with a label sync workflow, and release notes categories
- Workflows for package builds, licence compliance, text policy, Markdown, model zoo integrity, notebooks, repository configuration schemas, workflow linting, secret scanning, container scanning, pull request titles, path labels, link checking, stale items and first-time contributor greetings
- Pull request path labeler configuration, dependency review policy, link checker configuration and instructions for AI coding assistants in `.github/`
- Root project files: AUTHORS.md, MAINTAINERS.md, ROADMAP.md, RESPONSIBLE_USE.md, security-insights.yml (OpenSSF Security Insights 2.2), codemeta.json, codecov.yml, REUSE.toml, .mailmap, .env.example and .yamllint.yml
- REUSE compliance, Security Insights, Codecov, CodeMeta and yamllint checks in CI
- Locked dependency sets `requirements.txt` (runtime, ONNX and serving) and `requirements-dev.txt` (all extras) generated with `uv pip compile --universal` for Python 3.10 to 3.14, with `make lock` and `make lock-check`
- tox environments for the lowest supported dependency versions, formatting, security, text policy and package builds
- Codecov components per subpackage, and in-scope and out-of-scope lists, security champions and scan results in `security-insights.yml`
- Detailed third-party licence attribution in `NOTICE`, and dependency references in `CITATION.cff`

### Removed (Unreleased)

- MkDocs site configuration (`mkdocs.yml`), the Read the Docs configuration and the documentation deployment workflow; the documentation is maintained as Markdown under `docs/`
- The `docs` extra from `pyproject.toml`
- `yamllint` and `reuse` from the `dev` extra, because their GPL-3.0-or-later licence is denied by the dependency licence policy; they run in isolated pre-commit and tox environments

### Fixed (Unreleased)

- Invalid YAML in `.github/FUNDING.yml`
- SLSA provenance workflow, which called the reusable generator as a step and could not run
- Release workflow wrote `SHA256SUMS.txt` into `dist/`, which would have made the PyPI upload fail
- pre-commit hook pointing to a missing script
- markdownlint findings in the documentation
- Repository Config workflow failed because PyYAML was not installed
- README security controls table listed tools that the repository does not run
- `write_zarr` failed with zarr 3 because numcodecs compressors are only accepted for the version 2 storage format
- Zarr unit tests called `write_zarr` and `read_zarr` with the wrong argument order and return type
- The source distribution did not include `LICENSE.txt`, the notices, `REUSE.toml` or `CITATION.cff`
- Dependency floors that allowed releases with known vulnerabilities (Pillow, PyTorch, Requests, PyArrow, Starlette, python-multipart, pytest, GeoPandas, Click and Ray)

### Changed (Unreleased)

- Contact address in package metadata, citation files, the container image, the Helm chart and the security, privacy, conduct and support policies changed to `yunus.z.imanov@helsinki.fi`
- Relicensed the project from Apache-2.0 to the Mozilla Public License 2.0 (MPL-2.0)
- Added MPL-2.0 license headers to source files
- Updated package metadata, model manifests, model cards, and notebooks to reference MPL-2.0
- Raised all dependency floors to releases current in September 2026 that provide wheels for Python 3.10 to 3.14, with separate floors for Python 3.10 where newer releases dropped it
- Build backend requirement raised to hatchling 1.27 for Core Metadata 2.4 licence expressions (PEP 639)
- Rewrote `pyproject.toml`, `Makefile`, `tox.ini`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.pre-commit-config.yaml`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.mailmap`, `.env.example`, `.yamllint.yml`, `CITATION.cff`, `codemeta.json`, `codecov.yml`, `REUSE.toml` and `NOTICE` with explanatory comments
- Container image based on Python 3.14 on Debian 13 (trixie), running as a non-root user with a health check
- Docker Compose service hardened with a read-only root file system, dropped capabilities and `no-new-privileges`, with an optional GPU profile
- markdownlint configuration moved from `.markdownlint.json` to `.markdownlint.yaml` with a reason for each disabled rule, and markdownlint-cli updated to 0.49.1
- pre-commit hooks updated (ruff 0.16.8, bandit 1.9.4, yamllint 1.38.0, actionlint 1.7.12, check-jsonschema 0.38.1, reuse 6.2.0) and aligned with the CI checks
- The licence text is kept only in `LICENSE.txt`; REUSE checks copy it to the git-ignored `LICENSES/MPL-2.0.txt` before running `reuse lint`
- Trailing whitespace and missing final newlines removed from documentation, notebooks and scripts

---

## [1.0.0] - 2025-12-19

### Added (1.0.0)

- Complete Model Zoo with 130 models and 520 variants
- Production-grade pipeline framework
- CLI with inference and pipeline commands
- Comprehensive documentation with 130 notebooks
- SHA256 verification for all models

### Changed (1.0.0)

- Upgraded to ONNX Runtime for inference
- Standardized model naming convention

### Security (1.0.0)

- Added supply chain security measures
- Implemented model integrity verification

---

## [0.5.0] - 2025-06-01

### Added (0.5.0)

- Initial model zoo structure
- Basic pipeline framework
- Core capability registry

---

## [0.1.0] - 2025-01-15

### Added (0.1.0)

- Project initialization
- Basic architecture
- Development tooling
