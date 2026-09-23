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
- Root project files: AUTHORS.md, MAINTAINERS.md, ROADMAP.md, RESPONSIBLE_USE.md, security-insights.yml (OpenSSF Security Insights 2.0), codemeta.json, codecov.yml, REUSE.toml with LICENSES/, .mailmap, .env.example and .yamllint.yml
- REUSE compliance, Security Insights, Codecov, CodeMeta and yamllint checks in CI

### Fixed (Unreleased)

- Invalid YAML in `.github/FUNDING.yml`
- SLSA provenance workflow, which called the reusable generator as a step and could not run
- Release workflow wrote `SHA256SUMS.txt` into `dist/`, which would have made the PyPI upload fail
- pre-commit hook pointing to a missing script
- markdownlint findings in the documentation
- Repository Config workflow failed because PyYAML was not installed
- README security controls table listed tools that the repository does not run

### Changed (Unreleased)

- Relicensed the project from Apache-2.0 to the Mozilla Public License 2.0 (MPL-2.0)
- Added MPL-2.0 license headers to source files
- Updated package metadata, model manifests, model cards, and notebooks to reference MPL-2.0

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
