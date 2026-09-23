# Versioning Policy

## Overview

Unbihexium follows [Semantic Versioning 2.0.0](https://semver.org/) (SemVer) for all releases.

---

## Version Format

```text
MAJOR.MINOR.PATCH
```

### Version Components

| Component | Description | Example Change |
| ----------- | ------------- | ---------------- |
| MAJOR | Breaking API changes | Remove deprecated function |
| MINOR | New features, backward compatible | Add new model |
| PATCH | Bug fixes, backward compatible | Fix inference bug |

---

## Release Schedule

| Type | Frequency | Branch |
| ------ | ----------- | -------- |
| Major | Yearly | `main` |
| Minor | Quarterly | `main` |
| Patch | As needed | `main` |
| Pre-release | Before major/minor | `release/*` |

---

## Pre-release Versions

Pre-release versions use the following suffixes:

| Suffix | Stage | Example |
| -------- | ------- | --------- |
| `-alpha.N` | Early development | `2.0.0-alpha.1` |
| `-beta.N` | Feature complete | `2.0.0-beta.1` |
| `-rc.N` | Release candidate | `2.0.0-rc.1` |

---

## Model Zoo Versioning

Model weights are versioned separately from the library:

```text
model_name_variant_vX.Y.Z
```

| Component | Description |
| ----------- | ------------- |
| model_name | Capability identifier |
| variant | tiny, base, large, mega |
| vX.Y.Z | Model version |

---

## Python Version Support

Unbihexium supports every CPython release that is still maintained upstream. Support for a Python version is dropped in the first minor release after that version reaches its upstream end of life.

| Version | Release Date | Support Status | Bug-Fix Support Ends | End of Life (EOL) |
| --------- | -------------- | ---------------- | ---------------------- | ------------------- |
| 3.14 | 2025-10-07 | Bug-fix support | 2027-10-01 | 2030-10-31 |
| 3.13 | 2024-10-07 | Bug-fix support | 2026-10-01 | 2029-10-31 |
| 3.12 | 2023-10-02 | Security-only | 2025-04-02 | 2028-10-31 |
| 3.11 | 2022-10-24 | Security-only | 2024-04-01 | 2027-10-31 |
| 3.10 | 2021-10-04 | Security-only | 2023-04-05 | 2026-10-31 |

Every supported version is tested in CI on each pull request.

---

## Deprecation Policy

1. Features are marked deprecated in MINOR releases
2. Deprecated features emit warnings for at least 2 MINOR releases
3. Deprecated features are removed in the next MAJOR release

---

## Backward Compatibility

### Guaranteed Stable

- Public API functions
- Configuration file format
- CLI commands and options
- Model inference output format

### May Change

- Internal implementation details
- Performance characteristics
- Logging format
- Error message text

---

## Version Query

```python
import unbihexium

# Library version
print(unbihexium.__version__)

# Model version
model = unbihexium.zoo.get_model("ship_detector_mega")
print(model.version)
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.
