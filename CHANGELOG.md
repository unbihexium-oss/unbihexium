<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : CHANGELOG.md
Title       : Changelog
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Changelog

| Field | Value |
| --- | --- |
| Document | UBX-DOC-CHANGELOG |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | All tagged releases of Unbihexium (v1.0.0 and v1.0.1) and the unreleased changes on the main branch |

## Abstract

This document records every notable change to Unbihexium, release by release, so that users can decide whether to upgrade, packagers can see what changed between versions, and auditors and citing researchers can relate a version number to the behaviour of the software. It follows the Keep a Changelog 1.1.0 format [1] and the versioning rules in [VERSIONING.md](VERSIONING.md), which are based on Semantic Versioning 2.0.0 [2]. The entries were compiled from the Git history of the repository: the annotated tags v1.0.0 and v1.0.1, the commits between them, and every pull request merged into the main branch since v1.0.1. No entry describes a change that is not in that history.

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. \[Unreleased\]](#2-unreleased)
- [3. \[1.0.1\] - 2025-12-21](#3-101---2025-12-21)
- [4. \[1.0.0\] - 2025-12-21](#4-100---2025-12-21)
- [References](#references)

## 1. Scope and conventions

### 1.1 Format

Each release has one section, newest first, headed by the version number and the release date in ISO 8601 form (YYYY-MM-DD). The section `[Unreleased]` at the top collects changes that are merged into the main branch but not yet part of a tagged release. Within a section, changes are grouped under the Keep a Changelog [1] headings Added, Changed, Deprecated, Removed, Fixed and Security; a heading is omitted when it has no entries. The version headings link to the corresponding comparison on GitHub.

### 1.2 Sources of the entries

- Tagged releases: the annotated tags `v1.0.0` (commit `314d1b0`) and `v1.0.1` (commit `ccfcb64`), both created on 2025-12-21, and the commits they contain.
- Unreleased changes: every pull request merged into `main` after `v1.0.1`, as listed by `git log --merges origin/main`, and the two documentation commits pushed directly to `main` on 2025-12-22. Pull request numbers are given in parentheses, for example (#39).
- Changes made before v1.0.0 were not recorded in a changelog and are not reconstructed here. Earlier versions of this file listed releases 0.1.0 and 0.5.0; no such tags or published packages exist, and those entries have been removed.

### 1.3 Conventions

The key words MUST, SHOULD and MAY in this subsection are to be interpreted as described in RFC 2119 [3] and RFC 8174 [4] when, and only when, they appear in capitals.

- Every pull request that changes behaviour, the public API, the packaging or the supply chain SHOULD add an entry to `[Unreleased]` in the same pull request.
- At release time, the maintainer MUST rename `[Unreleased]` to the new version with its date, start a new empty `[Unreleased]` section and update the comparison links at the end of this file (see [VERSIONING.md](VERSIONING.md)).
- Entries MUST describe the change as a user observes it. Entries about the model zoo MUST NOT describe the starter models as trained.
- Breaking changes MUST be identified as such, because under [VERSIONING.md](VERSIONING.md) they require a new major version.

## 2. [Unreleased]

The changes below are merged into `main` but have not been released. The latest release on the Python Package Index is still 1.0.1, which does not contain them. The version number in `pyproject.toml` remains 1.0.1 until the next release. Because this section contains breaking changes (see [2.2 Changed](#22-changed)), the versioning policy requires the next release to increase the major version.

### 2.1 Added

Model zoo and machine learning:

- A model catalogue, `src/unbihexium/zoo/catalog.yaml` (catalogue version 2.0.0), which defines 130 model families in four size variants (tiny, base, large and mega), 520 models in total, with inputs, outputs, units, required training labels and suitable data for every family (#37).
- Trainable architectures for every task: a CenterNet detector, U-Net networks for segmentation, change detection, dense regression and enhancement, a pooled encoder for scene regression, an EDSR-style super-resolution network, and exact spectral index modules (#37).
- Deterministic, platform-independent starter weights derived from the model identifier, with published SHA-256 digests of all 520 models (`src/unbihexium/zoo/digests.json`) that are verified on load. These are untrained starter models: only the 28 models of the 7 spectral index families compute their published formulas without training; every other model must be trained on labelled data before its output means anything (#37).
- Checkpoints loaded with `torch.load(weights_only=True)`, ONNX export verified against ONNX Runtime, and a local model store under `$UNBIHEXIUM_CACHE` (#37).
- `python -m unbihexium.zoo.sync`, which writes the digests, manifests, model cards and inventory from the catalogue, and a JSON Schema for the manifests, `model_zoo/manifest.schema.json` (#37).
- Training for every learned task (`unbihexium.ai.training`, `unbihexium train`): dataset folders with GeoTIFF or NumPy images, class masks, pixel or GeoJSON boxes and scene targets, random and grid chips, augmentation, normalisation statistics stored in the checkpoint, task losses, AdamW with warm-up and cosine decay, validation, best and last checkpoints, early stopping and a JSON history (#38).
- Synthetic datasets for every trainable task, to check a training setup without data (`--synthetic`) (#38).
- Evaluation metrics for every task: mean average precision at IoU 0.5 and 0.5 to 0.95, confusion matrix with IoU, F1 and kappa, MAE, RMSE, bias and R squared, PSNR and SSIM (`unbihexium evaluate`) (#38).
- Tiled inference with blending, no-data handling and PyTorch or ONNX Runtime backends (`unbihexium.ai.inference.Predictor`); exported ONNX files carry their configuration, so inference with ONNX Runtime needs no PyTorch (#38).
- Task APIs backed by the model zoo: detectors (ships, SAR ships, buildings, aircraft, vehicles, greenhouses, crop fields, pivots and fires), segmenters (land cover, water, clouds, crops, SAR floods and oil spills), change detection, dense and scene regression, enhancement and super-resolution, with georeferenced results and GeoJSON and GeoTIFF output (#38).
- Commands `unbihexium train`, `evaluate` and `predict`, and `unbihexium zoo build`, `export`, `info` and `clear`, next to the existing `zoo list`, `verify` and `where` (#38).
- Guides [docs/model_zoo/training.md](docs/model_zoo/training.md) and [docs/model_zoo/inference.md](docs/model_zoo/inference.md) (#38).

Library packages (#39):

- Core: raster windows, clipping, masks, reprojection and resampling through GDAL, band math, statistics and Cloud Optimized GeoTIFF output; 27 registered spectral indices with Sentinel-2 and Landsat band aliases; Sentinel-1, Sentinel-2 and Landsat 8 and 9 band tables with radiometric helpers; tiling with blended mosaics and XYZ tile arithmetic; geodesic vector operations; scene harmonisation; products with checksums and STAC items; pipelines with step records, seeding, provenance and SHA-256 evidence.
- Input and output: RFC 7946 GeoJSON validation and reprojection, GeoTIFF windows, overviews and COG, GeoParquet with bounding box filters, offline STAC catalogues and a paging STAC API client, and Zarr version 3.
- Registries: 147 capabilities (one per model family and 17 library features), a model registry view over the zoo with band checks, and pipeline search.
- REST service: `POST /predict/{model_id}` for every zoo model, with band, size and value limits, JSON or base64 NumPy input, API keys and rate limiting.
- SAR: calibration to sigma0, beta0 and gamma0; Lee, Kuan, enhanced Lee, Frost, Gamma MAP and refined Lee speckle filters; Goldstein phase filtering; least-squares and quality-guided phase unwrapping; and Pauli, Freeman-Durden, Yamaguchi and H/A/alpha decompositions.
- Terrain: slope, aspect, hillshade, curvature, TPI, TRI, roughness and VRM, depression filling, D8 flow direction and accumulation, watersheds, streams, the topographic wetness index and viewsheds.
- Geostatistics: Matheron and Cressie-Hawkins variograms, ordinary and universal kriging with variance and cross-validation, inverse distance weighting, global and local Moran's I, Geary's C and Getis-Ord Gi*.
- Analysis: network routing with Dijkstra and A*, origin-destination matrices, cost distance and least-cost paths, AHP and weighted overlay suitability, and zonal statistics tables.
- Preprocessing (radiometry, masks, enhancement and pansharpening), postprocessing (map cleaning, connected components and vectorisation), accuracy assessment with the area estimators of Olofsson et al. (2014), and visualisation (colour maps, composites, relief and quicklooks).
- Settings loaded from defaults, YAML files and environment variables, and utilities for hashing, logging, timing, seeding and atomic file writes.
- Red edge, coastal and narrow NIR band mappings in `unbihexium index`.

Platform, repository and continuous integration:

- Official support for CPython 3.13 and 3.14; CPython 3.10 to 3.14 are tested in CI, and a Python version support policy was added to [VERSIONING.md](VERSIONING.md) (#21).
- Issue forms for bug reports, feature requests, documentation, the model zoo, performance, compliance, build and packaging, and questions; a rewritten pull request template; CODEOWNERS; label definitions synchronised from `.github/labels.yml`; and release note categories in `.github/release.yml` (#23).
- Workflows for package builds, licence compliance, the text policy, Markdown, model zoo integrity, repository configuration schemas, workflow linting (actionlint and shellcheck), secret scanning (TruffleHog), container scanning (Grype), conventional pull request titles, path labels, link checking, stale items and first-time contributors, and a dependency review policy (#30).
- Root project files: AUTHORS.md, MAINTAINERS.md, ROADMAP.md, RESPONSIBLE_USE.md, `security-insights.yml` (OpenSSF Security Insights), `codemeta.json`, `codecov.yml`, `REUSE.toml`, `.mailmap`, `.env.example` and `.yamllint.yml`, with REUSE, Security Insights, Codecov, CodeMeta and yamllint checks in CI (#31).
- Locked dependency sets `requirements.txt` (runtime with the `onnx` and `serving` extras) and `requirements-dev.txt` (all extras), compiled with `uv pip compile --universal` for Python 3.10 to 3.14, with `make lock` and `make lock-check`; tox environments for the lowest supported dependency versions, formatting, security, the text policy and package builds (#33).
- Codecov components per subpackage (#33).
- A Python comment style check (`.github/scripts/check_python_style.py`), extended to every package (#37, #39), and a documentation style check for configuration and data files (`.github/scripts/check_config_style.py`) (#42).
- A Model Zoo workflow that checks the generated files and rebuilds the starter weights to prove that their digests are reproducible (#37).
- Integration and end-to-end tests that run registered pipelines on GeoTIFFs, round-trip a model through the store, compare ONNX with PyTorch after training on a dataset folder, and run ship detection, burn severity, change detection and REST workflows, replacing empty placeholder tests (#41).
- Throughput and memory measurements in the benchmark tests, replacing empty benchmarks (#39).

### 2.2 Changed

Breaking changes:

- The project was relicensed from Apache-2.0 to the Mozilla Public License 2.0 (MPL-2.0); source files carry the MPL-2.0 notice (#20).
- Rewritten modules changed their interfaces (#39): writers take the data before the path (the previous order is still accepted); `read_geotiff` returns the transform as six coefficients and the CRS as a string; SAR angles are in degrees by default; a zero denominator of a spectral index gives NaN instead of using a small epsilon; `aspect` returns compass degrees and `hillshade` returns floating point values; `Evidence` and `ProvenanceRecord` have new fields; pipeline steps must return a mapping; `ssim` uses a Gaussian window; model configuration defaults to the base variant on the CPU.
- Task APIs take a catalogue model, a trained checkpoint or an ONNX file (`weights=`) and default to the base variant; `SuperResolution` uses the catalogue factor 4 unless `scale_factor` is given; `CropDetector` and `GreenhouseDetector` are detectors, as in the catalogue, and remain importable from `unbihexium.ai.segmentation` (#38).

Other changes:

- `unbihexium zoo download` and `unbihexium infer` are kept as hidden aliases of `zoo build` and `predict` (#38).
- `unbihexium index` computes the index and writes it as GeoTIFF (#38).
- All dependency lower bounds were raised to releases that were current in September 2026 and provide wheels for Python 3.10 to 3.14, with separate lower bounds for Python 3.10 where newer releases dropped it; the build backend requirement was raised to hatchling 1.27 for PEP 639 licence expressions (#33).
- The container image is based on Python 3.14 on Debian 13 (trixie), runs as a non-root user and has a health check; the Docker Compose service uses a read-only root file system, drops capabilities, sets `no-new-privileges` and has an optional GPU profile (#33).
- The documentation is maintained as Markdown under `docs/` instead of an MkDocs site (#33).
- The markdownlint configuration moved to `.markdownlint.yaml` with a reason for each disabled rule; markdownlint-cli was updated to 0.49.1; pre-commit hooks were updated and aligned with CI (#33).
- The licence text is kept only in `LICENSE.txt`; REUSE checks copy it to the ignored `LICENSES/MPL-2.0.txt` before `reuse lint` runs (#32, #33).
- `NOTICE` gives detailed licensing and third-party attribution, and `CITATION.cff` lists the works the library builds on (#33).
- The contact address in the package metadata, the citation files, the container image, the Helm chart and the security, privacy, conduct and support policies is `yunus.z.imanov@helsinki.fi` (#35).
- Python files in the package, the tests, `.github/scripts/`, `scripts/` and `examples/` use `#` comments with a standard header and footer; configuration and data files carry the same header (#36, #37, #39, #42).
- GitHub Actions dependencies were updated by Dependabot: `codecov/codecov-action` 4 to 5 (#12), `ossf/scorecard-action` 2.3.1 to 2.4.3 (#11), `slsa-framework/slsa-github-generator` 1.9.0 to 2.1.0 (#10), `actions/upload-pages-artifact` 3 to 4 (#9), `docker/build-push-action` 5 to 6 (#8), `actions/setup-python` 5 to 6 (#18), `softprops/action-gh-release` 1 to 2 (#17), `github/codeql-action` 3 to 4 (#16), `actions/checkout` 4 to 6 (#15) and `actions/attest-build-provenance` 1 to 3 (#14); the container base image moved from `python:3.12-slim` to `python:3.14-slim` (#13).
- A model path lookup that searches the parent directories for `model_zoo` (#7) and Git LFS set-up instructions (direct commits on 2025-12-22) were added for the example material and the documentation. Both referred to the model files that were removed later in this cycle and were superseded by the removals in [2.3 Removed](#23-removed).
- The documentation under `docs/` was rewritten to match the current code, in the same document layout as the root documents, with every code example and command executed against the current code.
- The PyPI `Homepage` link and the `url` fields of `CITATION.cff` and `codemeta.json` point to the GitHub repository, because the project has no separate web site; the Helm chart and the Kubernetes manifest use the reserved placeholder host `unbihexium.example.com`.

### 2.3 Removed

- The 130 example notebooks under `examples/notebooks/`, which loaded the removed model files and could not run as written, together with their format check, `make notebooks` and the nbformat CI dependency.
- The model files stored with Git LFS under `model_zoo/assets/`, the 520 per-variant model cards and their metrics, which were not the result of training or evaluation on real data (#37).
- Placeholder model classes in `unbihexium.ai.models`, `unbihexium.ai.change_detection`, `unbihexium.ai.super_resolution` and `unbihexium.ai.synthesis`, and the model zoo `cache` and `downloader` modules (#37).
- The MkDocs configuration, the Read the Docs configuration, the documentation deployment workflow and the `docs` extra (#33).
- `yamllint` and `reuse` from the `dev` extra, because the dependency licence policy denies their GPL-3.0-or-later licence; they run in isolated pre-commit and tox environments (#33).
- The separate SLSA generator workflow; the release workflow now attaches the provenance of its attestation instead (#47).
- The `LICENSES/` directory from version control (#32).

### 2.4 Fixed

- `unbihexium.ai` could not be imported because a module and a package were both named `super_resolution`, and `unbihexium.analysis.network` could not be imported for the same reason (#37, #39).
- The detection, segmentation and super-resolution classes returned empty or interpolated placeholder results instead of running a model (#38).
- `unbihexium index` read the input but did not compute or write the index, and `from unbihexium.cli import cli` failed (#38).
- Every unit test failure present before the rewrite: GeoJSON, GeoTIFF and GeoParquet writers, pipeline runs, evidence and provenance records, sigma0 and gamma0 with angles in degrees, and polarimetric decompositions (#39).
- The version tuple did not match the version string (#39).
- The CI test, coverage, integration and end-to-end jobs ignored test failures; they now install the optional backends and fail on any failing test (#39).
- The model zoo registry referenced a task that does not exist and failed on import (#37).
- `parse_datetime` read time zone digits into the fraction of a second, and `upsample_to_pan` overshot constant bands with SciPy releases before 1.17; results are now identical on Python 3.10 and 3.11 (#40).
- `write_zarr` failed with zarr 3, because numcodecs compressors are only accepted for the version 2 storage format; the Zarr unit tests used the wrong argument order and return type (#33).
- Invalid YAML in `.github/FUNDING.yml` (#23).
- The SLSA provenance workflow called the reusable generator as a step and could not run; the release workflow wrote `SHA256SUMS.txt` into `dist/`, which would have broken the PyPI upload; a pre-commit hook pointed to a missing script; markdownlint findings in the documentation (#30).
- The Repository Config workflow failed because PyYAML was not installed, and the README listed security tools that the repository did not run (#31).
- The source distribution did not include `LICENSE.txt`, the notices, `REUSE.toml` or `CITATION.cff` (#33).
- GeoJSON and STAC parsing raised `TypeError`, `OverflowError`, `IndexError` or `AttributeError` on malformed input, and rings that mixed 2-D and 3-D positions broke the area computation; malformed input now raises `ValueError` (found by fuzzing, #45).
- `rewind` reversed rings of zero or near-zero area, including collinear holes, on every call; such rings now keep their order (found by fuzzing, #45, #46).

### 2.5 Security

- Dependency lower bounds exclude releases with known vulnerabilities in Pillow, PyTorch, Requests, PyArrow, Starlette, python-multipart, pytest, GeoPandas, Click and Ray (#33).
- Model checkpoints are loaded with `torch.load(weights_only=True)`, which refuses arbitrary pickled objects, and weights are verified against their SHA-256 digests on load (#37).
- The REST prediction endpoint enforces band, size and value limits, API keys and rate limiting (#39).
- CodeQL analysis for Python and GitHub Actions (#45; an earlier CodeQL workflow had been removed in #22).
- Every GitHub Action is pinned to a full commit SHA, the actionlint download is verified, and write permissions are granted per job instead of per workflow (#45).
- CI tools, test dependencies, fuzzing dependencies and the CPU build of PyTorch are installed from hashed lock files under `.github/requirements/` (#45, #47).
- atheris fuzz targets for the GeoJSON and STAC parsers under `fuzz/`, run on pull requests and weekly (#45).
- The container base image is pinned by digest and wheels are installed with verified hashes; the SPDX software bill of materials is computed from the digest of the pushed image (#45).
- Release distributions are signed with Sigstore (`.sigstore.json` bundles), and each GitHub release carries the SLSA provenance of its GitHub artifact attestation as `unbihexium-<tag>.intoto.jsonl` (#45, #47).
- `.github/` is kept in source archives so that OpenSSF Scorecard can analyse the workflows (#46).

### 2.6 Merged pull requests

| Pull request | Merged | Title |
| --- | --- | --- |
| #7 | 2025-12-22 | Refactor model_zoo path detection for portability |
| #8 to #12 | 2025-12-28 | Dependabot updates of GitHub Actions |
| #13 to #18 | 2026-09-23 | Dependabot updates of GitHub Actions and the container base image |
| #20 | 2026-09-23 | Relicense project from Apache-2.0 to MPL-2.0 |
| #21 | 2026-09-23 | Support Python 3.10 through 3.14 |
| #22 | 2026-09-23 | Delete .github/workflows/codeql.yml |
| #23 | 2026-09-23 | Overhaul .github: issue forms, pull request template and repository configuration |
| #30 | 2026-09-23 | ci: add repository workflows and GitHub configuration |
| #31 | 2026-09-23 | chore: add standard root project files and fix repository checks |
| #32 | 2026-09-23 | Delete LICENSES directory |
| #33 | 2026-09-23 | build: update dependencies to September 2026 and document the root configuration files |
| #35 | 2026-09-23 | docs: use `yunus.z.imanov@helsinki.fi` as the contact address |
| #36 | 2026-09-23 | docs: comment the repository scripts and examples line by line with academic headers |
| #37 | 2026-09-23 | feat(zoo): replace the placeholder model zoo with 520 trainable starter models |
| #38 | 2026-09-23 | feat(ai): train, evaluate and run the model zoo models |
| #39 | 2026-09-23 | feat: rewrite and expand every package, fix the test suite and gate CI on tests |
| #40 | 2026-09-23 | fix: keep results identical on Python 3.10 and 3.11 |
| #41 | 2026-09-23 | test: replace placeholder integration and end-to-end tests |
| #42 | 2026-09-23 | chore: apply the documentation style to every configuration and data file |
| #45 | 2026-09-23 | ci: raise the OpenSSF Scorecard checks and fix the bugs found by fuzzing |
| #46 | 2026-09-23 | fix: keep collinear rings stable in rewind and let Scorecard read the workflows |
| #47 | 2026-09-23 | ci: pin PyTorch by hash and attach SLSA provenance to releases |

## 3. [1.0.1] - 2025-12-21

Patch release, tagged at commit `ccfcb64`. It is the only version published on the Python Package Index. It was built before release signing was introduced, so its artefacts have no Sigstore bundles or SLSA provenance assets.

### 3.1 Added

- Publication to the Python Package Index from the release workflow.

### 3.2 Fixed

- The Dockerfile referred to the wrong licence file name.
- Badges in the README pointed to unrelated logos; the GDAL badge uses the official icon.

## 4. [1.0.0] - 2025-12-21

First tagged release, at commit `314d1b0`, licensed under Apache-2.0 and declared for CPython 3.10 to 3.12. It was published as a GitHub release; no 1.0.0 package exists on the Python Package Index.

### 4.1 Added

- The `unbihexium` package with the subpackages `ai`, `analysis`, `cli`, `config`, `core`, `geostat`, `indices`, `io`, `metrics`, `postprocessing`, `preprocessing`, `registry`, `sar`, `serving`, `terrain`, `utils`, `visualization` and `zoo`.
- The command line interface with `info`, `index`, `infer`, `pipeline` and `zoo` (`list`, `download`, `verify` and `where`) commands.
- A FastAPI service with health, capability and inference endpoints.
- A model zoo of 130 capabilities in four variants (520 ONNX files, stored with Git LFS under `model_zoo/assets/`), with manifests and SHA-256 checksums. These files were not the result of training on real data and were removed in the unreleased changes (see [2.3 Removed](#23-removed)).
- 130 example notebooks, one per model zoo capability.
- Documentation built with MkDocs, a Dockerfile, Kubernetes and Helm deployment files, and sample test fixtures (simulated Sentinel-2, DEM and segmentation mask arrays).
- GitHub Actions workflows for CI, coverage, integration tests, CodeQL, OpenSSF Scorecard, security scanning, documentation, Docker images, SLSA provenance and releases.

### 4.2 Fixed

- The Dockerfile installs a GDAL library package that is available in the base image distribution.

## References

[1] Olivier Lacan and contributors. Keep a Changelog, version 1.1.0. 2023. <https://keepachangelog.com/en/1.1.0/>

[2] Tom Preston-Werner. Semantic Versioning 2.0.0. 2013. <https://semver.org/spec/v2.0.0.html>

[3] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[4] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[Unreleased]: https://github.com/unbihexium-oss/unbihexium/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/unbihexium-oss/unbihexium/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/unbihexium-oss/unbihexium/releases/tag/v1.0.0

<!--
=============================================================================
End of file CHANGELOG.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
