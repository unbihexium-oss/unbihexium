<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/getting_started/installation.md
Title       : Installation
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Installation

| Field | Value |
| --- | --- |
| Document | UBX-DOC-303 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch |

## Abstract

This document explains how to install Unbihexium: from the Python Package Index with the optional extras declared in `pyproject.toml`, reproducibly from the hashed lock file `requirements.txt`, as the container image `ghcr.io/unbihexium-oss/unbihexium`, and from a clone of the repository for development. It is written for users setting up a first environment, for operators who deploy the command line interface or the REST service, and for contributors who need the locked development environment. It states the supported interpreters and platforms, what each extra adds, how to verify an installation, and how to diagnose the most common installation problems. Running a first analysis is covered in [quickstart.md](quickstart.md), and runtime settings in [configuration.md](configuration.md).

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Requirements](#2-requirements)
- [3. Choosing an installation method](#3-choosing-an-installation-method)
- [4. Installing from PyPI](#4-installing-from-pypi)
- [5. Reproducible installation from lock files](#5-reproducible-installation-from-lock-files)
- [6. Container image](#6-container-image)
- [7. Installing from source](#7-installing-from-source)
- [8. Verifying the installation](#8-verifying-the-installation)
- [9. Platform notes](#9-platform-notes)
- [10. Upgrading and uninstalling](#10-upgrading-and-uninstalling)
- [11. Troubleshooting](#11-troubleshooting)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

This document covers the distribution `unbihexium` on PyPI, the lock files `requirements.txt` and `requirements-dev.txt` in the repository root, the container image built by `.github/workflows/docker.yml` from the [Dockerfile](../../Dockerfile), and editable installations from a Git clone. It does not cover the Helm chart and Kubernetes manifests under `deploy/`, which are described in [docs/operations/docker.md](../operations/docker.md).

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals. Shell commands are written for a POSIX shell (Bash or Zsh); Windows equivalents are given where they differ. `python` denotes the interpreter of the environment you install into.

## 2. Requirements

### 2.1 Python interpreter

Unbihexium supports CPython 3.10, 3.11, 3.12, 3.13 and 3.14. The package declares `requires-python = ">=3.10"`, so pip will also install it on newer interpreters, but only the five listed versions are tested (see [VERSIONING.md](../../VERSIONING.md)). Other Python implementations are not tested.

### 2.2 Native libraries

No compiler and no system installation of GDAL, PROJ or GEOS are needed. The binary wheels of rasterio, pyproj, shapely and onnxruntime bundle these libraries. Every locked version publishes a wheel, or a pure Python source distribution, for CPython 3.10 to 3.14 on Linux x86_64 (this is verified when the lock files are regenerated, as recorded in their headers).

### 2.3 Hardware

A CPU is sufficient for every part of the library. Building, training and exporting models of the model zoo requires PyTorch (extra `torch`); running exported ONNX models requires only ONNX Runtime (extra `onnx`). A CUDA GPU is optional and used only through a CUDA build of PyTorch (Section 4.3). Apple silicon GPUs can be selected in training and evaluation with `--device mps` when the installed PyTorch build supports it.

### 2.4 Disk space

The size of an environment depends mostly on the extras. As a reference, a virtual environment created from `requirements.txt` (runtime dependencies with the `onnx` and `serving` extras, no PyTorch) occupied 877 MB, measured with `du -sh` on Linux x86_64 with CPython 3.11 in September 2026. PyTorch adds substantially more, in particular the CUDA builds on Linux. A model built into the local model store stores its weights as 32-bit floats, about 4 bytes per parameter: the learned tiny models have 134,992 to 735,428 parameters (for example, the checkpoint of `water_surface_detector_tiny` is 2.9 MB) and the largest mega model 60,460,548 parameters (about 240 MB).

## 3. Choosing an installation method

| Method | Section | Suitable for | Contains PyTorch |
| --- | --- | --- | --- |
| `pip install unbihexium` with extras | [4](#4-installing-from-pypi) | Using the released version in your own environment | Only with the `torch` or `all` extra |
| Lock file `requirements.txt` | [5](#5-reproducible-installation-from-lock-files) | Reproducible deployments of the CLI and the REST service | No |
| Container image | [6](#6-container-image) | Running the CLI or the REST service without a local Python | Yes, CPU build |
| Git clone, editable install | [7](#7-installing-from-source) | The newest code, development and contributions | As selected |

**Release status.** The distribution on PyPI with version 1.0.1 was built from the tag `v1.0.1` (21 December 2025). The main branch has changed substantially since then (among other things model training and evaluation, the `predict` and `zoo build` commands, and the REST prediction route) and has not yet been released under a new version number, although `pyproject.toml` still declares 1.0.1. The documentation in `docs/` describes the main branch. Users who want the commands and functions described there SHOULD install from source (Section 7) until the next release; see [CHANGELOG.md](../../CHANGELOG.md).

## 4. Installing from PyPI

### 4.1 Core installation

Create a virtual environment [3] and install the package with pip [4]:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install unbihexium
```

The core installation depends on NumPy, SciPy, rasterio, shapely, GeoPandas, pyproj, click, rich, pydantic, PyYAML, requests, Pillow and scikit-image. It provides input and output of GeoTIFF and GeoJSON, preprocessing, spectral indices, SAR, terrain, geostatistics, spatial analysis, metrics, visualisation, the model catalogue (listing and describing models) and the `unbihexium` command.

### 4.2 Optional extras

The extras below are declared in `[project.optional-dependencies]` of [pyproject.toml](../../pyproject.toml). Several can be combined, for example `python -m pip install "unbihexium[torch,onnx,serving]"`. Quote the requirement so that the shell does not interpret the square brackets.

| Extra | Packages added | Needed for |
| --- | --- | --- |
| `onnx` | onnxruntime (1.23 series on Python 3.10, 1.24.1 or newer otherwise), onnx | Inference on ONNX exports without PyTorch |
| `torch` | torch 2.13 or newer, onnx | Building, training, evaluating and exporting model zoo models |
| `serving` | fastapi, starlette 1.3.1 or newer, uvicorn[standard] | The REST service `unbihexium.serving` |
| `zarr` | zarr (2.18 on Python 3.10, 3.1.4 or newer otherwise), numcodecs | Zarr input and output in `unbihexium.io` |
| `parquet` | pyarrow | GeoParquet input and output |
| `test` | pytest, pytest-cov, pytest-xdist, httpx | Running the test suite |
| `dev` | the `test` extra, ruff, pyright, scipy-stubs (Python 3.12 and newer), pre-commit, bandit, pip-audit, build, twine, tox | Development and release work |
| `all` | every extra above | A complete environment |

STAC search (`unbihexium.io.stac`) needs no extra; it uses requests. The functions of `unbihexium.io` import optional dependencies only when they are called, so a missing extra shows up as an `ImportError` at the first call of such a function, not at import time.

### 4.3 PyTorch builds

The `torch` extra installs whatever PyTorch wheel pip selects from PyPI; on Linux x86_64 that is a CUDA build with large NVIDIA runtime wheels. To obtain a CPU-only or a specific CUDA build, install PyTorch first from the index recommended by the PyTorch installation selector [5], then install Unbihexium with the extra; pip keeps the already installed PyTorch when it satisfies the requirement:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "unbihexium[torch]"
```

For a GPU, install the CUDA build of PyTorch that matches the driver in the same way, then the `torch` extra; there is no separate GPU extra.

## 5. Reproducible installation from lock files

### 5.1 Lock files of the repository

| File | Content | Hashes | Used by |
| --- | --- | --- | --- |
| `requirements.txt` | Runtime dependencies with the `onnx` and `serving` extras | SHA-256 for every package | `make install`, deployments without PyTorch |
| `requirements-dev.txt` | Runtime dependencies with the `all` extra | SHA-256 for every package | `make install-dev` |
| `.github/requirements/requirements-docker.txt` | Runtime dependencies with the `onnx` and `serving` extras and the dependencies of PyTorch | SHA-256 for every package | The Dockerfile |
| `.github/requirements/requirements-ci-torch.txt` | The CPU build of PyTorch from the PyTorch package index | SHA-256 for every package | The Dockerfile, CI workflows |
| Other `.github/requirements/*.txt` | Test, tool and fuzzing environments of the CI workflows | SHA-256 for every package | GitHub Actions |

All lock files are universal: one file covers CPython 3.10 to 3.14 on Linux, macOS and Windows, and environment markers select the newest release that supports each interpreter. They are generated from `pyproject.toml` with uv [6] by `make lock` and checked with `make lock-check`; they MUST NOT be edited by hand. Dependabot proposes updates weekly.

### 5.2 Hash-checked runtime installation

`requirements.txt` is installed in pip's hash-checking mode [4], which rejects any file whose SHA-256 differs from the lock file. The package itself is installed afterwards without resolving dependencies again, so that exactly the locked versions remain:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-deps .
python -m pip check
```

These commands were run in a fresh CPython 3.11 environment on Linux x86_64; `pip check` reported `No broken requirements found.` The resulting environment has no PyTorch: it runs the core functions, ONNX exports and the REST service with ONNX files, but it cannot build or train zoo models. `make install` runs the same two installation commands.

### 5.3 Locked development environment

```bash
python -m pip install --require-hashes -r requirements-dev.txt
python -m pip install --no-deps -e .
pre-commit install
```

This is what `make install-dev` runs. On Linux x86_64 the PyTorch wheel from PyPI pulls in the NVIDIA CUDA runtime wheels, which are several gigabytes; for a CPU-only environment, install the CPU build of PyTorch as described in Section 4.3 instead.

## 6. Container image

### 6.1 Published tags

The workflow `.github/workflows/docker.yml` builds the image from the Dockerfile and pushes it to the GitHub Container Registry as `ghcr.io/unbihexium-oss/unbihexium`. Pushes to `main` are tagged `main`, version tags `v<major>.<minor>.<patch>` are tagged `<major>.<minor>.<patch>` and `<major>.<minor>`, and every pushed image is also tagged `sha-<short commit>`. There is no `latest` tag. Pull requests only build the image.

### 6.2 Contents

The image is based on `python:3.14.7-slim-trixie`, pinned by digest. Its builder stage installs `.github/requirements/requirements-docker.txt` and then the CPU build of PyTorch from `.github/requirements/requirements-ci-torch.txt`, both with `--only-binary=:all: --require-hashes`, into `/opt/venv`; it then builds a wheel of the package, installs it without dependencies, runs `pip check` and imports `torch`, `onnxruntime` and `unbihexium.serving.app`. The runtime stage:

- contains the runtime dependencies, the `onnx` and `serving` extras and the CPU build of PyTorch, so it can build, train and export zoo models and run ONNX exports, but it has no CUDA support;
- contains no model weights; models are built into the model store on first use;
- runs as the unprivileged user `unbihexium` (UID and GID 1000) in `/home/unbihexium`;
- sets `UNBIHEXIUM_CACHE=/home/unbihexium/.cache/unbihexium` and declares that directory a volume;
- exposes port 8000, has a health check that imports the package, and runs `unbihexium --help` by default.

### 6.3 Running the image

```bash
docker pull ghcr.io/unbihexium-oss/unbihexium:main
docker run --rm ghcr.io/unbihexium-oss/unbihexium:main unbihexium info
docker run --rm -p 8000:8000 ghcr.io/unbihexium-oss/unbihexium:main \
    unbihexium serve --host 0.0.0.0 --port 8000
```

The REST service reads its settings from `UNBIHEXIUM_SERVING__*` environment variables, which are passed with `docker run -e`, or from a YAML file passed with `unbihexium serve --config`; see [configuration.md](configuration.md). Before the service is exposed beyond a trusted network, an API key and a rate limit SHOULD be set.

### 6.4 Building the image locally

```bash
docker build -t unbihexium:local .
make docker-build        # also passes VERSION, VCS_REF and BUILD_DATE as OCI labels
make docker-api          # serves the REST API on http://localhost:8000
```

[docker-compose.yml](../../docker-compose.yml) builds the same image, starts the REST service with a read-only root file system, no Linux capabilities and a named volume for the model store, and reads optional settings from a `.env` file created from [.env.example](../../.env.example). Operation of the image and of the Helm chart is described in [docs/operations/docker.md](../operations/docker.md).

## 7. Installing from source

### 7.1 Editable installation

```bash
git clone https://github.com/unbihexium-oss/unbihexium.git
cd unbihexium
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[torch,onnx,serving]"
unbihexium --version
```

The build backend is hatchling 1.27 or newer (PEP 517 [7]); pip installs it automatically in an isolated build environment. The release workflow and the Dockerfile instead install the hash-pinned hatchling of `.github/requirements/` and build without isolation, so no unpinned backend is fetched. The package uses the `src/` layout, and the wheel contains `src/unbihexium` including the model catalogue `zoo/catalog.yaml` and the published weight digests `zoo/digests.json`.

### 7.2 Development environment

Contributors SHOULD use the locked development environment of Section 5.3 (`make install-dev`), which also installs the pre-commit hooks. The Makefile lists every developer task with `make help`; the most important ones are `make test-fast` (unit tests in parallel, without slow and GPU tests), `make check` (the local equivalent of the CI checks) and `make build` (sdist and wheel in `dist/`). [tox.ini](../../tox.ini) defines isolated environments per interpreter, for example `tox -e py313`. The contribution workflow is described in [CONTRIBUTING.md](../../CONTRIBUTING.md).

### 7.3 Building distributions

```bash
python -m pip install build
python -m build
python -m twine check --strict dist/*
```

Official releases are not built locally: they are built, signed with Sigstore and given SLSA provenance by `.github/workflows/release.yml` (see [SECURITY.md](../../SECURITY.md) and [VERSIONING.md](../../VERSIONING.md)).

## 8. Verifying the installation

### 8.1 Version and registries

```bash
unbihexium --version
unbihexium info
```

Output on the main branch:

```text
unbihexium, version 1.0.1
Unbihexium v1.0.1
Registered capabilities: 147
Model zoo models: 520 (catalogue 2.0.0)
Registered pipelines: 5
```

The same information is available from Python, and `make verify` prints the version and the number of models:

```bash
python -c "import unbihexium; print(unbihexium.__version__)"
python -c "from unbihexium.zoo import list_models; print(len(list_models()))"
```

### 8.2 Optional components

Each extra can be checked by importing its main package:

```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"      # extra onnx
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # extra torch or gpu
python -c "import fastapi, uvicorn; print(fastapi.__version__)"     # extra serving
python -m pip check
```

With the `torch` extra, building a tiny starter model checks PyTorch and the weight verification in one step; the model is written to the directory named by `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`):

```bash
unbihexium zoo build water_surface_detector_tiny
unbihexium zoo verify water_surface_detector_tiny
```

The model zoo contains 520 models (130 families in the variants tiny, base, large and mega). Apart from the 28 models of the 7 spectral index families, which compute exact formulas, they are untrained starter models with deterministic weights: a successful build shows that the installation works, not that the model produces meaningful predictions. See [docs/model_zoo/training.md](../model_zoo/training.md).

### 8.3 Test suite

In a source checkout with the `test` or `dev` extra:

```bash
python -m pytest tests/ -n auto
```

## 9. Platform notes

### 9.1 Linux

All CI workflows run on GitHub-hosted Ubuntu runners (x86_64); this is the tested platform. The container image is built on the same runners and therefore for `linux/amd64` only.

### 9.2 macOS and Windows

The package is pure Python and its dependencies publish wheels for macOS and Windows, and the lock files resolve for both. These platforms are not exercised by CI, so problems SHOULD be reported through the issue tracker (see [SUPPORT.md](../../SUPPORT.md)). On Windows, activate a virtual environment with `.venv\Scripts\activate` and quote extras with double quotes in `cmd.exe`, for example `python -m pip install "unbihexium[onnx]"`. On macOS with Apple silicon, PyTorch's `mps` device can be passed to `unbihexium train` and `unbihexium evaluate`.

### 9.3 Python 3.10

Several scientific packages dropped Python 3.10 in their newest releases. Environment markers in `pyproject.toml` therefore select older series on Python 3.10: NumPy 2.2, SciPy 1.15, pyproj 3.7.1, scikit-image 0.25, onnxruntime 1.23 and zarr 2.18 with numcodecs 0.13. Zarr stores written with zarr 3 in the Zarr v3 format cannot be read by zarr 2.

### 9.4 Conda environments

The project publishes distributions only on PyPI and the container image on GHCR. In a conda environment, install Unbihexium with pip after creating the environment with a supported Python version; mixing conda and pip builds of the geospatial libraries in one environment can load two copies of GDAL and SHOULD be avoided.

### 9.5 Networks with proxies

pip, the STAC client and model downloads from registered URLs use the standard `HTTPS_PROXY` and `NO_PROXY` variables. The starter models of the zoo are built locally and need no network access.

## 10. Upgrading and uninstalling

Upgrade from PyPI with `python -m pip install --upgrade unbihexium`, or in a clone with `git pull` followed by `python -m pip install -e .` (plus the extras). When a lock file changed, reinstall from it as in Section 5.

Uninstalling the package does not remove the local model store. Remove cached models first, then the package:

```bash
unbihexium zoo clear --yes
python -m pip uninstall unbihexium
```

The model store is the directory `models` below `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium/models`) and can also be deleted manually.

## 11. Troubleshooting

| Symptom | Cause | Remedy |
| --- | --- | --- |
| `Error: No module named 'torch'; install PyTorch with pip install ...` from `unbihexium zoo build` or `unbihexium train` | The `torch` extra is not installed | `python -m pip install "unbihexium[torch]"` (Section 4.3) |
| `ModuleNotFoundError: No module named 'torch'` from `unbihexium predict <model id>` | Model ids and checkpoints need PyTorch | Install the `torch` extra, or predict with an ONNX export and `--backend onnx` |
| `ImportError` for zarr, pyarrow or fastapi | The corresponding extra is missing | Install the extra listed in Section 4.2 |
| `ERROR: Hashes are required in --require-hashes mode` | A requirements file without hashes, or an additional requirement given on the command line, in hash-checking mode | Install such packages in a separate command without `--require-hashes` |
| `unbihexium --help` does not list `infer` or `zoo download` | Both are hidden aliases kept for compatibility; they still work | Prefer `unbihexium predict` and `unbihexium zoo build`; see [docs/reference/cli.md](../reference/cli.md) |
| Commands or functions described in `docs/` are missing | PyPI 1.0.1 predates the main branch | Install from source (Section 7) |
| `pip` tries to compile rasterio, pyproj or onnxruntime | No wheel for the interpreter or platform, for example an unsupported Python version | Use CPython 3.10 to 3.14 on a platform with wheels |

## References

[1] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Python Software Foundation. venv: Creation of virtual environments. 2026. <https://docs.python.org/3/library/venv.html>

[4] The pip developers. Secure installs (hash-checking mode), pip documentation. 2026. <https://pip.pypa.io/en/stable/topics/secure-installs/>

[5] PyTorch Foundation. Get Started Locally. 2026. <https://pytorch.org/get-started/locally/>

[6] Astral Software. uv: pip compile and locking. 2026. <https://docs.astral.sh/uv/pip/compile/>

[7] Nathaniel J. Smith and Thomas Kluyver. PEP 517: A build-system independent format for source trees. Python Software Foundation, 2015. <https://peps.python.org/pep-0517/>

<!--
=============================================================================
End of file docs/getting_started/installation.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
