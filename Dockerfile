# syntax=docker/dockerfile:1
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# File        : Dockerfile
# Title       : Unbihexium container image
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Format      : Dockerfile, built by Docker BuildKit
# =============================================================================
#
# Abstract
# --------
# Builds the Unbihexium container image with the command line interface, the
# FastAPI REST service, the CPU build of PyTorch and ONNX Runtime. The
# dependencies come from two hashed lock files: .github/requirements/
# requirements-docker.txt (runtime, onnx and serving extras and the
# dependencies of PyTorch, from PyPI) and .github/requirements/
# requirements-ci-torch.txt (PyTorch itself, from the PyTorch CPU index).
# Every wheel is checked against its SHA-256 hash.
#
# Model weights are not part of the image. On first use of a model the
# library builds its deterministic starter weights from the model catalogue,
# checks them against the published SHA-256 digest and stores them in the
# cache directory UNBIHEXIUM_CACHE; nothing is downloaded. The starter models
# are untrained, except the spectral index models (see README.md).
#
# Base image
#   python:3.14.7-slim-trixie: CPython 3.14.7 on Debian 13 (trixie), the
#   current stable Debian release in September 2026. Both stages pin the
#   image by the digest of its multi-platform manifest list, so a rebuild
#   uses exactly the reviewed image (OpenSSF Scorecard, Pinned-Dependencies).
#   Dependabot proposes tag and digest updates monthly
#   (.github/dependabot.yml).
#
# Build
#   docker build -t unbihexium:local .
#
#   Optional build arguments add OCI metadata:
#     --build-arg VERSION=1.0.1
#     --build-arg VCS_REF="$(git rev-parse HEAD)"
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
#
# Run
#   Command line interface:
#     docker run --rm unbihexium:local unbihexium --help
#
#   REST API on http://localhost:8000 (OpenAPI docs at /docs):
#     docker run --rm -p 8000:8000 unbihexium:local \
#       unbihexium serve --host 0.0.0.0 --port 8000
#
#   Keep built models between runs with a volume:
#     docker run --rm -v unbihexium-cache:/home/unbihexium/.cache/unbihexium ...
#
# Security
#   - Multi-stage build: pip caches and build files never reach the runtime
#     image.
#   - Only binary wheels are installed (--only-binary=:all:), so no code from
#     source distributions is built or executed during the installation.
#   - No system packages are installed: the rasterio, pyproj, shapely,
#     onnxruntime and torch wheels bundle GDAL, PROJ, GEOS and their other
#     native libraries.
#   - The container runs as the unprivileged user "unbihexium" (UID 1000).
#   - The image is scanned with Grype by .github/workflows/container-scan.yml
#     and an SBOM is attached by .github/workflows/docker.yml.
#
# Docker reads a `#` only at the start of a line as a comment, so every
# comment stands on its own line above the instruction it explains.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# Creates a virtual environment in /opt/venv with all dependencies and the
# Unbihexium package.
# -----------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS builder

# Disable the pip cache and version check, silence the root user warning and
# skip .pyc files, which keeps the build stage small and quiet.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

# Working directory for the package sources during the build.
WORKDIR /build

# Create the virtual environment that is later copied into the runtime stage.
RUN python -m venv /opt/venv
# Put the virtual environment first on PATH so that python and pip use it.
ENV PATH="/opt/venv/bin:${PATH}"

# Install the locked dependencies first. This layer is cached as long as the
# lock files do not change, which keeps rebuilds after source changes fast.
COPY .github/requirements/requirements-docker.txt .github/requirements/requirements-ci-torch.txt ./
# Install binary wheels only, each checked against its hash in the lock file:
# first the runtime dependencies from PyPI, then the CPU build of PyTorch
# without dependencies (they are in the first file) from the PyTorch index
# named in requirements-ci-torch.txt.
RUN python -m pip install --only-binary=:all: --require-hashes -r requirements-docker.txt \
    && python -m pip install --only-binary=:all: --require-hashes --no-deps -r requirements-ci-torch.txt

# Install Unbihexium itself without resolving dependencies again, so exactly
# the locked versions are used. The licence files are required by the package
# metadata (PEP 639).
COPY pyproject.toml README.md LICENSE.txt NOTICE NOTICE.md ./
# Copy the package sources.
COPY src/ ./src/
# Separate build environment with the hash-pinned build backend, so that
# building the wheel fetches nothing unpinned.
COPY .github/requirements/requirements-build.txt ./
# Create it and install the backend, each file checked against its hash.
RUN python -m venv /opt/build \
    && /opt/build/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-build.txt
# Build the wheel of the package with that backend, install it without
# dependencies, verify that the installed requirements are consistent and
# that the REST service, PyTorch and ONNX Runtime import.
RUN /opt/build/bin/python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/dist . \
    && python -m pip install --no-deps /tmp/dist/unbihexium-*.whl \
    && python -m pip check \
    && python -c "import torch, onnxruntime, unbihexium.serving.app"

# -----------------------------------------------------------------------------
# Stage 2: runtime
# Contains only the Python runtime and the virtual environment.
# -----------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS runtime

# Package version recorded in the image metadata.
ARG VERSION=1.0.1
# Git commit the image was built from.
ARG VCS_REF=unknown
# Build time in RFC 3339 format.
ARG BUILD_DATE=unknown

# Open Container Initiative image annotations.
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.title="Unbihexium" \
      org.opencontainers.image.description="Earth observation, geospatial, remote sensing and SAR library for Python: command line interface and REST service" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MPL-2.0" \
      org.opencontainers.image.vendor="Unbihexium OSS Foundation" \
      org.opencontainers.image.authors="Unbihexium OSS Foundation <yunus.z.imanov@helsinki.fi>" \
      org.opencontainers.image.url="https://github.com/unbihexium-oss/unbihexium" \
      org.opencontainers.image.source="https://github.com/unbihexium-oss/unbihexium" \
      org.opencontainers.image.documentation="https://github.com/unbihexium-oss/unbihexium/tree/main/docs" \
      org.opencontainers.image.base.name="docker.io/library/python:3.14.7-slim-trixie"

# Unprivileged user with a fixed UID and GID, so that mounted volumes can be
# given matching ownership on the host.
RUN groupadd --gid 1000 unbihexium \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin unbihexium

# Copy the ready virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Use the virtual environment, write logs unbuffered, skip .pyc files and
# point Unbihexium at its model cache directory.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UNBIHEXIUM_CACHE=/home/unbihexium/.cache/unbihexium

# Drop root privileges for everything that follows, including the container.
USER unbihexium
# Start in the home directory of the unprivileged user.
WORKDIR /home/unbihexium

# Model cache. Mount a volume here to keep built models between runs.
RUN mkdir -p "${UNBIHEXIUM_CACHE}"
# Declare the cache as a volume so that it survives container restarts.
VOLUME ["/home/unbihexium/.cache/unbihexium"]

# Port of the REST API when started with uvicorn.
EXPOSE 8000

# The check confirms that the package and its native libraries load. The
# Compose file overrides it with a request to the /health endpoint of the API.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import unbihexium"]

# Default: show the command line help. Override the command to run the API or
# any other unbihexium command.
CMD ["unbihexium", "--help"]

# =============================================================================
# End of file Dockerfile
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
