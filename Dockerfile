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
# Builds the Unbihexium container image with the ONNX Runtime inference
# backend and the FastAPI REST service, installed from the pinned versions in
# requirements.txt. Model weights are not included; they are downloaded into
# the cache directory on first use and verified with SHA256. The first line
# selects the Dockerfile syntax and must stay the first line of the file.
#
# Base image
#   python:3.14.7-slim-trixie: CPython 3.14.7 on Debian 13 (trixie), the
#   current stable Debian release in September 2026. Dependabot proposes base
#   image updates monthly (.github/dependabot.yml).
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
#       uvicorn unbihexium.serving.app:app --host 0.0.0.0 --port 8000
#
#   Persist downloaded models between runs with a volume:
#     docker run --rm -v unbihexium-cache:/home/unbihexium/.cache/unbihexium ...
#
# Security
#   - Multi-stage build: pip caches and build files never reach the runtime
#     image.
#   - Only binary wheels are installed (--only-binary=:all:), so no code from
#     source distributions is built or executed during the installation.
#   - No system packages are installed: rasterio, pyproj, shapely and
#     onnxruntime wheels bundle GDAL, PROJ, GEOS and their other native
#     libraries.
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
FROM python:3.14.7-slim-trixie AS builder

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

# Install the locked dependencies first. This layer is cached as long as
# requirements.txt does not change, which keeps rebuilds after source changes
# fast.
COPY requirements.txt ./
# Upgrade pip, then install binary wheels only from the lock file.
RUN python -m pip install --upgrade pip \
    && python -m pip install --only-binary=:all: -r requirements.txt

# Install Unbihexium itself without resolving dependencies again, so exactly
# the locked versions are used. The licence files are required by the package
# metadata (PEP 639).
COPY pyproject.toml README.md LICENSE.txt NOTICE NOTICE.md ./
# Copy the package sources.
COPY src/ ./src/
# Install the package without dependencies and verify that the installed
# requirements are consistent.
RUN python -m pip install --no-deps . \
    && python -m pip check

# -----------------------------------------------------------------------------
# Stage 2: runtime
# Contains only the Python runtime and the virtual environment.
# -----------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie AS runtime

# Package version recorded in the image metadata.
ARG VERSION=1.0.1
# Git commit the image was built from.
ARG VCS_REF=unknown
# Build time in RFC 3339 format.
ARG BUILD_DATE=unknown

# Open Container Initiative image annotations.
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.title="Unbihexium" \
      org.opencontainers.image.description="Production-grade Earth Observation, Geospatial, Remote Sensing, and SAR Python library" \
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
# point Unbihexium at its home and model cache directories.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UNBIHEXIUM_HOME=/home/unbihexium \
    UNBIHEXIUM_CACHE=/home/unbihexium/.cache/unbihexium

# Drop root privileges for everything that follows, including the container.
USER unbihexium
# Start in the home directory of the unprivileged user.
WORKDIR /home/unbihexium

# Model cache. Mount a volume here to keep downloaded models between runs.
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
