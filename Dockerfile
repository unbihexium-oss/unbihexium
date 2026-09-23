# syntax=docker/dockerfile:1
# =============================================================================
# Unbihexium container image
# =============================================================================
#
# Contents
#   The image contains Unbihexium with the ONNX Runtime inference backend and
#   the FastAPI REST service, installed from the pinned versions in
#   requirements.txt. Model weights are not included; they are downloaded into
#   the cache directory on first use and verified with SHA256.
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
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# Creates a virtual environment in /opt/venv with all dependencies and the
# Unbihexium package.
# -----------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install the locked dependencies first. This layer is cached as long as
# requirements.txt does not change, which keeps rebuilds after source changes
# fast.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --only-binary=:all: -r requirements.txt

# Install Unbihexium itself without resolving dependencies again, so exactly
# the locked versions are used. The licence files are required by the package
# metadata (PEP 639).
COPY pyproject.toml README.md LICENSE.txt NOTICE NOTICE.md ./
COPY src/ ./src/
RUN python -m pip install --no-deps . \
    && python -m pip check

# -----------------------------------------------------------------------------
# Stage 2: runtime
# Contains only the Python runtime and the virtual environment.
# -----------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie AS runtime

ARG VERSION=1.0.1
ARG VCS_REF=unknown
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

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UNBIHEXIUM_HOME=/home/unbihexium \
    UNBIHEXIUM_CACHE=/home/unbihexium/.cache/unbihexium

USER unbihexium
WORKDIR /home/unbihexium

# Model cache. Mount a volume here to keep downloaded models between runs.
RUN mkdir -p "${UNBIHEXIUM_CACHE}"
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
