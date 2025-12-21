# Unbihexium Production Dockerfile
# Multi-stage build for optimized image size and security

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.14-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Create virtual environment and install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install package with production dependencies
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir .

# Install optional serving dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.14-slim AS runtime

LABEL maintainer="Unbihexium OSS Foundation <opensource@unbihexium.org>"
LABEL description="Production-grade Earth Observation and Geospatial AI"
LABEL version="1.0.0"
LABEL license="Apache-2.0"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 unbihexium
WORKDIR /home/unbihexium
USER unbihexium

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UNBIHEXIUM_HOME=/home/unbihexium
ENV UNBIHEXIUM_CACHE=/home/unbihexium/.cache/unbihexium

# Create cache directory
RUN mkdir -p /home/unbihexium/.cache/unbihexium

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import unbihexium; print(unbihexium.__version__)" || exit 1

# Expose API port
EXPOSE 8000

# Default command: run CLI help
CMD ["unbihexium", "--help"]

# Alternative: Run API server
# CMD ["uvicorn", "unbihexium.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
