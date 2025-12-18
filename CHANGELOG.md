# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-18

### Added

- Core abstractions: Raster, Vector, Tile, Scene, SensorModel, Product
- Pipeline framework with provenance tracking
- Registry system for capabilities, models, and pipelines
- AI products: ship detection, building detection, aircraft detection, vehicle detection
- Segmentation: change detection, water detection, crop detection, greenhouse detection
- Super-resolution pipeline
- Spectral indices: NDVI, NDWI, NBR, EVI, SAVI, MSI
- Geostatistics: variogram analysis, Ordinary Kriging, Universal Kriging
- Spatial autocorrelation: Moran's I, Geary's C
- Suitability analysis: AHP, weighted overlay
- Network analysis: routing, accessibility, service areas
- Zonal statistics
- Model zoo with 5 tiny models for smoke tests
- CLI with zoo, pipeline, and index commands
- Documentation site with MkDocs
- CI/CD workflows: CI, coverage, docs, security, scorecard, SBOM, CodeQL, release, publish
- Security policies and responsible use guidelines

### Security

- SHA256 verification for model artifacts
- SBOM generation with CycloneDX
- OpenSSF Scorecard integration
- CodeQL scanning
- Signed releases with attestations

---

## Navigation

[README](README.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md) | [License](LICENSE)
