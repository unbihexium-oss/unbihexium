# Table of Contents

## Purpose

Navigation guide for Unbihexium documentation.

## Documentation Map

```mermaid
graph LR
    A[Start] --> B[Getting Started]
    B --> C[Tutorials]
    C --> D[Reference]
    D --> E[Architecture]
    E --> F[Capabilities]
    F --> G[Model Zoo]
    G --> H[Security]
    H --> I[Operations]
```

## Coverage Statistics

$$
\text{Completeness} = \frac{\text{Documented Features}}{\text{Total Features}} = 100\%
$$

## Contents

| Section | Pages | Description |
| --------- | ------- | ------------- |
| Getting Started | 3 | Installation, quickstart, config |
| Tutorials | 5 | Hands-on guides |
| Reference | 5 | API, CLI, schemas |
| Architecture | 5 | System design |
| Capabilities | 13 | Domain features |
| Model Zoo | 5 | Model documentation |
| Security | 5 | Security practices |
| Operations | 3 | Deployment, CI/CD |

## Getting Started

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [Configuration](getting_started/configuration.md)

## Tutorials

- [Tutorial Index](tutorials/index.md)
- [End-to-End Pipeline](tutorials/end_to_end.md)
- [Object Detection](tutorials/detection.md)
- [Spectral Indices](tutorials/indices.md)
- [Geostatistics](tutorials/geostat.md)
- [Notebooks](tutorials/notebooks.md)

## Reference

- [CLI Reference](reference/cli.md)
- [API Reference](reference/api.md)
- [Config Schema](reference/config_schema.md)
- [Data Formats](reference/data_formats.md)
- [Model Zoo Reference](reference/model_zoo.md)

## Architecture

- [Overview](architecture/overview.md)
- [Pipeline Framework](architecture/pipeline_framework.md)
- [Capability Registry](architecture/capability_registry.md)
- [Model Zoo Architecture](architecture/model_zoo_architecture.md)
- [Security Model](architecture/security_model.md)

## Capabilities (12 Domains)

- [Capabilities Index](capabilities/index.md)
- [01: AI Products](capabilities/01_ai_products.md)
- [02: Tourism/Data Processing](capabilities/02_tourism_data_processing.md)
- [03: Indices/Flood/Water](capabilities/03_indices_flood_water.md)
- [04: Environment/Forestry](capabilities/04_environment_forestry_image_processing.md)
- [05: Asset Management/Energy](capabilities/05_asset_management_energy.md)
- [06: Urban/Agriculture](capabilities/06_urban_agriculture.md)
- [07: Risk/Defense (Neutral)](capabilities/07_risk_defense_neutral.md)
- [08: Value-Added Imagery](capabilities/08_value_added_imagery.md)
- [09: Benefits Narrative](capabilities/09_benefits_narrative.md)
- [10: Satellite Imagery Features](capabilities/10_satellite_imagery_features.md)
- [11: Resolution/Metadata/QA](capabilities/11_resolution_metadata_qa.md)
- [12: Radar/SAR](capabilities/12_radar_sar.md)

## Model Zoo

- [Model Catalog](model_zoo/model_catalog.md)
- [Training](model_zoo/training.md)
- [Inference](model_zoo/inference.md)
- [Download and Verify](model_zoo/download_and_verify.md)
- [Distribution](model_zoo/distribution.md)
- [Licensing](model_zoo/licensing_and_provenance.md)
- [Adding Models](model_zoo/how_to_add_models.md)

## Security

- [Responsible Use](security/responsible_use.md)
- [Vulnerability Management](security/vulnerability_management.md)
- [Supply Chain](security/supply_chain_security.md)
- [Model Integrity](security/model_integrity.md)
- [Secrets and Tokens](security/secrets_and_tokens.md)

## Operations

- [Docker](operations/docker.md)
- [Releasing](operations/releasing.md)
- [CI/CD](operations/ci_cd.md)

## Resources

- [Glossary](glossary.md)
- [FAQ](faq.md)
- [Style Guide](style_guide.md)
