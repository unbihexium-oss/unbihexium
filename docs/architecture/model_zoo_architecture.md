# Model Zoo Architecture

## Purpose

Design of the model zoo infrastructure for model management, versioning, and distribution.

## Architecture Overview

```mermaid
graph TB
    subgraph "Model Storage"
        A1[Local Cache]
        A2[Remote Registry]
        A3[GitHub Releases]
    end

    subgraph "Model Manager"
        B1[Discovery]
        B2[Download]
        B3[Verification]
        B4[Loading]
    end

    subgraph "Runtime"
        C1[ONNX Session]
        C2[Inference]
    end

    A2 --> B1
    B1 --> B2
    B2 --> A1
    A1 --> B3
    B3 --> B4
    B4 --> C1
    C1 --> C2
```

## Model Identification

$$
\text{ModelID} = \text{Name} + \text{Variant} + \text{Version}
$$

Example: `ship_detector_base_v1.0.0`

## Storage Structure

| Path | Contents |
| ------ | ---------- |
| `model_zoo/assets/{variant}/{id}/` | Model artifacts |
| `model_zoo/manifests/{id}.json` | Model metadata |
| `model_zoo/cards/{id}.md` | Documentation |
| `model_zoo/inventory.yaml` | Catalog |

## Model Artifacts

| File | Purpose | Required |
| ------ | --------- | ---------- |
| `model.onnx` | ONNX model weights | Yes |
| `config.json` | Input/output spec | Yes |
| `model.sha256` | Integrity hash | Yes |
| `labels.json` | Class labels | If applicable |
| `metrics.json` | Performance data | Yes |

## Verification Flow

```mermaid
sequenceDiagram
    participant User
    participant Manager
    participant Cache
    participant Hash

    User->>Manager: Load model
    Manager->>Cache: Check cache
    Cache-->>Manager: Model found
    Manager->>Hash: Compute SHA256
    Hash-->>Manager: Hash value
    Manager->>Manager: Compare hashes
    alt Match
        Manager-->>User: Model loaded
    else Mismatch
        Manager-->>User: Verification error
    end
```

## Version Compatibility

$$
\text{Compatible}(v_{\text{lib}}, v_{\text{model}}) = \text{Major}(v_{\text{lib}}) = \text{Major}(v_{\text{model}})
$$
