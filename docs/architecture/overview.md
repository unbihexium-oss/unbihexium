# Architecture Overview

## System Architecture

Unbihexium is designed as a modular, extensible framework for geospatial AI pipelines. The architecture follows a layered design with clear separation of concerns.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        A1[CLI Interface]
        A2[Python API]
        A3[REST API]
    end
    
    subgraph "Orchestration Layer"
        B1[Pipeline Orchestrator]
        B2[Task Scheduler]
        B3[Resource Manager]
    end
    
    subgraph "Core Services"
        C1[Capability Registry]
        C2[Model Zoo Manager]
        C3[Data Loader]
        C4[Inference Engine]
    end
    
    subgraph "Processing Layer"
        D1[Tiling Engine]
        D2[Preprocessing]
        D3[Postprocessing]
        D4[Georeferencing]
    end
    
    subgraph "Storage Layer"
        E1[Model Cache]
        E2[Result Store]
        E3[Config Store]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> C1
    B1 --> C2
    B2 --> C3
    B3 --> C4
    
    C4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    C2 --> E1
    D4 --> E2
    B1 --> E3
```

---

## Component Details

### Pipeline Orchestrator

Coordinates end-to-end processing workflows:

$$
\text{Pipeline} = \{S_1, S_2, ..., S_n\} \quad \text{where } S_i \text{ is a processing stage}
$$

### Capability Registry

Central registry of 130 base models across 12 domains:

| Attribute | Description |
|-----------|-------------|
| capability_id | Unique identifier |
| domain | Capability domain (01-12) |
| task | Task type (detection, segmentation, etc.) |
| models | Available model variants |
| metadata | Version, license, dependencies |

### Model Zoo Manager

Manages 520 model artifacts:

| Function | Description |
|----------|-------------|
| list() | Enumerate available models |
| get() | Load model by ID |
| download() | Fetch from remote |
| verify() | SHA256 validation |
| cache() | Local caching |

### Inference Engine

ONNX Runtime-based inference:

$$
\hat{y} = f_\theta(x) \quad \text{where } \theta \in \{tiny, base, large, mega\}
$$

---

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant R as Registry
    participant M as Model Zoo
    participant I as Inference
    
    U->>P: run(config)
    P->>R: resolve_capability()
    R->>M: get_model(id, variant)
    M->>M: verify_checksum()
    M->>I: load_model()
    I->>I: run_inference()
    I->>P: predictions
    P->>U: results
```

---

## Memory Model

Total memory consumption:

$$
M_{total} = M_{base} + M_{model} + N_{tiles} \times M_{tile}
$$

| Component | Tiny | Base | Large | Mega |
|-----------|------|------|-------|------|
| Model | 2 MB | 10 MB | 30 MB | 100 MB |
| Per Tile | 1 MB | 4 MB | 16 MB | 64 MB |
| Overhead | 50 MB | 100 MB | 200 MB | 500 MB |

---

## Extensibility

New capabilities can be added via:

1. Model registration in capability registry
2. Pipeline stage implementation
3. Pre/post-processor plugins
4. Custom data loaders
