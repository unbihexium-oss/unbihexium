# Examples

This directory contains example scripts, notebooks, and applications demonstrating the Unbihexium library capabilities.

## Directory Structure

```
examples/
├── notebooks/          # Jupyter notebooks for interactive exploration
│   ├── 01_getting_started.ipynb
│   ├── 02_model_inference.ipynb
│   └── ...
├── scripts/            # Standalone Python scripts
│   └── batch_processing.py
└── serving/            # Model serving examples
    ├── fastapi_server.py
    └── docker-compose.yml
```

## Quick Start

### Prerequisites

```bash
pip install unbihexium[torch]
```

### Running Examples

```bash
# Run a script
python examples/scripts/batch_processing.py --input data/ --output results/

# Start Jupyter
jupyter notebook examples/notebooks/
```

## Notebooks Overview

| Notebook | Description | Level |
| ---------- | ------------- | ------- |
| 01_getting_started | Installation and basic usage | Beginner |
| 02_model_inference | Running inference with pre-trained models | Beginner |
| 03_pipeline_basics | Building processing pipelines | Intermediate |
| 04_change_detection | Bi-temporal change detection | Intermediate |
| 05_sar_processing | SAR amplitude and InSAR | Advanced |
| 06_custom_models | Training custom models | Advanced |

## Scripts Overview

| Script | Description |
| -------- | ------------- |
| batch_processing.py | Process multiple images in parallel |
| model_benchmark.py | Benchmark model performance |

## Serving Overview

| Example | Description |
| --------- | ------------- |
| fastapi_server.py | REST API for model inference |
| docker-compose.yml | Containerized deployment |

## License

Examples are licensed under MPL-2.0, same as the main library.
