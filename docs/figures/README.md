# Documentation Figures

This directory contains generated figures for documentation.

## Generation

Figures are generated deterministically using `scripts/generate_figures.py`:

```bash
python scripts/generate_figures.py
```

## Available Figures

| Figure | Description | Used In |
|--------|-------------|---------|
| ndvi_colormap.png | NDVI colormap example | index.md |
| variogram.png | Variogram plot | geostat.md |
| detection_example.png | Detection visualization | detection.md |
| pipeline_timing.png | Pipeline performance | architecture.md |

## Figure Requirements

All figures must be:

1. Deterministically generated (seeded random)
2. Resolution: 150 DPI minimum
3. Format: PNG with transparent background where appropriate

## Mathematical Notation

Figures may include LaTeX-rendered formulas:

$$f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

## Diagram Style

```mermaid
graph LR
    A[Script] --> B[Generate]
    B --> C[Save PNG]
    C --> D[docs/figures/]
```
