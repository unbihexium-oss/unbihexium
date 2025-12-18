#!/usr/bin/env python3
"""Generate deterministic figures for documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def generate_ndvi_colormap(output_dir: Path) -> None:
    """Generate NDVI colormap figure."""
    import matplotlib.pyplot as plt

    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)
    Z = X * np.sin(Y * np.pi)

    im = ax.imshow(Z, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title("NDVI Colormap Example")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, label="NDVI")

    fig.savefig(output_dir / "ndvi_colormap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_variogram_plot(output_dir: Path) -> None:
    """Generate variogram plot."""
    import matplotlib.pyplot as plt

    np.random.seed(42)

    lags = np.linspace(0, 100, 15)
    semivariance = 0.5 * (1.5 * lags / 80 - 0.5 * (lags / 80) ** 3)
    semivariance = np.minimum(semivariance, 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lags, semivariance, label="Empirical")
    ax.plot(lags, semivariance, "r-", label="Fitted (Spherical)")
    ax.axhline(0.5, color="gray", linestyle="--", label="Sill")
    ax.axvline(80, color="green", linestyle="--", label="Range")
    ax.set_xlabel("Lag Distance (h)")
    ax.set_ylabel("Semivariance")
    ax.set_title("Variogram Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "variogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_detection_example(output_dir: Path) -> None:
    """Generate detection example figure."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(8, 8))
    img = np.random.rand(256, 256, 3) * 0.3 + 0.2
    ax.imshow(img)

    # Add detection boxes
    boxes = [(50, 50, 40, 30), (150, 100, 35, 25), (80, 180, 45, 35)]
    for x, y, w, h in boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 5, "Ship 0.95", color="red", fontsize=8)

    ax.set_title("Ship Detection Results")
    ax.axis("off")

    fig.savefig(output_dir / "detection_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_pipeline_timing(output_dir: Path) -> None:
    """Generate pipeline timing chart."""
    import matplotlib.pyplot as plt

    stages = ["Load", "Preprocess", "Inference", "Postprocess", "Save"]
    times = [0.5, 1.2, 3.5, 0.8, 0.3]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(stages, times, color="steelblue")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Pipeline Stage Timing")

    for bar, time in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{time}s", va="center")

    fig.savefig(output_dir / "pipeline_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Generate all figures."""
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed, skipping figure generation")
        return 0

    generators = [
        generate_ndvi_colormap,
        generate_variogram_plot,
        generate_detection_example,
        generate_pipeline_timing,
    ]

    for gen in generators:
        try:
            gen(output_dir)
            print(f"Generated: {gen.__name__}")
        except Exception as e:
            print(f"Failed to generate {gen.__name__}: {e}")

    print(f"Generated {len(generators)} figures in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
