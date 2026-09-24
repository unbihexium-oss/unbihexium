# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/visualization/output.py
# Title       : Quicklook PNG writing, world files and legends
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and Pillow; matplotlib
#               for legend_figure only
# =============================================================================
#
# Abstract
# --------
# Writing of display images and their legends:
#
#   save_png          write an (H, W), (H, W, 3) or (H, W, 4) uint8 image,
#                     optionally with an ESRI world file (.pgw)
#   world_file_lines  the six lines of a world file for an affine transform
#   quicklook         reduced-size RGB composite or colour-mapped band
#                     written as PNG
#   legend_image      legend of colour patches and labels drawn with Pillow
#   create_legend     legend array of colour patches (as in earlier releases)
#   legend_figure     matplotlib figure with a legend, imported lazily
#
# A world file holds x pixel size, row rotation, column rotation, y pixel
# size and the map coordinates of the CENTRE of the upper-left pixel, so
# half a pixel is added to the corner coordinates of the transform.
#
# References
# ----------
#   ESRI. World files for raster datasets (ArcGIS documentation).
#   W3C (2003). Portable Network Graphics (PNG) specification, 2nd ed.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# File system paths.
from pathlib import Path

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Colour mapping of single bands.
from unbihexium.visualization.colormaps import apply_colormap, color_to_rgb

# Colour composites.
from unbihexium.visualization.composites import rgb_composite


# World file lines for an affine transform (a, b, c, d, e, f).
def world_file_lines(transform: Sequence[float]) -> list[str]:
    # Coefficients of x = a col + b row + c, y = d col + e row + f.
    a, b, c, d, e, f = (float(v) for v in tuple(transform)[:6])
    # Centre of the upper-left pixel.
    x0 = c + a / 2.0 + b / 2.0
    # Its y coordinate.
    y0 = f + d / 2.0 + e / 2.0
    # Order of the world file: A, D, B, E, C, F.
    return [repr(v) for v in (a, d, b, e, x0, y0)]


# Write an 8-bit image as PNG.
def save_png(
    path: str | Path,  # Output file.
    image: NDArray[Any],  # (H, W), (H, W, 3) or (H, W, 4) uint8 image.
    transform: Sequence[float] | None = None,  # Affine transform for a world file.
) -> Path:  # Path of the PNG.
    # Pillow is imported only when writing.
    from PIL import Image

    # Image as array.
    arr = np.asarray(image)
    # Only bytes are written.
    if arr.dtype != np.uint8:
        # Explain the requirement.
        raise ValueError("image must be uint8; use to_uint8 or rgb_composite first")
    # Only grey, RGB and RGBA are supported.
    if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
        # Explain the requirement.
        raise ValueError(f"expected (H, W), (H, W, 3) or (H, W, 4), got shape {arr.shape}")
    # Output path.
    out = Path(path)
    # Create the directory.
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pillow picks L, RGB or RGBA from the shape.
    Image.fromarray(arr).save(out, format="PNG")
    # Write the world file next to the image.
    if transform is not None:
        # Lines of the world file.
        lines = world_file_lines(transform)
        # Same name with the .pgw extension.
        out.with_suffix(".pgw").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    # Return the path.
    return out


# Write a reduced-size quicklook of a band stack or a single band.
def quicklook(
    path: str | Path,  # Output PNG.
    image: NDArray[Any],  # (H, W) band or (C, H, W) stack.
    bands: Sequence[int | str] | str = (0, 1, 2),  # RGB bands of a stack.
    band_names: Sequence[str] | None = None,  # Names of the bands of a stack.
    max_size: int = 1024,  # Longest side of the quicklook in pixels.
    cmap: str = "viridis",  # Colour map of single bands.
    transform: Sequence[float] | None = None,  # Transform of the full image.
    **stretch: Any,  # Stretch options of rgb_composite.
) -> Path:  # Path of the PNG.
    # Image as array.
    arr = np.asarray(image)
    # The size limit must be positive.
    if max_size < 1:
        # Explain the requirement.
        raise ValueError("max_size must be positive")
    # Integer decimation step so that the longest side fits.
    step = max(1, int(np.ceil(max(arr.shape[-2:]) / max_size)))
    # Decimate rows and columns.
    small = arr[..., ::step, ::step]
    # Single bands are colour-mapped.
    if small.ndim == 2:
        # RGBA image.
        rgb = apply_colormap(small, cmap)
    # Stacks become colour composites.
    else:
        # RGBA composite.
        rgb = rgb_composite(small, bands, band_names, alpha=True, **stretch)
    # Pixel vectors of the decimated grid.
    scaled = None
    # Scale the transform when given.
    if transform is not None:
        # Coefficients of the full-resolution grid.
        a, b, c, d, e, f = (float(v) for v in tuple(transform)[:6])
        # Every pixel vector grows by the step.
        scaled = (a * step, b * step, c, d * step, e * step, f)
    # Write the PNG.
    return save_png(path, rgb, scaled)


# Legend of colour patches and text labels drawn with Pillow.
def legend_image(
    labels: Sequence[str],  # Class names.
    colors: Sequence[str | Sequence[int]],  # One colour per class.
    patch: tuple[int, int] = (24, 16),  # Patch width and height in pixels.
    padding: int = 6,  # Space around patches and text.
    width: int | None = None,  # Image width; None fits the longest label.
    background: tuple[int, int, int] = (255, 255, 255),  # Background colour.
    text_color: tuple[int, int, int] = (0, 0, 0),  # Label colour.
) -> NDArray[np.uint8]:  # (H, W, 3) image.
    # Pillow drawing.
    from PIL import Image, ImageDraw, ImageFont

    # One colour per label.
    if len(labels) != len(colors):
        # Explain the requirement.
        raise ValueError("labels and colors must have the same length")
    # Default bitmap font of Pillow.
    font = ImageFont.load_default()
    # Width of the longest label.
    text_w = max((int(font.getlength(str(t))) for t in labels), default=0)
    # Image width.
    w = width if width is not None else patch[0] + text_w + 3 * padding
    # Height of one row.
    row = patch[1] + padding
    # Image height.
    h = padding + row * len(labels)
    # Canvas.
    img = Image.new("RGB", (w, h), background)
    # Drawing context.
    draw = ImageDraw.Draw(img)
    # Draw every entry.
    for k, (label, color) in enumerate(zip(labels, colors)):
        # Top of the row.
        y = padding + k * row
        # Corners of the patch.
        box = [padding, y, padding + patch[0] - 1, y + patch[1] - 1]
        # Colour patch with a thin outline.
        draw.rectangle(box, fill=color_to_rgb(color), outline=text_color)
        # Label to the right of the patch.
        draw.text((2 * padding + patch[0], y + 2), str(label), fill=text_color, font=font)
    # Return the pixels.
    return np.asarray(img, dtype=np.uint8)


# Legend array of colour patches and labels, with the geometry of earlier releases.
def create_legend(
    labels: Sequence[str],  # Class names.
    colors: Sequence[tuple[int, int, int]],  # Class colours.
    size: tuple[int, int] = (200, 20),  # Width and height of each patch.
) -> NDArray[np.uint8]:  # (len(labels) * height, width + 100, 3) image.
    # Pillow drawing.
    from PIL import Image, ImageDraw, ImageFont

    # One colour per label.
    if len(labels) != len(colors):
        # Explain the requirement.
        raise ValueError("labels and colors must have the same length")
    # Patch width and row height.
    pw, ph = size
    # White canvas with room for the labels.
    img = Image.new("RGB", (pw + 100, ph * len(labels)), (255, 255, 255))
    # Drawing context.
    draw = ImageDraw.Draw(img)
    # Default bitmap font of Pillow.
    font = ImageFont.load_default()
    # Draw every entry.
    for k, (label, color) in enumerate(zip(labels, colors)):
        # Patch filling the row.
        draw.rectangle([0, k * ph, pw - 1, (k + 1) * ph - 1], fill=color_to_rgb(color))
        # Label right of the patch.
        draw.text((pw + 4, k * ph + 2), str(label), fill=(0, 0, 0), font=font)
    # Return the pixels.
    return np.asarray(img, dtype=np.uint8)


# Matplotlib figure with a legend of colour patches.
def legend_figure(
    labels: Sequence[str],  # Class names.
    colors: Sequence[str | Sequence[int]],  # One colour per class.
    title: str | None = None,  # Legend title.
) -> Any:  # matplotlib.figure.Figure.
    # Matplotlib is optional.
    try:
        # Figure without the global pyplot state; matplotlib is optional and
        # is not part of the type check environment.
        from matplotlib.figure import Figure  # pyright: ignore[reportMissingImports]

        # Colour patches for the legend.
        from matplotlib.patches import Patch  # pyright: ignore[reportMissingImports]
    # Explain how to install it.
    except ImportError as exc:
        # Re-raise with a hint.
        raise ImportError("legend_figure requires matplotlib: pip install matplotlib") from exc
    # One colour per label.
    if len(labels) != len(colors):
        # Explain the requirement.
        raise ValueError("labels and colors must have the same length")
    # Figure without pyplot state.
    fig = Figure(figsize=(3, 0.3 * len(labels) + 0.5))
    # Colours in [0, 1] as matplotlib expects.
    rgb = [np.divide(color_to_rgb(c), 255.0) for c in colors]
    # Legend handles.
    handles = [Patch(color=c, label=str(t)) for t, c in zip(labels, rgb)]
    # Draw the legend alone.
    fig.legend(handles=handles, loc="center", frameon=False, title=title)
    # Return the figure.
    return fig


# =============================================================================
# End of module src/unbihexium/visualization/output.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
