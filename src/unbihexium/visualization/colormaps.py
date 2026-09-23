# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/visualization/colormaps.py
# Title       : Colour maps for continuous indices and class maps
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Colour tables that need no plotting library:
#
#   COLORMAPS            named continuous colour maps as control points
#   colormap_lut         (N, 3) uint8 lookup table of a colour map
#   apply_colormap       values to RGBA, NaN and nodata transparent
#   classify_colors      values to RGBA by class breaks (e.g. NDVI classes)
#   WORLDCOVER_PALETTE   ESA WorldCover 10 m class colours
#   DEFAULT_PALETTE      distinct colours for arbitrary class ids
#   colorize_mask        class map to RGB with a palette
#   colorize_classes     class map to RGBA with transparent nodata
#   hex_to_rgb           "#rrggbb" to an RGB tuple
#
# The diverging and sequential maps use the ColorBrewer schemes RdYlGn,
# RdBu, BrBG and Blues (11 and 9 classes) as control points; "viridis" is
# interpolated from nine samples of the perceptually uniform matplotlib map.
# Colour maps are linear between control points in sRGB.
#
# References
# ----------
#   Harrower, M., Brewer, C. A. (2003). ColorBrewer.org: an online tool for
#     selecting colour schemes for maps. The Cartographic Journal 40(1),
#     27-37.
#   van der Walt, S., Smith, N. (2015). A better default colormap for
#     matplotlib (viridis). SciPy 2015 conference.
#   Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo,
#     doi:10.5281/zenodo.7254221 (product user manual, class legend).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and sequences.
from typing import Any, Mapping, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Named colour maps as hex control points, evenly spaced from 0 to 1.
COLORMAPS: dict[str, tuple[str, ...]] = {
    "greys": ("#000000", "#ffffff"),  # Black to white.
    "viridis": (  # Perceptually uniform, dark blue to yellow.
        "#440154",  # 0.000.
        "#472d7b",  # 0.125.
        "#3b528b",  # 0.250.
        "#2c728e",  # 0.375.
        "#21918c",  # 0.500.
        "#28ae80",  # 0.625.
        "#5ec962",  # 0.750.
        "#addc30",  # 0.875.
        "#fde725",  # 1.000.
    ),  # End of viridis.
    "rdylgn": (  # ColorBrewer RdYlGn, 11 classes.
        "#a50026",  # Dark red.
        "#d73027",  # Red.
        "#f46d43",  # Orange red.
        "#fdae61",  # Orange.
        "#fee08b",  # Light orange.
        "#ffffbf",  # Pale yellow.
        "#d9ef8b",  # Light green.
        "#a6d96a",  # Yellow green.
        "#66bd63",  # Green.
        "#1a9850",  # Dark green.
        "#006837",  # Darkest green.
    ),  # End of RdYlGn, the usual NDVI scheme.
    "rdbu": (  # ColorBrewer RdBu, 11 classes.
        "#67001f",  # Dark red.
        "#b2182b",  # Red.
        "#d6604d",  # Light red.
        "#f4a582",  # Salmon.
        "#fddbc7",  # Pale red.
        "#f7f7f7",  # Neutral.
        "#d1e5f0",  # Pale blue.
        "#92c5de",  # Light blue.
        "#4393c3",  # Blue.
        "#2166ac",  # Dark blue.
        "#053061",  # Darkest blue.
    ),  # End of RdBu, for anomalies and change.
    "brbg": (  # ColorBrewer BrBG, 11 classes.
        "#543005",  # Dark brown.
        "#8c510a",  # Brown.
        "#bf812d",  # Light brown.
        "#dfc27d",  # Tan.
        "#f6e8c3",  # Pale tan.
        "#f5f5f5",  # Neutral.
        "#c7eae5",  # Pale teal.
        "#80cdc1",  # Light teal.
        "#35978f",  # Teal.
        "#01665e",  # Dark teal.
        "#003c30",  # Darkest teal.
    ),  # End of BrBG, for moisture and water indices.
    "blues": (  # ColorBrewer Blues, 9 classes.
        "#f7fbff",  # Almost white.
        "#deebf7",  # Very light blue.
        "#c6dbef",  # Light blue.
        "#9ecae1",  # Sky blue.
        "#6baed6",  # Medium light blue.
        "#4292c6",  # Medium blue.
        "#2171b5",  # Blue.
        "#08519c",  # Dark blue.
        "#08306b",  # Darkest blue.
    ),  # End of Blues, for water depth and frequency.
}  # End of the colour maps.

# Colour maps recommended for common indices.
INDEX_COLORMAPS = {
    "ndvi": "rdylgn",  # Vegetation greenness.
    "evi": "rdylgn",  # Enhanced vegetation index.
    "ndwi": "brbg",  # Water and moisture.
    "mndwi": "brbg",  # Modified water index.
    "ndbi": "rdbu",  # Built-up index, diverging around zero.
    "nbr": "rdylgn",  # Burn ratio.
    "dnbr": "rdbu",  # Burn severity change.
}  # End of the index colour maps.

# ESA WorldCover class values and colours.
WORLDCOVER_PALETTE: dict[int, tuple[str, str]] = {
    10: ("Tree cover", "#006400"),  # Dark green.
    20: ("Shrubland", "#ffbb22"),  # Orange.
    30: ("Grassland", "#ffff4c"),  # Yellow.
    40: ("Cropland", "#f096ff"),  # Pink.
    50: ("Built-up", "#fa0000"),  # Red.
    60: ("Bare / sparse vegetation", "#b4b4b4"),  # Grey.
    70: ("Snow and ice", "#f0f0f0"),  # Near white.
    80: ("Permanent water bodies", "#0064c8"),  # Blue.
    90: ("Herbaceous wetland", "#0096a0"),  # Teal.
    95: ("Mangroves", "#00cf75"),  # Green.
    100: ("Moss and lichen", "#fae6a0"),  # Beige.
}  # End of the WorldCover legend.

# Distinct colours for class ids 0 to 5, as in earlier releases.
DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # Background.
    1: (255, 0, 0),  # Class 1.
    2: (0, 255, 0),  # Class 2.
    3: (0, 0, 255),  # Class 3.
    4: (255, 255, 0),  # Class 4.
    5: (255, 0, 255),  # Class 5.
}  # End of the default palette.


# "#rrggbb" to an (r, g, b) tuple of integers.
def hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Remove the leading hash.
    h = color.lstrip("#")
    # Exactly six hexadecimal digits are expected.
    if len(h) != 6:
        # Explain the requirement.
        raise ValueError(f"expected a colour of the form #rrggbb, got {color!r}")
    # Parse the three channels.
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# Colour as an (r, g, b) tuple from a hex string or a tuple.
def color_to_rgb(color: str | Sequence[int]) -> tuple[int, int, int]:
    # Hex strings are parsed.
    if isinstance(color, str):
        # Parse.
        return hex_to_rgb(color)
    # Tuples keep their first three channels.
    return (int(color[0]), int(color[1]), int(color[2]))


# Lookup table of a colour map.
def colormap_lut(
    name: str | Sequence[str | Sequence[int]],  # Name in COLORMAPS or control colours.
    n: int = 256,  # Number of entries.
    reverse: bool = False,  # Reverse the order.
) -> NDArray[np.uint8]:  # (n, 3) RGB table.
    # Control points of a named map.
    if isinstance(name, str):
        # The name must be known.
        if name.lower() not in COLORMAPS:
            # Explain the accepted names.
            raise ValueError(f"unknown colour map {name!r}; known: {sorted(COLORMAPS)}")
        # Control colours of the map.
        points = COLORMAPS[name.lower()]
    # Explicit control colours.
    else:
        # Use them as given.
        points = tuple(name)
    # At least two colours and two entries are needed.
    if len(points) < 2 or n < 2:
        # Explain the requirement.
        raise ValueError("a colour map needs at least two colours and two entries")
    # Control colours as an array.
    rgb = np.array([color_to_rgb(c) for c in points], dtype=np.float64)
    # Reverse on request.
    if reverse:
        # Flip the control points.
        rgb = rgb[::-1]
    # Positions of the control points.
    xp = np.linspace(0.0, 1.0, len(rgb))
    # Positions of the table entries.
    x = np.linspace(0.0, 1.0, n)
    # Interpolate every channel.
    table = np.stack([np.interp(x, xp, rgb[:, ch]) for ch in range(3)], axis=1)
    # Round to bytes.
    return np.round(table).astype(np.uint8)


# Continuous values to RGBA through a colour map.
def apply_colormap(
    values: NDArray[Any],  # (H, W) values.
    cmap: str | Sequence[str | Sequence[int]] = "viridis",  # Colour map.
    vmin: float | None = None,  # Value of the first colour; None uses the minimum.
    vmax: float | None = None,  # Value of the last colour; None uses the maximum.
    nodata: float | None = None,  # Value drawn transparent.
    reverse: bool = False,  # Reverse the colour map.
    n: int = 256,  # Size of the lookup table.
) -> NDArray[np.uint8]:  # (H, W, 4) RGBA image.
    # Values as float.
    v = np.array(values, dtype=np.float64, copy=True)
    # Only single bands are supported.
    if v.ndim != 2:
        # Explain the requirement.
        raise ValueError(f"expected an (H, W) array, got shape {v.shape}")
    # Nodata becomes NaN.
    if nodata is not None:
        # Mark missing pixels.
        v[v == nodata] = np.nan
    # Valid pixels.
    valid = np.isfinite(v)
    # Default bounds from the data.
    lo = float(np.min(v[valid])) if vmin is None and valid.any() else vmin
    # Upper bound.
    hi = float(np.max(v[valid])) if vmax is None and valid.any() else vmax
    # Bounds of an empty image.
    lo, hi = (0.0 if lo is None else lo), (1.0 if hi is None else hi)
    # Lookup table.
    lut = colormap_lut(cmap, n, reverse)
    # Position in [0, 1]; constant images map to the first colour.
    t = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    # Table index of every pixel.
    idx = np.round(np.clip(np.nan_to_num(t), 0.0, 1.0) * (n - 1)).astype(np.intp)
    # RGBA output.
    out = np.zeros((*v.shape, 4), dtype=np.uint8)
    # Colours.
    out[..., :3] = lut[idx]
    # Opaque where valid.
    out[..., 3] = np.where(valid, 255, 0)
    # Return the image.
    return out


# Values to RGBA by class breaks.
def classify_colors(
    values: NDArray[Any],  # (H, W) values.
    breaks: Sequence[float],  # Increasing upper bounds of all classes but the last.
    colors: Sequence[str | Sequence[int]],  # One colour per class, len(breaks) + 1.
) -> NDArray[np.uint8]:  # (H, W, 4) RGBA image.
    # Values as float.
    v = np.asarray(values, dtype=np.float64)
    # Breaks as an array.
    b = np.asarray(breaks, dtype=np.float64)
    # One more colour than breaks.
    if len(colors) != b.size + 1:
        # Explain the requirement.
        raise ValueError("expected len(breaks) + 1 colours")
    # Breaks must increase.
    if np.any(np.diff(b) <= 0):
        # Explain the requirement.
        raise ValueError("breaks must be strictly increasing")
    # Class of every pixel: values equal to a break belong to the lower class.
    idx = np.searchsorted(b, np.nan_to_num(v), side="left")
    # Colour table.
    table = np.array([color_to_rgb(c) for c in colors], dtype=np.uint8)
    # RGBA output.
    out = np.zeros((*v.shape, 4), dtype=np.uint8)
    # Colours.
    out[..., :3] = table[idx]
    # Opaque where valid.
    out[..., 3] = np.where(np.isfinite(v), 255, 0)
    # Return the image.
    return out


# Palette of RGB tuples from a mapping of hex strings, tuples or (name, colour).
def _palette(palette: Mapping[Any, Any]) -> dict[int, tuple[int, int, int]]:
    # Result.
    out = {}
    # Visit every entry.
    for key, entry in palette.items():
        # (name, colour) pairs such as WORLDCOVER_PALETTE.
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
            # Keep the colour.
            entry = entry[1]
        # Store the RGB tuple.
        out[int(key)] = color_to_rgb(entry)
    # Return the palette.
    return out


# Class map to RGB.
def colorize_mask(
    mask: NDArray[Any],  # (H, W) class ids.
    colormap: Mapping[int, Any] | None = None,  # Class id to colour.
) -> NDArray[np.uint8]:  # (H, W, 3) RGB; unknown classes are black.
    # Palette of RGB tuples.
    palette = _palette(DEFAULT_PALETTE if colormap is None else colormap)
    # Class ids as array.
    m = np.asarray(mask)
    # RGB output.
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    # Paint every class.
    for class_id, color in palette.items():
        # Pixels of the class.
        rgb[m == class_id] = color
    # Return the image.
    return rgb


# Class map to RGBA with transparent nodata and unknown classes.
def colorize_classes(
    labels: NDArray[Any],  # (H, W) class ids.
    palette: Mapping[int, Any] = WORLDCOVER_PALETTE,  # Class id to colour.
    nodata: int | None = None,  # Class drawn transparent.
) -> NDArray[np.uint8]:  # (H, W, 4) RGBA image.
    # Class ids as array.
    m = np.asarray(labels)
    # RGBA output, transparent by default.
    out = np.zeros((*m.shape, 4), dtype=np.uint8)
    # Paint every class.
    for class_id, color in _palette(palette).items():
        # The nodata class stays transparent.
        if class_id == nodata:
            # Next class.
            continue
        # Pixels of the class.
        hit = m == class_id
        # Colour.
        out[hit, :3] = color
        # Opaque.
        out[hit, 3] = 255
    # Return the image.
    return out


# =============================================================================
# End of module src/unbihexium/visualization/colormaps.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
