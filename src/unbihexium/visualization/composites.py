# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/visualization/composites.py
# Title       : Colour composites, display stretches and overlays
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Display images from multispectral bands:
#
#   SENTINEL2_COMPOSITES   band combinations of common Sentinel-2 composites
#   LANDSAT89_COMPOSITES   the same combinations for Landsat 8/9 OLI bands
#   rgb_composite          three bands to an 8-bit RGB(A) image with a
#                          percentile, min-max or fixed stretch and gamma
#   normalize_for_display  percentile stretch of any array to uint8
#   to_uint8               values in [0, 1] to bytes
#   overlay_mask           colour a mask over an RGB image
#   alpha_composite        "over" compositing of an RGBA layer on an image
#
# Band names follow the ESA (B02, B8A, ...) and USGS (B1 to B7) naming. The
# stretches are those of unbihexium.preprocessing.enhancement.
#
# References
# ----------
#   Porter, T., Duff, T. (1984). Compositing digital images. ACM SIGGRAPH
#     Computer Graphics 18(3), 253-259.
#   ESA. Sentinel-2 user handbook (spectral bands and band combinations).
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values and sequences.
from typing import Any, Sequence

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Stretches.
from unbihexium.preprocessing.enhancement import (
    gamma_correction,  # Power-law adjustment.
    linear_stretch,  # Fixed bounds.
    percentile_stretch,  # Percentile bounds.
)  # End of the stretch imports.

# Sentinel-2 MSI band combinations (red, green, blue channels).
SENTINEL2_COMPOSITES = {
    "true_color": ("B04", "B03", "B02"),  # Natural colour.
    "color_infrared": ("B08", "B04", "B03"),  # Vegetation in red.
    "swir": ("B12", "B8A", "B04"),  # Short-wave infrared.
    "agriculture": ("B11", "B08", "B02"),  # Crop vigour.
    "geology": ("B12", "B11", "B02"),  # Lithology.
    "bathymetric": ("B04", "B03", "B01"),  # Shallow water.
}  # End of the Sentinel-2 composites.

# Landsat 8/9 OLI band combinations (red, green, blue channels).
LANDSAT89_COMPOSITES = {
    "true_color": ("B4", "B3", "B2"),  # Natural colour.
    "color_infrared": ("B5", "B4", "B3"),  # Vegetation in red.
    "swir": ("B7", "B5", "B4"),  # Short-wave infrared.
    "agriculture": ("B6", "B5", "B2"),  # Crop vigour.
    "geology": ("B7", "B6", "B2"),  # Lithology.
}  # End of the Landsat composites.


# Values in [0, 1] to uint8, NaN as zero.
def to_uint8(image: NDArray[Any]) -> NDArray[np.uint8]:
    # Clip to the unit interval and replace NaN.
    x = np.clip(np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0), 0.0, 1.0)
    # Scale and round.
    return np.round(x * 255.0).astype(np.uint8)


# Resolve band selectors to indices.
def _band_indices(
    bands: Sequence[int | str],  # Indices or names.
    names: Sequence[str] | None,  # Names of the bands of the stack.
) -> list[int]:  # Indices.
    # Indices of the selection.
    out = []
    # Visit every selector.
    for band in bands:
        # Names are looked up.
        if isinstance(band, str):
            # Names need a band list.
            if names is None or band not in names:
                # Explain the problem.
                raise ValueError(f"band {band!r} not found in band names {names}")
            # Position of the name.
            out.append(list(names).index(band))
        # Indices are used directly.
        else:
            # Keep the index.
            out.append(int(band))
    # Return the indices.
    return out


# Three bands of a (C, H, W) stack to an 8-bit colour image.
def rgb_composite(
    stack: NDArray[Any],  # (C, H, W) bands.
    bands: Sequence[int | str] | str = (0, 1, 2),  # Indices, names or a composite name.
    band_names: Sequence[str] | None = None,  # Names of the bands of the stack.
    stretch: str = "percentile",  # "percentile", "minmax" or "fixed".
    low: float | Sequence[float] = 2.0,  # Lower percentile or fixed value(s).
    high: float | Sequence[float] = 98.0,  # Upper percentile or fixed value(s).
    gamma: float = 1.0,  # Gamma applied after the stretch.
    per_band: bool = True,  # Stretch bands separately or with common bounds.
    nodata: float | None = None,  # Value drawn transparent.
    alpha: bool = False,  # Add an alpha channel.
) -> NDArray[np.uint8]:  # (H, W, 3) or (H, W, 4) image.
    # Composite names are looked up in the Sentinel-2 and Landsat tables.
    if isinstance(bands, str):
        # Band triples of the name in either table.
        tables = (SENTINEL2_COMPOSITES, LANDSAT89_COMPOSITES)
        # Candidates in table order.
        candidates = [t[bands] for t in tables if bands in t]
        # The name must be known.
        if not candidates:
            # Explain the accepted names.
            raise ValueError(f"unknown composite {bands!r}; known: {sorted(SENTINEL2_COMPOSITES)}")
        # Triples whose names all occur in the stack.
        found = [c for c in candidates if band_names and all(b in band_names for b in c)]
        # Prefer a matching triple, else the Sentinel-2 one.
        bands = (found or candidates)[0]
    # Stack as float.
    x = np.asarray(stack, dtype=np.float64)
    # A band axis is required.
    if x.ndim != 3:
        # Explain the requirement.
        raise ValueError(f"expected a (C, H, W) stack, got shape {x.shape}")
    # Three bands are required.
    if len(bands) != 3:
        # Explain the requirement.
        raise ValueError("exactly three bands are needed")
    # Selected bands.
    rgb = x[_band_indices(bands, band_names)].copy()
    # Mark nodata.
    if nodata is not None:
        # Missing values become NaN.
        rgb[rgb == nodata] = np.nan
    # Pixels with every channel valid.
    valid = np.isfinite(rgb).all(axis=0)
    # Percentile stretch.
    if stretch == "percentile":
        # Separate bounds per band.
        if per_band:
            # Stretch every band.
            s = percentile_stretch(rgb, float(np.ravel(low)[0]), float(np.ravel(high)[0]))
        # Common bounds keep the colour balance.
        else:
            # Bounds over all three bands.
            lo, hi = np.nanpercentile(rgb, [float(np.ravel(low)[0]), float(np.ravel(high)[0])])
            # Stretch with the common bounds.
            s = linear_stretch(rgb, lo, hi if hi > lo else lo + 1.0)
    # Min-max stretch.
    elif stretch == "minmax":
        # Band minima, or the overall minimum.
        lo = np.nanmin(rgb, axis=(1, 2)) if per_band else np.nanmin(rgb)
        # Band maxima, or the overall maximum.
        hi = np.nanmax(rgb, axis=(1, 2)) if per_band else np.nanmax(rgb)
        # Constant bands get a unit range.
        s = linear_stretch(rgb, lo, np.where(hi > lo, hi, lo + 1.0))
    # Fixed bounds, for example reflectance 0 to 0.3.
    elif stretch == "fixed":
        # Stretch with the given bounds.
        s = linear_stretch(rgb, low, high)
    # Unknown stretches are rejected.
    else:
        # Explain the accepted names.
        raise ValueError("stretch must be 'percentile', 'minmax' or 'fixed'")
    # Gamma adjustment.
    s = gamma_correction(np.nan_to_num(s), gamma)
    # Channel-last bytes.
    out = to_uint8(np.moveaxis(s, 0, -1))
    # Black where invalid.
    out[~valid] = 0
    # Without alpha the RGB image is complete.
    if not alpha:
        # Return RGB.
        return out
    # Alpha channel, opaque where valid.
    a = np.where(valid, 255, 0).astype(np.uint8)
    # Return RGBA.
    return np.concatenate([out, a[..., None]], axis=-1)


# Percentile stretch of an image to uint8 for display.
def normalize_for_display(
    image: NDArray[Any],  # (H, W) or (C, H, W) values.
    percentile: tuple[float, float] = (2, 98),  # Lower and upper percentile.
) -> NDArray[np.uint8]:  # Bytes of the same shape; NaN becomes 0.
    # Values as float; any shape is accepted.
    x = np.asarray(image, dtype=np.float64)
    # Common bounds over all values keep the relation between bands.
    lo, hi = np.nanpercentile(x, percentile)
    # Constant images get a unit range.
    span = hi - lo if hi > lo else 1.0
    # Stretch, clip and convert to bytes.
    return to_uint8((x - lo) / span)


# Colour a binary mask over an RGB image.
def overlay_mask(
    image: NDArray[Any],  # (H, W, 3) uint8 image.
    mask: NDArray[Any],  # (H, W) mask, nonzero is drawn.
    alpha: float = 0.5,  # Opacity of the colour.
    color: tuple[int, int, int] = (255, 0, 0),  # Colour of the mask.
) -> NDArray[np.uint8]:  # Blended image.
    # Opacity must lie in [0, 1].
    if not 0.0 <= alpha <= 1.0:
        # Explain the requirement.
        raise ValueError("alpha must lie between 0 and 1")
    # Image as float.
    base = np.asarray(image, dtype=np.float64)
    # Pixels to colour.
    m = np.asarray(mask) > 0
    # Colour as float.
    c = np.asarray(color, dtype=np.float64)
    # Blend only the masked pixels.
    base[m] = (1.0 - alpha) * base[m] + alpha * c
    # Round to bytes.
    return np.round(base).clip(0, 255).astype(np.uint8)


# Composite an RGBA layer over an RGB or RGBA image ("over" operator).
def alpha_composite(
    background: NDArray[Any],  # (H, W, 3) or (H, W, 4) uint8 image.
    layer: NDArray[Any],  # (H, W, 4) uint8 layer.
    opacity: float = 1.0,  # Extra opacity of the layer.
) -> NDArray[np.uint8]:  # (H, W, 3) image.
    # Background colour in [0, 1].
    bg = np.asarray(background, dtype=np.float64)[..., :3] / 255.0
    # Layer in [0, 1].
    fg = np.asarray(layer, dtype=np.float64) / 255.0
    # The layer needs an alpha channel.
    if fg.shape[-1] != 4 or fg.shape[:2] != bg.shape[:2]:
        # Explain the requirement.
        raise ValueError("layer must be (H, W, 4) with the size of the background")
    # Effective alpha.
    a = fg[..., 3:4] * float(np.clip(opacity, 0.0, 1.0))
    # Porter-Duff "over" on an opaque background.
    out = fg[..., :3] * a + bg * (1.0 - a)
    # Bytes.
    return to_uint8(out)


# =============================================================================
# End of module src/unbihexium/visualization/composites.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
