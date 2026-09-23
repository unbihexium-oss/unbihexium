# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/visualization/__init__.py
# Title       : Display images, colour maps, relief shading and legends
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and Pillow; matplotlib
#               is optional
# =============================================================================
#
# Abstract
# --------
# Rendering of imagery and map products without a plotting library:
#
#   colormaps    colour maps for indices, class palettes (ESA WorldCover),
#                class breaks
#   composites   RGB composites with stretches and gamma, overlays
#   relief       Horn hillshade, multidirectional hillshade, shaded images
#   output       PNG quicklooks with world files, legends (Pillow; a
#                matplotlib figure only when matplotlib is installed)
#
# Usage
# -----
#   from unbihexium.visualization import rgb_composite, save_png
#   rgb = rgb_composite(stack, "true_color", band_names=names, gamma=1.2)
#   save_png("quicklook.png", rgb, transform=raster_transform)
# =============================================================================

# Colour maps and palettes.
from unbihexium.visualization.colormaps import (
    COLORMAPS,  # Named colour maps.
    DEFAULT_PALETTE,  # Default class colours.
    INDEX_COLORMAPS,  # Colour maps of common indices.
    WORLDCOVER_PALETTE,  # ESA WorldCover legend.
    apply_colormap,  # Values to RGBA.
    classify_colors,  # Values to RGBA by breaks.
    color_to_rgb,  # Colour to an RGB tuple.
    colorize_classes,  # Class map to RGBA.
    colorize_mask,  # Class map to RGB.
    colormap_lut,  # Lookup table.
    hex_to_rgb,  # Hex string to RGB.
)  # End of the colour map imports.

# Composites and overlays.
from unbihexium.visualization.composites import (
    LANDSAT89_COMPOSITES,  # Landsat band combinations.
    SENTINEL2_COMPOSITES,  # Sentinel-2 band combinations.
    alpha_composite,  # Porter-Duff over.
    normalize_for_display,  # Percentile stretch to bytes.
    overlay_mask,  # Coloured mask overlay.
    rgb_composite,  # Three bands to RGB.
    to_uint8,  # Unit interval to bytes.
)  # End of the composite imports.

# Output.
from unbihexium.visualization.output import (
    create_legend,  # Legend array.
    legend_figure,  # Matplotlib legend.
    legend_image,  # Pillow legend.
    quicklook,  # Reduced PNG.
    save_png,  # PNG writer.
    world_file_lines,  # World file content.
)  # End of the output imports.

# Relief shading.
from unbihexium.visualization.relief import (
    hillshade,  # Illumination.
    multidirectional_hillshade,  # Several light directions.
    shade_image,  # Shaded RGB image.
    slope_aspect,  # Slope and aspect.
)  # End of the relief imports.

# Public names of the package.
__all__ = [
    "COLORMAPS",  # Named colour maps.
    "DEFAULT_PALETTE",  # Default class colours.
    "INDEX_COLORMAPS",  # Colour maps of common indices.
    "LANDSAT89_COMPOSITES",  # Landsat band combinations.
    "SENTINEL2_COMPOSITES",  # Sentinel-2 band combinations.
    "WORLDCOVER_PALETTE",  # ESA WorldCover legend.
    "alpha_composite",  # Porter-Duff over.
    "apply_colormap",  # Values to RGBA.
    "classify_colors",  # Values to RGBA by breaks.
    "color_to_rgb",  # Colour to an RGB tuple.
    "colorize_classes",  # Class map to RGBA.
    "colorize_mask",  # Class map to RGB.
    "colormap_lut",  # Lookup table.
    "create_legend",  # Legend array.
    "hex_to_rgb",  # Hex string to RGB.
    "hillshade",  # Illumination.
    "legend_figure",  # Matplotlib legend.
    "legend_image",  # Pillow legend.
    "multidirectional_hillshade",  # Several light directions.
    "normalize_for_display",  # Percentile stretch to bytes.
    "overlay_mask",  # Coloured mask overlay.
    "quicklook",  # Reduced PNG.
    "rgb_composite",  # Three bands to RGB.
    "save_png",  # PNG writer.
    "shade_image",  # Shaded RGB image.
    "slope_aspect",  # Slope and aspect.
    "to_uint8",  # Unit interval to bytes.
    "world_file_lines",  # World file content.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/visualization/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
