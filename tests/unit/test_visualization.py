# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_visualization.py
# Title       : Tests of colour maps, composites, hillshading and PNG output
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and Pillow
# =============================================================================
#
# Abstract
# --------
# Checks the visualization package: colour map end points and
# interpolation, class palettes, composites with named band combinations,
# the Horn hillshade of flat and planar surfaces against the analytic
# illumination, PNG files with world files read back with Pillow, legends,
# and the lazy matplotlib import.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Module table, to hide matplotlib.
import sys

# Arrays.
import numpy as np

# Test framework.
import pytest

# Image reading for round trips.
from PIL import Image

# Functions under test.
from unbihexium.visualization import (
    WORLDCOVER_PALETTE,  # WorldCover legend.
    alpha_composite,  # Porter-Duff over.
    apply_colormap,  # Values to RGBA.
    classify_colors,  # Class breaks.
    colorize_classes,  # Class map to RGBA.
    colorize_mask,  # Class map to RGB.
    colormap_lut,  # Lookup table.
    create_legend,  # Legend array.
    hex_to_rgb,  # Hex parsing.
    hillshade,  # Illumination.
    legend_figure,  # Matplotlib legend.
    legend_image,  # Pillow legend.
    multidirectional_hillshade,  # Several directions.
    normalize_for_display,  # Display stretch.
    overlay_mask,  # Mask overlay.
    quicklook,  # Reduced PNG.
    rgb_composite,  # Composites.
    save_png,  # PNG writer.
    shade_image,  # Shaded image.
    slope_aspect,  # Slope and aspect.
    world_file_lines,  # World file.
)  # End of the imports under test.


# Colour map tables and value mapping.
def test_colormaps() -> None:
    # Two-colour map from black to white.
    lut = colormap_lut("greys", 3)
    # End points and midpoint.
    np.testing.assert_array_equal(lut, [[0, 0, 0], [128, 128, 128], [255, 255, 255]])
    # Viridis starts and ends at its published colours.
    vir = colormap_lut("viridis")
    # First and last entries.
    assert tuple(vir[0]) == hex_to_rgb("#440154") and tuple(vir[-1]) == hex_to_rgb("#fde725")
    # Values with NaN and a nodata value.
    values = np.array([[0.0, 1.0], [np.nan, -9.0]])
    # Map with fixed bounds.
    rgba = apply_colormap(values, "greys", vmin=0.0, vmax=1.0, nodata=-9.0)
    # Black, white and two transparent pixels.
    assert tuple(rgba[0, 0]) == (0, 0, 0, 255) and tuple(rgba[0, 1]) == (255, 255, 255, 255)
    # Transparency of missing values.
    assert rgba[1, 0, 3] == 0 and rgba[1, 1, 3] == 0
    # Reversed map starts white.
    assert tuple(colormap_lut("greys", 2, reverse=True)[0]) == (255, 255, 255)
    # Unknown names are rejected.
    with pytest.raises(ValueError, match="unknown colour map"):
        # Misspelled name.
        colormap_lut("virdis")


# Class breaks and class palettes.
def test_class_colours() -> None:
    # NDVI values around breaks 0 and 0.5; values equal to a break go to the lower class.
    ndvi = np.array([[-0.2, 0.0, 0.3, 0.8]])
    # Three classes.
    rgba = classify_colors(ndvi, [0.0, 0.5], ["#0000ff", "#ffff00", "#00ff00"])
    # Class colours of the four pixels.
    expected = [[0, 0, 255], [0, 0, 255], [255, 255, 0], [0, 255, 0]]
    # Compare the RGB channels.
    np.testing.assert_array_equal(rgba[0, :, :3], expected)
    # WorldCover tree cover and water.
    labels = np.array([[10, 80], [0, 50]])
    # Palette colours; unknown class 0 is transparent.
    wc = colorize_classes(labels, WORLDCOVER_PALETTE)
    # Tree cover is #006400.
    assert tuple(wc[0, 0]) == (0, 100, 0, 255)
    # Class 0 is not in the legend.
    assert wc[1, 0, 3] == 0
    # Legacy default palette: class 1 is red.
    assert tuple(colorize_mask(np.array([[1]]))[0, 0]) == (255, 0, 0)


# Composites with named band combinations.
def test_rgb_composite() -> None:
    # Four bands with distinct constant values.
    names = ["B02", "B03", "B04", "B08"]
    # Stack of reflectances.
    stack = np.stack([np.full((2, 2), v) for v in (0.05, 0.10, 0.15, 0.30)])
    # Fixed stretch from 0 to 0.3.
    rgb = rgb_composite(stack, "color_infrared", band_names=names, stretch="fixed", low=0, high=0.3)
    # Red is B08 (0.3), green B04 (0.15), blue B03 (0.10).
    assert tuple(rgb[0, 0]) == (255, 128, 85)
    # Nodata becomes transparent.
    stack[:, 0, 0] = -1
    # Composite with alpha.
    rgba = rgb_composite(stack, (2, 1, 0), stretch="fixed", low=0, high=0.3, nodata=-1, alpha=True)
    # Alpha channel.
    assert rgba[0, 0, 3] == 0 and rgba[1, 1, 3] == 255
    # Unknown bands are reported.
    with pytest.raises(ValueError, match="B12"):
        # SWIR composite needs B12.
        rgb_composite(stack, "swir", band_names=names)


# Display stretch and overlays.
def test_display_helpers() -> None:
    # Values 0 to 100.
    x = np.arange(101, dtype=float)
    # Full-range stretch maps 50 to 128.
    assert normalize_for_display(x, (0, 100))[50] == 128
    # Grey image.
    grey = np.full((1, 2, 3), 100, dtype=np.uint8)
    # Half-transparent red over the first pixel.
    out = overlay_mask(grey, np.array([[1, 0]]), alpha=0.5)
    # 0.5 * 100 + 0.5 * 255 in red, 50 in green and blue.
    assert tuple(out[0, 0]) == (178, 50, 50) and tuple(out[0, 1]) == (100, 100, 100)
    # Opaque white layer over black.
    layer = np.full((1, 1, 4), 255, dtype=np.uint8)
    # Half opacity gives mid grey.
    blended = alpha_composite(np.zeros((1, 1, 3), np.uint8), layer, 0.5)
    # Round(127.5) in every channel.
    assert tuple(blended[0, 0]) == (128, 128, 128)


# Hillshade of flat and planar surfaces.
def test_hillshade_analytic() -> None:
    # Flat terrain is lit by cos(zenith) = sin(altitude).
    flat = hillshade(np.zeros((5, 5)), altitude=45.0)
    # sin(45 degrees).
    np.testing.assert_allclose(flat, np.sin(np.radians(45.0)))
    # A 45 degree slope rising towards the east faces west.
    ramp = np.tile(np.arange(5, dtype=float), (5, 1))
    # Slope and aspect of the plane.
    slope, aspect = slope_aspect(ramp, cellsize=1.0)
    # 45 degrees and aspect pi (west in the mathematical convention).
    assert slope[2, 2] == pytest.approx(np.pi / 4) and aspect[2, 2] == pytest.approx(np.pi)
    # Sun in the west at 45 degrees shines perpendicular on the plane.
    assert hillshade(ramp, azimuth=270.0, altitude=45.0)[2, 2] == pytest.approx(1.0)
    # Sun in the east at 45 degrees grazes it.
    assert hillshade(ramp, azimuth=90.0, altitude=45.0)[2, 2] == pytest.approx(0.0, abs=1e-12)
    # Byte output of flat terrain: round(255 * 0.7071).
    assert hillshade(np.zeros((3, 3)), as_uint8=True)[1, 1] == 180
    # Several directions on flat terrain give the same illumination.
    np.testing.assert_allclose(multidirectional_hillshade(np.zeros((3, 3))), np.sin(np.radians(45)))
    # Full shading of a white image by a half-lit surface.
    shaded = shade_image(np.full((1, 1, 3), 200, np.uint8), np.array([[0.5]]), strength=1.0)
    # 200 * 0.5.
    assert tuple(shaded[0, 0]) == (100, 100, 100)


# PNG files with world files.
def test_save_png_and_world_file(tmp_path) -> None:
    # 10 m pixels with the upper-left corner at (500000, 7000000).
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 7000000.0)
    # Centre of the first pixel is half a pixel inside.
    assert world_file_lines(transform) == ["10.0", "0.0", "0.0", "-10.0", "500005.0", "6999995.0"]
    # RGB image.
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    # One red pixel.
    image[0, 0] = (255, 0, 0)
    # Write the PNG and the world file.
    path = save_png(tmp_path / "out" / "image.png", image, transform)
    # Read the PNG back.
    back = np.asarray(Image.open(path))
    # Identical pixels.
    np.testing.assert_array_equal(back, image)
    # World file next to the image.
    assert (tmp_path / "out" / "image.pgw").read_text().split() == world_file_lines(transform)
    # Float images are rejected.
    with pytest.raises(ValueError, match="uint8"):
        # Wrong type.
        save_png(tmp_path / "bad.png", image.astype(float))


# Quicklooks are decimated to the size limit.
def test_quicklook(tmp_path) -> None:
    # Stack of three bands, 100 x 60 pixels.
    stack = np.random.default_rng(0).uniform(size=(3, 100, 60))
    # Longest side limited to 25 pixels: step 4.
    path = quicklook(tmp_path / "q.png", stack, max_size=25, transform=(1, 0, 0, 0, -1, 0))
    # Size of the quicklook.
    assert Image.open(path).size == (15, 25)
    # The world file carries the decimated pixel size.
    assert (tmp_path / "q.pgw").read_text().split()[0] == "4.0"
    # Single bands are colour-mapped.
    single = quicklook(tmp_path / "s.png", stack[0], max_size=50)
    # RGBA output of half size.
    assert Image.open(single).mode == "RGBA" and Image.open(single).size == (30, 50)


# Legends drawn with Pillow and matplotlib.
def test_legends(monkeypatch) -> None:
    # Pillow legend with two entries.
    img = legend_image(["Water", "Forest"], ["#0064c8", (0, 100, 0)], patch=(20, 10), padding=5)
    # Height: padding plus two rows of patch height plus padding.
    assert img.shape[0] == 5 + 2 * 15
    # First patch colour.
    assert tuple(img[10, 10]) == (0, 100, 200)
    # Legacy geometry of create_legend.
    legacy = create_legend(["a", "b"], [(255, 0, 0), (0, 0, 255)], size=(50, 10))
    # Shape and patch colours.
    assert legacy.shape == (20, 150, 3) and tuple(legacy[15, 5]) == (0, 0, 255)
    # Hide matplotlib to check the error message.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    # Also hide its submodules.
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    # The missing dependency is explained.
    with pytest.raises(ImportError, match="requires matplotlib"):
        # Matplotlib legend.
        legend_figure(["a"], ["#000000"])


# =============================================================================
# End of module tests/unit/test_visualization.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
