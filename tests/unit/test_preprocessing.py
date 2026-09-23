# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_preprocessing.py
# Title       : Tests of radiometric and geometric preprocessing
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest
# =============================================================================
#
# Abstract
# --------
# Checks the preprocessing package against hand-computed values: Sentinel-2
# and Landsat calibration, the Earth-Sun distance at perihelion, DOS1, SCL
# and QA_PIXEL masks, stretches and histogram matching, the defining
# identities of the pansharpening methods, resampling and aggregation with
# missing values, and the array transforms.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Functions under test.
from unbihexium.preprocessing import (
    Compose,  # Chain of transforms.
    Normalize,  # Standardisation.
    Pad,  # Padding.
    Resize,  # Resizing.
    aggregate,  # Block aggregation.
    apply_mask,  # Masking.
    brovey,  # Ratio transform.
    buffer_mask,  # Mask buffer.
    dark_object_subtraction,  # DOS1.
    earth_sun_distance,  # Earth-Sun distance.
    from_tensor,  # Channel-last layout.
    gamma_correction,  # Gamma.
    gram_schmidt,  # Gram-Schmidt adaptive.
    histogram_equalize,  # Equalisation.
    histogram_match,  # Matching.
    ihs,  # Fast IHS.
    landsat_brightness_temperature,  # Thermal conversion.
    landsat_c2l2_reflectance,  # Level-2 reflectance.
    landsat_c2l2_temperature,  # Level-2 temperature.
    landsat_cloud_confidence,  # QA confidence.
    landsat_qa_mask,  # QA flags.
    landsat_radiance,  # Level-1 radiance.
    landsat_toa_reflectance_from_mtl,  # TOA reflectance from MTL.
    linear_stretch,  # Fixed stretch.
    minmax_normalize,  # Min-max scaling.
    parse_landsat_mtl,  # MTL parser.
    percentile_stretch,  # Percentile stretch.
    qa_bits,  # Bit fields.
    radiance_to_reflectance,  # Radiance to reflectance.
    resample,  # Interpolation.
    scaled_transform,  # Transform of a resampled grid.
    scl_valid_mask,  # SCL mask.
    sentinel2_reflectance,  # Sentinel-2 calibration.
    standardize,  # Z-scores.
    to_tensor,  # Band-first layout.
    upsample_to_pan,  # Upsampling.
)  # End of the imports under test.

# Excerpt of a Landsat 9 Collection 2 Level-1 MTL file.
MTL_TEXT = """GROUP = LANDSAT_METADATA_FILE
  GROUP = PRODUCT_CONTENTS
    LANDSAT_PRODUCT_ID = "LC09_L1TP_188018_20230601_20230601_02_T1"
    DATE_ACQUIRED = 2023-06-01
  END_GROUP = PRODUCT_CONTENTS
  GROUP = IMAGE_ATTRIBUTES
    SUN_ELEVATION = 30.00000000
    WRS_PATH = 188
  END_GROUP = IMAGE_ATTRIBUTES
  GROUP = LEVEL1_RADIOMETRIC_RESCALING
    REFLECTANCE_MULT_BAND_4 = 2.0000E-05
    REFLECTANCE_ADD_BAND_4 = -0.100000
  END_GROUP = LEVEL1_RADIOMETRIC_RESCALING
END_GROUP = LANDSAT_METADATA_FILE
END
"""  # End of the MTL excerpt.


# Sentinel-2 offset depends on the processing baseline; DN 0 is no data.
def test_sentinel2_reflectance_offsets() -> None:
    # Digital numbers with a missing pixel.
    dn = np.array([[0, 2000], [11000, 1000]], dtype=np.uint16)
    # Baseline 05.10 has the -1000 offset: (2000 - 1000) / 10000.
    rho = sentinel2_reflectance(dn, processing_baseline="05.10")
    # Missing pixel.
    assert np.isnan(rho[0, 0])
    # Hand-computed reflectances.
    np.testing.assert_allclose(rho[[0, 1, 1], [1, 0, 1]], [0.1, 1.0, 0.0])
    # Baseline 03.01 has no offset.
    old = sentinel2_reflectance(dn, processing_baseline="03.01")
    # 2000 / 10000.
    assert old[0, 1] == pytest.approx(0.2)
    # Explicit offsets override the baseline.
    assert sentinel2_reflectance(dn, offset=0.0)[0, 1] == pytest.approx(0.2)


# MTL parsing and Landsat TOA reflectance with the sun elevation correction.
def test_landsat_toa_reflectance_from_mtl() -> None:
    # Parse the excerpt.
    mtl = parse_landsat_mtl(MTL_TEXT)
    # Quoted values are strings.
    assert mtl["LANDSAT_PRODUCT_ID"].startswith("LC09")
    # Integers keep their type.
    assert mtl["WRS_PATH"] == 188
    # Exponent notation is parsed.
    assert mtl["REFLECTANCE_MULT_BAND_4"] == pytest.approx(2e-5)
    # Bare dates stay strings.
    assert mtl["DATE_ACQUIRED"] == "2023-06-01"
    # DN 10000 gives (2e-5 * 10000 - 0.1) / sin(30 deg) = 0.2.
    rho = landsat_toa_reflectance_from_mtl(np.array([10000, 0]), mtl, band=4)
    # Hand-computed value.
    assert rho[0] == pytest.approx(0.2)
    # Fill pixel.
    assert np.isnan(rho[1])
    # Missing keys are reported.
    with pytest.raises(KeyError, match="REFLECTANCE_MULT_BAND_5"):
        # Band 5 is not in the excerpt.
        landsat_toa_reflectance_from_mtl(np.array([1]), mtl, band=5)


# Thermal radiance and brightness temperature invert each other.
def test_landsat_thermal_conversion() -> None:
    # Landsat 9 TIRS band 10 constants.
    k1, k2 = 774.8853, 1321.0789
    # Radiance of a 300 K blackbody from the inverse Planck relation.
    radiance = k1 / (np.exp(k2 / 300.0) - 1.0)
    # Brightness temperature of that radiance.
    assert landsat_brightness_temperature(np.array([radiance]), k1, k2)[0] == pytest.approx(300.0)
    # Radiance rescaling L = M Q + A.
    assert landsat_radiance(np.array([100]), 3.342e-4, 0.1)[0] == pytest.approx(0.1334200)


# Collection 2 Level-2 scale factors.
def test_landsat_level2_scale_factors() -> None:
    # 2.75e-5 * 10000 - 0.2.
    assert landsat_c2l2_reflectance(np.array([10000]))[0] == pytest.approx(0.075)
    # 0.00341802 * 44000 + 149.
    assert landsat_c2l2_temperature(np.array([44000]))[0] == pytest.approx(299.39288)
    # Fill value 0 is missing.
    assert np.isnan(landsat_c2l2_reflectance(np.array([0]))[0])


# Earth-Sun distance near perihelion and aphelion.
def test_earth_sun_distance() -> None:
    # Day 1: the day angle is zero, so E0 = 1.000110 + 0.034221 + 0.000719.
    expected = 1.0 / np.sqrt(1.000110 + 0.034221 + 0.000719)
    # Hand-computed value.
    assert earth_sun_distance(1) == pytest.approx(expected)
    # Early July the Earth is about 1.0167 AU from the Sun.
    assert earth_sun_distance(185) == pytest.approx(1.0167, abs=5e-4)
    # Days outside a year are rejected.
    with pytest.raises(ValueError, match="day_of_year"):
        # Day 0 does not exist.
        earth_sun_distance(0)


# Radiance to TOA reflectance with ESUN.
def test_radiance_to_reflectance() -> None:
    # Sun zenith 60 degrees: cos = 0.5.
    rho = radiance_to_reflectance(np.array([100.0]), esun=1500.0, sun_zenith=60.0, day_of_year=1)
    # pi L d^2 / (ESUN cos).
    expected = np.pi * 100.0 * earth_sun_distance(1) ** 2 / (1500.0 * 0.5)
    # Hand-computed value.
    assert rho[0] == pytest.approx(expected)


# DOS1 subtracts the dark object minus 1 % reflectance.
def test_dark_object_subtraction() -> None:
    # Band whose darkest pixel is 0.05.
    band = np.array([[0.05, 0.10], [0.20, 0.30]])
    # Percentile 0 selects the minimum.
    out, haze = dark_object_subtraction(band, dark_percentile=0.0)
    # Path reflectance 0.05 - 0.01.
    assert haze[0] == pytest.approx(0.04)
    # Every pixel is reduced by the path reflectance.
    np.testing.assert_allclose(out, band - 0.04)
    # Dark bands below 1 % are left unchanged.
    out2, haze2 = dark_object_subtraction(band * 0.1, dark_percentile=0.0)
    # No haze.
    assert haze2[0] == 0.0


# SCL classes 0, 1, 3, 8, 9 and 10 are invalid by default.
def test_scl_valid_mask() -> None:
    # Every SCL class once.
    scl = np.arange(12, dtype=np.uint8)
    # Valid classes.
    valid = np.flatnonzero(scl_valid_mask(scl))
    # Dark areas, vegetation, bare soil, water, unclassified and snow.
    np.testing.assert_array_equal(valid, [2, 4, 5, 6, 7, 11])
    # Floating point bands are rejected.
    with pytest.raises(ValueError, match="integer"):
        # Float SCL.
        scl_valid_mask(scl.astype(float))


# QA_PIXEL bit fields.
def test_landsat_qa_pixel() -> None:
    # Clear (bit 6), cloud (bit 3), shadow (bit 4), snow (bit 5), fill (bit 0).
    qa = np.array([1 << 6, 1 << 3, 1 << 4, 1 << 5, 1], dtype=np.uint16)
    # Default flags: fill, dilated cloud, cirrus, cloud and shadow.
    np.testing.assert_array_equal(landsat_qa_mask(qa), [False, True, True, False, True])
    # Snow is flagged on request.
    assert landsat_qa_mask(qa, snow=True)[3]
    # High cloud confidence in bits 8-9 and low cirrus confidence in bits 14-15.
    conf = np.array([(3 << 8) | (1 << 14)], dtype=np.uint16)
    # Cloud confidence.
    assert landsat_cloud_confidence(conf, "cloud")[0] == 3
    # Cirrus confidence.
    assert landsat_cloud_confidence(conf, "cirrus")[0] == 1
    # Generic bit extraction: bits 2-3 of 0b1100.
    assert qa_bits(np.array([12]), 2, 2)[0] == 3


# Buffers grow masks by a disk; masked pixels become NaN.
def test_buffer_and_apply_mask() -> None:
    # One flagged pixel.
    mask = np.zeros((5, 5), dtype=bool)
    # Centre pixel.
    mask[2, 2] = True
    # A disk of radius 1 is a cross of five pixels.
    assert buffer_mask(mask, 1).sum() == 5
    # A disk of radius 2 has 13 pixels.
    assert buffer_mask(mask, 2).sum() == 13
    # Masked image.
    out = apply_mask(np.ones((2, 5, 5)), mask)
    # Both bands are masked.
    assert np.isnan(out[:, 2, 2]).all() and np.isfinite(out).sum() == 48


# Linear and percentile stretches and gamma.
def test_stretches() -> None:
    # Values 0 to 100.
    x = np.arange(101, dtype=float).reshape(1, 101)
    # Stretch between 0 and 100 percentiles is x / 100.
    np.testing.assert_allclose(percentile_stretch(x, 0, 100), x / 100.0)
    # Fixed bounds with clipping.
    np.testing.assert_allclose(linear_stretch(np.array([[0.0, 5.0, 20.0]]), 0, 10), [[0, 0.5, 1]])
    # Per-band bounds of a (C, H, W) stack.
    stack = np.stack([np.full((1, 2), 5.0), np.full((1, 2), 15.0)])
    # Band 0 maps [0, 10], band 1 maps [10, 20].
    np.testing.assert_allclose(linear_stretch(stack, [0, 10], [10, 20]), 0.5)
    # Gamma 2 is the square root.
    assert gamma_correction(np.array([0.25]), 2.0)[0] == pytest.approx(0.5)
    # Empty ranges are rejected.
    with pytest.raises(ValueError, match="high"):
        # Equal bounds.
        linear_stretch(x, 1, 1)


# Histogram equalisation and matching.
def test_histogram_operations() -> None:
    # Four distinct values.
    x = np.array([[1.0, 2.0], [3.0, np.nan], [4.0, 4.0]])
    # Empirical CDF: 1/5, 2/5, 3/5 and 1 for the value 4.
    eq = histogram_equalize(x)
    # Hand-computed values.
    np.testing.assert_allclose(eq[[0, 0, 1, 2], [0, 1, 0, 0]], [0.2, 0.4, 0.6, 1.0])
    # NaN stays NaN.
    assert np.isnan(eq[1, 1])
    # Source and reference with the same ranks map exactly.
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Reference with the same ranks in another order and size.
    reference = np.array([[10.0, 40.0, 20.0, 30.0]])
    # Match the histograms.
    matched = histogram_match(source, reference)
    # Each rank takes the reference value of that rank.
    np.testing.assert_allclose(matched, [[10.0, 20.0], [30.0, 40.0]])


# Brovey and IHS preserve the intensity; GSA leaves consistent inputs unchanged.
def test_pansharpening_identities() -> None:
    # Random generator with a fixed seed.
    rng = np.random.default_rng(0)
    # Three positive bands.
    ms = rng.uniform(0.1, 0.5, size=(3, 8, 8))
    # Pan with extra detail.
    pan = ms.mean(axis=0) + rng.normal(0, 0.02, size=(8, 8))
    # Brovey without matching: the band mean equals the pan.
    np.testing.assert_allclose(brovey(pan, ms, match=False).mean(axis=0), pan)
    # IHS without matching: the band mean equals the pan too.
    np.testing.assert_allclose(ihs(pan, ms, match=False).mean(axis=0), pan)
    # A pan that is exactly a linear combination of the bands carries no new detail.
    exact = 0.2 * ms[0] + 0.5 * ms[1] + 0.3 * ms[2] + 0.01
    # GSA reproduces the bands.
    np.testing.assert_allclose(gram_schmidt(exact, ms), ms, atol=1e-10)
    # Mismatched grids are rejected.
    with pytest.raises(ValueError, match="ms must be"):
        # Wrong spatial size.
        ihs(pan, ms[:, :4, :4])
    # Upsampling keeps constant bands constant and reaches the pan grid.
    up = upsample_to_pan(np.full((2, 4, 4), 3.0), (8, 8))
    # Shape and values.
    assert up.shape == (2, 8, 8) and np.allclose(up, 3.0)


# Block aggregation with missing values.
def test_aggregate() -> None:
    # Values 0 to 15.
    x = np.arange(16, dtype=float).reshape(4, 4)
    # Means of the 2 x 2 blocks.
    np.testing.assert_allclose(aggregate(x, 2), [[2.5, 4.5], [10.5, 12.5]])
    # Sums of the blocks.
    np.testing.assert_allclose(aggregate(x, 2, "sum"), [[10, 18], [42, 50]])
    # Missing value: the block mean ignores it.
    x[0, 0] = np.nan
    # (1 + 4 + 5) / 3.
    assert aggregate(x, 2)[0, 0] == pytest.approx(10 / 3)
    # Majority of a class map; ties go to the smaller class.
    labels = np.array([[1, 1, 2, 3], [1, 2, 3, 2]])
    # Block modes 1 and 2.
    np.testing.assert_array_equal(aggregate(labels, 2, "mode"), [[1, 2]])
    # Nodata blocks are NaN.
    assert np.isnan(aggregate(np.zeros((2, 2)), 2, nodata=0)[0, 0])


# Interpolation keeps constants and handles missing values.
def test_resample() -> None:
    # Constant image with one missing pixel.
    x = np.full((4, 4), 5.0)
    # Missing value.
    x[1, 1] = -9999
    # Bilinear upsampling with normalised convolution.
    up = resample(x, (8, 8), "bilinear", nodata=-9999)
    # Valid output values stay exactly 5.
    np.testing.assert_allclose(up[np.isfinite(up)], 5.0)
    # Nearest neighbour keeps the type of class maps.
    labels = resample(np.array([[1, 2], [3, 4]], dtype=np.uint8), (4, 4), "nearest")
    # Each class becomes a 2 x 2 block.
    np.testing.assert_array_equal(labels, np.kron([[1, 2], [3, 4]], np.ones((2, 2))))
    # Data type.
    assert labels.dtype == np.uint8
    # Halving the pixel count doubles the pixel size.
    assert scaled_transform((10, 0, 500, 0, -10, 900), (4, 4), (2, 2)) == (20, 0, 500, 0, -20, 900)


# Array transforms.
def test_transforms() -> None:
    # Two bands of constant value.
    image = np.stack([np.full((2, 2), 3.0), np.full((2, 2), 10.0)])
    # Standardise with per-band statistics.
    out = Normalize(mean=[1.0, 4.0], std=[2.0, 3.0])(image)
    # (3 - 1) / 2 and (10 - 4) / 3.
    np.testing.assert_allclose(out[:, 0, 0], [1.0, 2.0])
    # Min-max scaling of [0, 255].
    assert Normalize(max_val=255.0)(np.array([51.0]))[0] == pytest.approx(0.2)
    # Padding at the bottom and right.
    padded = Pad((3, 4))(np.ones((1, 2, 2)))
    # New shape and zero fill.
    assert padded.shape == (1, 3, 4) and padded.sum() == 4
    # Composition applies in order.
    assert Compose([lambda v: v + 1, lambda v: v * 2])(1) == 4
    # Layout round trip.
    hwc = np.zeros((4, 5, 3))
    # (H, W, C) to (C, H, W) and back.
    assert to_tensor(hwc).shape == (3, 4, 5) and from_tensor(to_tensor(hwc)).shape == hwc.shape
    # Nearest resizing keeps labels.
    resized = Resize((4, 4), "nearest")(np.array([[1, 2], [3, 4]], dtype=np.uint8))
    # Labels are unchanged in value set.
    assert set(np.unique(resized)) == {1, 2, 3, 4}
    # Min-max normalisation per band ignores NaN.
    mm = minmax_normalize(np.array([[2.0, 4.0, np.nan, 6.0]]))
    # Hand-computed values.
    np.testing.assert_allclose(mm[0, [0, 1, 3]], [0.0, 0.5, 1.0])
    # Z-scores of [1, 3]: mean 2, population std 1.
    np.testing.assert_allclose(standardize(np.array([[1.0, 3.0]])), [[-1.0, 1.0]])


# =============================================================================
# End of module tests/unit/test_preprocessing.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
