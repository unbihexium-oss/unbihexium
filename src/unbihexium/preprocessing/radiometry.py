# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/preprocessing/radiometry.py
# Title       : Radiometric calibration and dark object subtraction
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Conversion of digital numbers (DN) to physical quantities for the optical
# missions most used in Earth observation, and image-based atmospheric
# correction:
#
#   sentinel2_reflectance       Sentinel-2 L1C/L2A DN to reflectance
#   parse_landsat_mtl           Landsat MTL.txt metadata to a flat dict
#   landsat_radiance            Landsat DN to at-sensor spectral radiance
#   landsat_toa_reflectance     Landsat DN to sun-corrected TOA reflectance
#   landsat_brightness_temperature
#                               thermal radiance to brightness temperature
#   landsat_c2l2_reflectance    Collection 2 Level-2 surface reflectance
#   landsat_c2l2_temperature    Collection 2 Level-2 surface temperature
#   earth_sun_distance          Earth-Sun distance in astronomical units
#   radiance_to_reflectance     radiance to TOA reflectance with ESUN
#   dark_object_subtraction     DOS1 haze removal in the reflectance domain
#
# Method
# ------
# Sentinel-2: reflectance = (DN + offset) / quantification. From processing
# baseline 04.00 (25 January 2022) the offset is -1000 (RADIO_ADD_OFFSET for
# L1C, BOA_ADD_OFFSET for L2A) and the quantification value 10000; older
# products have no offset. DN 0 marks no data.
#
# Landsat 8/9 Collection 2 Level-1: L = M_L Q + A_L and
# rho = (M_rho Q + A_rho) / sin(theta_SE), with the rescaling factors and the
# sun elevation theta_SE of the MTL file; brightness temperature
# T = K2 / ln(K1 / L + 1). Level-2 products use the fixed scale factors
# 2.75e-5 and offset -0.2 (reflectance) and 0.00341802 and offset 149.0
# (surface temperature in kelvin).
#
# DOS1 assumes that the darkest pixels of a band (a low percentile) would
# have a surface reflectance of 1 %, so the path reflectance is
# rho_dark - 0.01 and is subtracted from every pixel (Chavez, 1996).
#
# References
# ----------
#   ESA. Sentinel-2 products specification document, S2-PDGS-TAS-DI-PSD
#     (radiometric offset introduced with processing baseline 04.00).
#   USGS. Landsat 8-9 Collection 2 Level 1 data format control book
#     (LSDS-1822) and Collection 2 Level 2 science product guide (LSDS-1619).
#   Chander, G., Markham, B. L., Helder, D. L. (2009). Summary of current
#     radiometric calibration coefficients for Landsat MSS, TM, ETM+, and EO-1
#     ALI sensors. Remote Sensing of Environment 113(5), 893-903.
#   Spencer, J. W. (1971). Fourier series representation of the position of
#     the sun. Search 2(5), 172.
#   Chavez, P. S. (1988). An improved dark-object subtraction technique for
#     atmospheric scattering correction of multispectral data. Remote Sensing
#     of Environment 24(3), 459-479.
#   Chavez, P. S. (1996). Image-based atmospheric corrections, revisited and
#     improved. Photogrammetric Engineering and Remote Sensing 62(9),
#     1025-1036.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Regular expressions for the MTL parser.
import re

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Float conversion and band layout helpers.
from unbihexium.preprocessing.transforms import as_float, band_first

# Quantification value of Sentinel-2 L1C and L2A products.
S2_QUANTIFICATION = 10000.0

# Radiometric offset of Sentinel-2 products from processing baseline 04.00.
S2_OFFSET_PB04 = -1000.0

# Scale of Landsat Collection 2 Level-2 surface reflectance.
LANDSAT_C2L2_SR_SCALE = 2.75e-5

# Offset of Landsat Collection 2 Level-2 surface reflectance.
LANDSAT_C2L2_SR_OFFSET = -0.2

# Scale of Landsat Collection 2 Level-2 surface temperature (kelvin per DN).
LANDSAT_C2L2_ST_SCALE = 0.00341802

# Offset of Landsat Collection 2 Level-2 surface temperature (kelvin).
LANDSAT_C2L2_ST_OFFSET = 149.0

# One "KEY = value" line of an MTL file.
_MTL_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")


# Sentinel-2 L1C or L2A digital numbers to reflectance.
def sentinel2_reflectance(
    dn: NDArray[Any],  # Digital numbers, (H, W) or (C, H, W).
    offset: float | None = None,  # Additive offset; None derives it from the baseline.
    quantification: float = S2_QUANTIFICATION,  # Quantification value of the product.
    processing_baseline: str | float | None = None,  # Baseline such as "05.10".
    nodata: float | None = 0,  # DN of missing pixels.
) -> NDArray[np.float64]:  # Reflectance with NaN for missing pixels.
    # The quantification value must be positive.
    if quantification <= 0:
        # Explain the requirement.
        raise ValueError("quantification must be positive")
    # Derive the offset from the processing baseline when it is not given.
    if offset is None:
        # Without a baseline the modern convention is assumed.
        if processing_baseline is None:
            # Offset of every product processed since January 2022.
            offset = S2_OFFSET_PB04
        # Otherwise compare the baseline with 04.00.
        else:
            # Baselines are written as "NN.NN".
            baseline = float(processing_baseline)
            # Only baseline 04.00 and later carry the offset.
            offset = S2_OFFSET_PB04 if baseline >= 4.0 else 0.0
    # DN as float with nodata as NaN.
    x = as_float(dn, nodata)
    # Apply the offset and the quantification value.
    return (x + offset) / quantification


# Parse a Landsat MTL.txt file into a flat dictionary of values.
def parse_landsat_mtl(text: str) -> dict[str, Any]:
    # Parsed values by key.
    values: dict[str, Any] = {}
    # Visit every line of the file.
    for line in text.splitlines():
        # Match "KEY = value".
        match = _MTL_LINE.match(line)
        # Skip lines of other forms, such as END.
        if match is None:
            # Next line.
            continue
        # Key and raw value.
        key, raw = match.group(1), match.group(2)
        # Group markers structure the file but carry no value.
        if key in ("GROUP", "END_GROUP"):
            # Next line.
            continue
        # Quoted values are strings.
        if raw.startswith('"') and raw.endswith('"'):
            # Remove the quotes.
            values[key] = raw[1:-1]
            # Next line.
            continue
        # Unquoted values are numbers when they parse as such.
        try:
            # Integers keep their type.
            values[key] = int(raw)
        # Values that are not integers.
        except ValueError:
            # Try a float next.
            try:
                # Floats, including exponent notation.
                values[key] = float(raw)
            # Dates and other bare words stay strings.
            except ValueError:
                # Keep the raw text.
                values[key] = raw
    # Return the flat dictionary.
    return values


# Look up a per-band MTL value and fail with a clear message.
def _mtl_band_value(mtl: dict[str, Any], prefix: str, band: int | str) -> float:
    # Key such as REFLECTANCE_MULT_BAND_4.
    key = f"{prefix}_BAND_{band}"
    # The key must exist.
    if key not in mtl:
        # Explain what is missing.
        raise KeyError(f"MTL metadata has no {key}")
    # Return the value as float.
    return float(mtl[key])


# Landsat Level-1 DN to at-sensor spectral radiance in W / (m^2 sr um).
def landsat_radiance(
    dn: NDArray[Any],  # Quantised calibrated pixel values.
    mult: float,  # RADIANCE_MULT_BAND_x of the MTL file.
    add: float,  # RADIANCE_ADD_BAND_x of the MTL file.
    nodata: float | None = 0,  # DN of fill pixels.
) -> NDArray[np.float64]:  # Radiance with NaN for fill pixels.
    # Linear rescaling L = M_L Q + A_L.
    return mult * as_float(dn, nodata) + add


# Landsat Level-1 DN to sun-corrected top-of-atmosphere reflectance.
def landsat_toa_reflectance(
    dn: NDArray[Any],  # Quantised calibrated pixel values.
    mult: float,  # REFLECTANCE_MULT_BAND_x of the MTL file.
    add: float,  # REFLECTANCE_ADD_BAND_x of the MTL file.
    sun_elevation: float | NDArray[Any],  # SUN_ELEVATION in degrees, scalar or per pixel.
    nodata: float | None = 0,  # DN of fill pixels.
) -> NDArray[np.float64]:  # Reflectance with NaN for fill pixels.
    # Sun elevation in radians.
    elevation = np.deg2rad(np.asarray(sun_elevation, dtype=np.float64))
    # The sun must be above the horizon.
    if np.any(elevation <= 0):
        # Explain the requirement.
        raise ValueError("sun_elevation must be greater than 0 degrees")
    # Reflectance without the sun angle correction.
    rho = mult * as_float(dn, nodata) + add
    # Divide by the sine of the sun elevation.
    return rho / np.sin(elevation)


# Landsat Level-1 DN of one band to TOA reflectance with MTL metadata.
def landsat_toa_reflectance_from_mtl(
    dn: NDArray[Any],  # Pixel values of the band.
    mtl: dict[str, Any],  # Parsed MTL metadata.
    band: int | str,  # Band number as used in the MTL keys.
    nodata: float | None = 0,  # DN of fill pixels.
) -> NDArray[np.float64]:  # Reflectance with NaN for fill pixels.
    # Multiplicative rescaling factor of the band.
    mult = _mtl_band_value(mtl, "REFLECTANCE_MULT", band)
    # Additive rescaling factor of the band.
    add = _mtl_band_value(mtl, "REFLECTANCE_ADD", band)
    # The sun elevation of the scene centre is required.
    if "SUN_ELEVATION" not in mtl:
        # Explain what is missing.
        raise KeyError("MTL metadata has no SUN_ELEVATION")
    # Convert the band.
    return landsat_toa_reflectance(dn, mult, add, float(mtl["SUN_ELEVATION"]), nodata)


# Thermal radiance to at-sensor brightness temperature in kelvin.
def landsat_brightness_temperature(
    radiance: NDArray[Any],  # Spectral radiance of a thermal band.
    k1: float,  # K1_CONSTANT_BAND_x of the MTL file.
    k2: float,  # K2_CONSTANT_BAND_x of the MTL file.
) -> NDArray[np.float64]:  # Brightness temperature, NaN for non-positive radiance.
    # Radiance as float.
    radiance = np.asarray(radiance, dtype=np.float64)
    # Non-positive radiance has no temperature.
    safe = np.where(radiance > 0, radiance, np.nan)
    # Inverse Planck function T = K2 / ln(K1 / L + 1).
    return k2 / np.log(k1 / safe + 1.0)


# Landsat Collection 2 Level-2 DN to surface reflectance.
def landsat_c2l2_reflectance(dn: NDArray[Any], nodata: float | None = 0) -> NDArray[np.float64]:
    # Fixed scale and offset of the Level-2 surface reflectance product.
    return LANDSAT_C2L2_SR_SCALE * as_float(dn, nodata) + LANDSAT_C2L2_SR_OFFSET


# Landsat Collection 2 Level-2 DN to surface temperature in kelvin.
def landsat_c2l2_temperature(dn: NDArray[Any], nodata: float | None = 0) -> NDArray[np.float64]:
    # Fixed scale and offset of the Level-2 surface temperature product.
    return LANDSAT_C2L2_ST_SCALE * as_float(dn, nodata) + LANDSAT_C2L2_ST_OFFSET


# Earth-Sun distance in astronomical units for a day of the year.
def earth_sun_distance(day_of_year: int | NDArray[Any]) -> float | NDArray[np.float64]:
    # Day of the year as float.
    doy = np.asarray(day_of_year, dtype=np.float64)
    # Days must lie within a year.
    if np.any((doy < 1) | (doy > 366)):
        # Explain the valid range.
        raise ValueError("day_of_year must be between 1 and 366")
    # Day angle of Spencer (1971) in radians.
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    # Eccentricity correction factor (r0 / r)^2 of Spencer (1971).
    e0 = (
        1.000110  # Constant term.
        + 0.034221 * np.cos(gamma)  # First cosine harmonic.
        + 0.001280 * np.sin(gamma)  # First sine harmonic.
        + 0.000719 * np.cos(2 * gamma)  # Second cosine harmonic.
        + 0.000077 * np.sin(2 * gamma)  # Second sine harmonic.
    )  # End of the Fourier series.
    # Distance in astronomical units.
    d = 1.0 / np.sqrt(e0)
    # Scalars stay scalars.
    return float(d) if np.ndim(d) == 0 else d


# At-sensor radiance to TOA reflectance with the solar exoatmospheric irradiance.
def radiance_to_reflectance(
    radiance: NDArray[Any],  # Spectral radiance in W / (m^2 sr um).
    esun: float,  # Mean exoatmospheric solar irradiance of the band, W / (m^2 um).
    sun_zenith: float | NDArray[Any],  # Solar zenith angle in degrees.
    day_of_year: int,  # Acquisition day for the Earth-Sun distance.
) -> NDArray[np.float64]:  # TOA reflectance.
    # The irradiance must be positive.
    if esun <= 0:
        # Explain the requirement.
        raise ValueError("esun must be positive")
    # Cosine of the solar zenith angle.
    cos_zenith = np.cos(np.deg2rad(np.asarray(sun_zenith, dtype=np.float64)))
    # The sun must be above the horizon.
    if np.any(cos_zenith <= 0):
        # Explain the requirement.
        raise ValueError("sun_zenith must be below 90 degrees")
    # Earth-Sun distance in astronomical units.
    d = earth_sun_distance(day_of_year)
    # rho = pi L d^2 / (ESUN cos(theta_s)).
    return np.pi * np.asarray(radiance, dtype=np.float64) * d**2 / (esun * cos_zenith)


# DOS1 haze removal of TOA reflectance, band by band.
def dark_object_subtraction(
    reflectance: NDArray[Any],  # TOA reflectance, (H, W) or (C, H, W).
    dark_percentile: float = 0.01,  # Percentile that defines the dark object.
    dark_reflectance: float = 0.01,  # Assumed surface reflectance of the dark object.
    clip: bool = True,  # Clip negative results to zero.
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # Corrected bands and path reflectance.
    # The percentile must lie in [0, 100].
    if not 0.0 <= dark_percentile <= 100.0:
        # Explain the valid range.
        raise ValueError("dark_percentile must be between 0 and 100")
    # Float copy in band-first layout.
    x = band_first(np.asarray(reflectance, dtype=np.float64))
    # Every band needs finite values.
    if not np.isfinite(x).any(axis=(1, 2)).all():
        # Explain the requirement.
        raise ValueError("every band needs at least one finite value")
    # Reflectance of the dark object per band.
    dark = np.nanpercentile(x, dark_percentile, axis=(1, 2))
    # Path reflectance, never negative.
    haze = np.maximum(dark - dark_reflectance, 0.0)
    # Subtract the path reflectance.
    out = x - haze[:, None, None]
    # Negative reflectance is not physical.
    if clip:
        # Clip, keeping NaN.
        out = np.where(out < 0, 0.0, out)
    # Restore the input rank.
    return (out[0] if np.ndim(reflectance) == 2 else out), haze


# =============================================================================
# End of module src/unbihexium/preprocessing/radiometry.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
