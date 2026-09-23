# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/sensor.py
# Title       : Satellite sensor definitions and radiometric conversions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Band tables of common Earth observation sensors and the conversions from
# stored digital numbers (DN) to physical quantities:
#
#   SpectralBand          one band: name, common name, centre, width, pixel
#   SARMode               one SAR acquisition mode: swath and resolution
#   SensorModel           a sensor with its bands and orbit parameters
#   SENSORS, get_sensor   Sentinel-2 MSI, Landsat 8 OLI/TIRS, Landsat 9
#                         OLI-2/TIRS-2 and Sentinel-1 C-SAR
#   to_reflectance        DN * scale + offset with no-data handling
#   landsat_toa_reflectance, landsat_brightness_temperature
#
# Common band names (BLUE, GREEN, RED, NIR, SWIR1, ...) let the spectral
# index library use the same formula for every sensor; band_aliases maps the
# product band names (B04, SR_B4, ...) to them.
#
# Band values
# -----------
# Sentinel-2: central wavelength and bandwidth of Sentinel-2A from the ESA
# Sentinel-2 spectral response functions.
# Landsat 8 and 9: band limits from the USGS Landsat 8-9 Data Users Handbook;
# the centre is the middle of the limits. Sentinel-1: centre frequency and
# mode resolutions (range x azimuth, single look) from the ESA Sentinel-1
# User Handbook.
#
# Radiometric conversions
# -----------------------
# Sentinel-2 Level-2A stores surface reflectance as DN / 10000 with an
# offset of -1000 DN (-0.1 reflectance) from processing baseline 04.00 (25
# January 2022) on. Landsat Collection 2 Level-2 stores surface reflectance
# as DN * 2.75e-5 - 0.2 and surface temperature as DN * 0.00341802 + 149.0
# Kelvin. Landsat Level-1 top-of-atmosphere reflectance is
# (M * DN + A) / sin(sun elevation), and brightness temperature is
# K2 / ln(K1 / L + 1) with the radiance L = ML * DN + AL.
#
# References
# ----------
#   Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution
#     mission for GMES operational services. Remote Sensing of Environment
#     120, 25-36.
#   Torres, R., et al. (2012). GMES Sentinel-1 mission. Remote Sensing of
#     Environment 120, 9-24.
#   U.S. Geological Survey (2019). Landsat 8 (L8) Data Users Handbook.
#     LSDS-1574.
#   U.S. Geological Survey (2022). Landsat 8-9 Collection 2 Level 2 Science
#     Product Guide. LSDS-1619.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Record containers.
from dataclasses import dataclass, field

# Sensor families.
from enum import Enum

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Speed of light in vacuum in metres per second (exact by definition).
SPEED_OF_LIGHT = 299_792_458.0

# Sentinel-2 Level-2A reflectance scale (quantification value 10000).
SENTINEL2_L2A_SCALE = 1.0e-4

# Sentinel-2 Level-2A additive offset from processing baseline 04.00 on.
SENTINEL2_L2A_OFFSET = -0.1

# Landsat Collection 2 Level-2 surface reflectance scale.
LANDSAT_C2_SR_SCALE = 2.75e-5

# Landsat Collection 2 Level-2 surface reflectance offset.
LANDSAT_C2_SR_OFFSET = -0.2

# Landsat Collection 2 Level-2 surface temperature scale (Kelvin per DN).
LANDSAT_C2_ST_SCALE = 0.00341802

# Landsat Collection 2 Level-2 surface temperature offset (Kelvin).
LANDSAT_C2_ST_OFFSET = 149.0

# Thermal constants K1 (W m-2 sr-1 um-1) and K2 (K) of Landsat 8 TIRS.
LANDSAT8_THERMAL_CONSTANTS = {
    "B10": (774.8853, 1321.0789),  # Band 10, 10.9 um.
    "B11": (480.8883, 1201.1442),  # Band 11, 12.0 um.
}  # End of the thermal constants.


# Families of sensors.
class SensorType(str, Enum):
    # Optical imagers in general.
    OPTICAL = "optical"
    # Synthetic aperture radar.
    SAR = "sar"
    # Imagers with a few broad bands.
    MULTISPECTRAL = "multispectral"
    # Imagers with many narrow bands.
    HYPERSPECTRAL = "hyperspectral"
    # Single broad visible band.
    PANCHROMATIC = "panchromatic"
    # Thermal infrared imagers.
    THERMAL = "thermal"


# One spectral band of an optical sensor.
@dataclass(frozen=True)
class SpectralBand:
    # Band name in the products, for example "B04".
    name: str
    # Common name used by the index library, for example "RED".
    common_name: str
    # Central wavelength in nanometres.
    center_nm: float
    # Bandwidth in nanometres.
    bandwidth_nm: float
    # Ground sampling distance in metres.
    resolution_m: float

    # Lower limit of the band in nanometres.
    @property
    def lower_nm(self) -> float:
        # Centre minus half the width.
        return self.center_nm - self.bandwidth_nm / 2.0

    # Upper limit of the band in nanometres.
    @property
    def upper_nm(self) -> float:
        # Centre plus half the width.
        return self.center_nm + self.bandwidth_nm / 2.0

    # Whether a wavelength in nanometres falls inside the band.
    def contains(self, wavelength_nm: float) -> bool:
        # Closed interval test.
        return self.lower_nm <= wavelength_nm <= self.upper_nm


# Band from its lower and upper limits in nanometres.
def band_from_limits(
    name: str,  # Band name.
    common_name: str,  # Common name.
    lower_nm: float,  # Lower limit.
    upper_nm: float,  # Upper limit.
    resolution_m: float,  # Ground sampling distance.
) -> SpectralBand:  # The band.
    # Limits must be ordered.
    if upper_nm <= lower_nm:
        # Explain the problem.
        raise ValueError(f"band {name}: upper limit {upper_nm} <= lower limit {lower_nm}")
    # Centre and width from the limits.
    center = (lower_nm + upper_nm) / 2.0
    # Band with the centre and the width.
    return SpectralBand(name, common_name, center, upper_nm - lower_nm, resolution_m)


# One acquisition mode of a SAR sensor.
@dataclass(frozen=True)
class SARMode:
    # Mode name, for example "IW".
    name: str
    # Swath width in kilometres.
    swath_km: float
    # Single-look ground range resolution in metres.
    range_resolution_m: float
    # Single-look azimuth resolution in metres.
    azimuth_resolution_m: float
    # Available polarisations.
    polarisations: tuple[str, ...] = ("VV", "VH", "HH", "HV")


# A satellite sensor with its bands and orbit.
@dataclass
class SensorModel:
    # Sensor name.
    name: str
    # Sensor family.
    sensor_type: SensorType
    # Platform or constellation.
    platform: str = ""
    # Finest ground sampling distance in metres.
    resolution: float = 1.0
    # Swath width in kilometres.
    swath_width: float = 0.0
    # Band names; filled from spectral_bands when empty.
    bands: list[str] = field(default_factory=list)
    # Band limits in nanometres; filled from spectral_bands when empty.
    band_wavelengths: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Revisit time of the constellation in days.
    revisit_time_days: float = 0.0
    # Mean orbit altitude in kilometres.
    altitude_km: float = 0.0
    # Orbit inclination in degrees.
    inclination_deg: float = 0.0
    # Launch date of the first satellite (ISO 8601).
    launch_date: str = ""
    # Operator of the mission.
    operator: str = ""
    # Detailed optical bands.
    spectral_bands: tuple[SpectralBand, ...] = ()
    # Centre frequency of a radar in gigahertz.
    frequency_ghz: float | None = None
    # Acquisition modes of a radar.
    sar_modes: tuple[SARMode, ...] = ()
    # Product name prefixes that alias the band names, e.g. "SR_" for Landsat.
    alias_prefixes: tuple[str, ...] = ()

    # Fill the simple band fields from the detailed bands.
    def __post_init__(self) -> None:
        # Band names in order.
        if not self.bands and self.spectral_bands:
            # One name per band.
            self.bands = [b.name for b in self.spectral_bands]
        # Band limits.
        if not self.band_wavelengths and self.spectral_bands:
            # Lower and upper limit per band.
            self.band_wavelengths = {b.name: (b.lower_nm, b.upper_nm) for b in self.spectral_bands}

    # Band limits in nanometres, or None for unknown bands.
    def get_band_wavelength(self, band: str) -> tuple[float, float] | None:
        # Look the band up by name or alias.
        found = self.find_band(band)
        # Limits of the band.
        return (found.lower_nm, found.upper_nm) if found else self.band_wavelengths.get(band)

    # Band by product name, alias or common name (case-insensitive).
    def find_band(self, name: str) -> SpectralBand | None:
        # Canonical common name of the requested name, when it is an alias.
        target = self.band_aliases().get(name.upper(), name.upper())
        # Search the bands.
        for band in self.spectral_bands:
            # Match on the name or on the common name.
            if target in (band.name.upper(), band.common_name):
                # Found.
                return band
        # Unknown band.
        return None

    # Band by name; raises for unknown bands.
    def band(self, name: str) -> SpectralBand:
        # Look the band up.
        found = self.find_band(name)
        # Unknown bands are an error.
        if found is None:
            # Explain the problem with the known bands.
            raise KeyError(f"{self.name} has no band {name!r}; bands: {', '.join(self.bands)}")
        # Return the band.
        return found

    # Map from product band names (upper case) to common names.
    def band_aliases(self) -> dict[str, str]:
        # Collected aliases.
        aliases: dict[str, str] = {}
        # Visit every band.
        for band in self.spectral_bands:
            # Product name.
            names = [band.name.upper()]
            # "B04" is also written "B4".
            if len(band.name) == 3 and band.name[1] == "0":
                # Name without the leading zero.
                names.append(band.name[0] + band.name[2])
            # Prefixed product names such as "SR_B4".
            names += [prefix + n for prefix in self.alias_prefixes for n in list(names)]
            # Record every form.
            for alias in names:
                # Alias to common name.
                aliases[alias] = band.common_name
        # Return the map.
        return aliases

    # Bands whose ground sampling distance equals the given value.
    def bands_at_resolution(self, resolution_m: float) -> list[str]:
        # Names of the matching bands.
        return [b.name for b in self.spectral_bands if b.resolution_m == resolution_m]

    # Radar wavelength in metres, from the centre frequency.
    @property
    def wavelength_m(self) -> float | None:
        # Only radars have a centre frequency.
        if self.frequency_ghz is None:
            # Not a radar.
            return None
        # Wavelength = c / f.
        return SPEED_OF_LIGHT / (self.frequency_ghz * 1e9)

    # Acquisition mode by name.
    def mode(self, name: str) -> SARMode:
        # Search the modes.
        for mode in self.sar_modes:
            # Case-insensitive match.
            if mode.name.upper() == name.upper():
                # Found.
                return mode
        # Unknown modes are an error.
        raise KeyError(f"{self.name} has no acquisition mode {name!r}")

    # Plain dictionary for JSON output.
    def to_dict(self) -> dict[str, Any]:
        # One entry per field.
        return {
            "name": self.name,  # Sensor name.
            "sensor_type": self.sensor_type.value,  # Family.
            "platform": self.platform,  # Platform.
            "resolution": self.resolution,  # Finest pixel.
            "swath_width": self.swath_width,  # Swath.
            "bands": list(self.bands),  # Band names.
            "band_wavelengths": {k: list(v) for k, v in self.band_wavelengths.items()},  # Limits.
            "revisit_time_days": self.revisit_time_days,  # Revisit.
            "altitude_km": self.altitude_km,  # Altitude.
            "inclination_deg": self.inclination_deg,  # Inclination.
            "launch_date": self.launch_date,  # Launch.
            "operator": self.operator,  # Operator.
            "frequency_ghz": self.frequency_ghz,  # Radar frequency.
            "sar_modes": [m.name for m in self.sar_modes],  # Radar modes.
        }  # End of the dictionary.


# Sentinel-2A MSI bands: name, common name, centre, width (nm), pixel (m).
_S2_BANDS = (
    SpectralBand("B01", "COASTAL", 442.7, 21.0, 60.0),  # Coastal aerosol.
    SpectralBand("B02", "BLUE", 492.4, 66.0, 10.0),  # Blue.
    SpectralBand("B03", "GREEN", 559.8, 36.0, 10.0),  # Green.
    SpectralBand("B04", "RED", 664.6, 31.0, 10.0),  # Red.
    SpectralBand("B05", "REDEDGE1", 704.1, 15.0, 20.0),  # Vegetation red edge 1.
    SpectralBand("B06", "REDEDGE2", 740.5, 15.0, 20.0),  # Vegetation red edge 2.
    SpectralBand("B07", "REDEDGE3", 782.8, 20.0, 20.0),  # Vegetation red edge 3.
    SpectralBand("B08", "NIR", 832.8, 106.0, 10.0),  # Broad near infrared.
    SpectralBand("B8A", "NIR08", 864.7, 21.0, 20.0),  # Narrow near infrared.
    SpectralBand("B09", "WATERVAPOUR", 945.1, 20.0, 60.0),  # Water vapour.
    SpectralBand("B10", "CIRRUS", 1373.5, 31.0, 60.0),  # Cirrus.
    SpectralBand("B11", "SWIR1", 1613.7, 91.0, 20.0),  # Shortwave infrared 1.6 um.
    SpectralBand("B12", "SWIR2", 2202.4, 175.0, 20.0),  # Shortwave infrared 2.2 um.
)  # End of the Sentinel-2 bands.

# Landsat 8 OLI and TIRS bands from their limits in nanometres.
_LANDSAT_BANDS = (
    band_from_limits("B1", "COASTAL", 430.0, 450.0, 30.0),  # Coastal aerosol.
    band_from_limits("B2", "BLUE", 450.0, 510.0, 30.0),  # Blue.
    band_from_limits("B3", "GREEN", 530.0, 590.0, 30.0),  # Green.
    band_from_limits("B4", "RED", 640.0, 670.0, 30.0),  # Red.
    band_from_limits("B5", "NIR", 850.0, 880.0, 30.0),  # Near infrared.
    band_from_limits("B6", "SWIR1", 1570.0, 1650.0, 30.0),  # Shortwave infrared 1.
    band_from_limits("B7", "SWIR2", 2110.0, 2290.0, 30.0),  # Shortwave infrared 2.
    band_from_limits("B8", "PAN", 500.0, 680.0, 15.0),  # Panchromatic.
    band_from_limits("B9", "CIRRUS", 1360.0, 1380.0, 30.0),  # Cirrus.
    band_from_limits("B10", "TIR1", 10600.0, 11190.0, 100.0),  # Thermal infrared 1.
    band_from_limits("B11", "TIR2", 11500.0, 12510.0, 100.0),  # Thermal infrared 2.
)  # End of the Landsat bands.

# Sentinel-1 C-SAR acquisition modes (single-look resolution, range x azimuth).
_S1_MODES = (
    SARMode("SM", 80.0, 5.0, 5.0),  # Stripmap.
    SARMode("IW", 250.0, 5.0, 20.0),  # Interferometric Wide swath.
    SARMode("EW", 400.0, 20.0, 40.0),  # Extra Wide swath.
    SARMode("WV", 20.0, 5.0, 5.0, ("VV", "HH")),  # Wave mode vignettes.
)  # End of the Sentinel-1 modes.

# Sensors by identifier.
SENSORS: dict[str, SensorModel] = {
    "sentinel2_msi": SensorModel(  # Sentinel-2 Multispectral Instrument.
        name="Sentinel-2 MSI",  # Multispectral Instrument.
        sensor_type=SensorType.MULTISPECTRAL,  # Family.
        platform="Sentinel-2",  # Constellation.
        resolution=10.0,  # Finest pixel.
        swath_width=290.0,  # Swath in km.
        revisit_time_days=5.0,  # With two satellites at the equator.
        altitude_km=786.0,  # Mean altitude.
        inclination_deg=98.62,  # Sun-synchronous orbit.
        launch_date="2015-06-23",  # Sentinel-2A.
        operator="ESA",  # European Space Agency.
        spectral_bands=_S2_BANDS,  # Bands.
    ),
    "landsat8_oli": SensorModel(  # Landsat 8.
        name="Landsat 8 OLI/TIRS",  # Operational Land Imager and TIRS.
        sensor_type=SensorType.MULTISPECTRAL,  # Family.
        platform="Landsat 8",  # Satellite.
        resolution=15.0,  # Panchromatic pixel.
        swath_width=185.0,  # Swath in km.
        revisit_time_days=16.0,  # Repeat cycle of one satellite.
        altitude_km=705.0,  # Mean altitude.
        inclination_deg=98.2,  # Sun-synchronous orbit.
        launch_date="2013-02-11",  # Launch.
        operator="USGS/NASA",  # Operators.
        spectral_bands=_LANDSAT_BANDS,  # Bands.
        alias_prefixes=("SR_", "ST_"),  # Collection 2 Level-2 band names.
    ),
    "landsat9_oli2": SensorModel(  # Landsat 9.
        name="Landsat 9 OLI-2/TIRS-2",  # Second-generation instruments.
        sensor_type=SensorType.MULTISPECTRAL,  # Family.
        platform="Landsat 9",  # Satellite.
        resolution=15.0,  # Panchromatic pixel.
        swath_width=185.0,  # Swath in km.
        revisit_time_days=16.0,  # Repeat cycle; 8 days together with Landsat 8.
        altitude_km=705.0,  # Mean altitude.
        inclination_deg=98.2,  # Sun-synchronous orbit.
        launch_date="2021-09-27",  # Launch.
        operator="USGS/NASA",  # Operators.
        spectral_bands=_LANDSAT_BANDS,  # Same band limits as Landsat 8.
        alias_prefixes=("SR_", "ST_"),  # Collection 2 Level-2 band names.
    ),
    "sentinel1_sar": SensorModel(  # Sentinel-1 radar.
        name="Sentinel-1 C-SAR",  # C-band synthetic aperture radar.
        sensor_type=SensorType.SAR,  # Family.
        platform="Sentinel-1",  # Constellation.
        resolution=5.0,  # Finest single-look resolution.
        swath_width=250.0,  # Swath of the default IW mode.
        bands=["VV", "VH", "HH", "HV"],  # Polarisations.
        revisit_time_days=12.0,  # Repeat cycle of one satellite.
        altitude_km=693.0,  # Mean altitude.
        inclination_deg=98.18,  # Sun-synchronous orbit.
        launch_date="2014-04-03",  # Sentinel-1A.
        operator="ESA",  # European Space Agency.
        frequency_ghz=5.405,  # C-band centre frequency.
        sar_modes=_S1_MODES,  # Acquisition modes.
    ),
}  # End of the sensors.

# Short names accepted by get_sensor.
_SENSOR_ALIASES = {
    "sentinel2": "sentinel2_msi",  # Sentinel-2.
    "s2": "sentinel2_msi",  # Sentinel-2.
    "landsat8": "landsat8_oli",  # Landsat 8.
    "l8": "landsat8_oli",  # Landsat 8.
    "landsat9": "landsat9_oli2",  # Landsat 9.
    "l9": "landsat9_oli2",  # Landsat 9.
    "sentinel1": "sentinel1_sar",  # Sentinel-1.
    "s1": "sentinel1_sar",  # Sentinel-1.
}  # End of the sensor aliases.


# Sensor by identifier or short name, or None when unknown.
def get_sensor(sensor_id: str) -> SensorModel | None:
    # Normalised key.
    key = sensor_id.lower().replace("-", "").replace(" ", "")
    # Resolve short names.
    key = _SENSOR_ALIASES.get(key, key)
    # Look the sensor up.
    return SENSORS.get(key)


# Identifiers of the known sensors.
def list_sensors() -> list[str]:
    # Keys of the table.
    return list(SENSORS)


# Convert stored digital numbers to physical values: DN * scale + offset.
def to_reflectance(
    dn: NDArray[Any],  # Stored values.
    scale: float,  # Multiplicative factor.
    offset: float = 0.0,  # Additive offset.
    nodata: float | None = None,  # Stored no-data value, becomes NaN.
) -> NDArray[np.float64]:  # Physical values.
    # Work in float64.
    values = np.asarray(dn, dtype=np.float64)
    # Linear conversion.
    result = values * scale + offset
    # Mark missing values.
    if nodata is not None:
        # NaN where the stored value is the no-data value.
        result[values == nodata] = np.nan
    # Return the physical values.
    return result


# Additive offset of Sentinel-2 L2A for a processing baseline such as "04.00".
def sentinel2_l2a_offset(processing_baseline: str) -> float:
    # Baseline as a number; "N0400" and "04.00" are both accepted.
    text = processing_baseline.upper().lstrip("N")
    # "0400" means 04.00.
    number = float(text) if "." in text else float(text) / 100.0
    # The offset exists from baseline 04.00 on.
    return SENTINEL2_L2A_OFFSET if number >= 4.0 else 0.0


# Landsat Level-1 top-of-atmosphere reflectance with sun angle correction.
def landsat_toa_reflectance(
    dn: NDArray[Any],  # Quantised calibrated values (Q_cal).
    mult: float,  # REFLECTANCE_MULT_BAND_x of the metadata.
    add: float,  # REFLECTANCE_ADD_BAND_x of the metadata.
    sun_elevation_deg: float,  # SUN_ELEVATION of the metadata.
    nodata: float | None = 0.0,  # Fill value of Level-1 products.
) -> NDArray[np.float64]:  # Reflectance.
    # The sun must be above the horizon.
    if not 0.0 < sun_elevation_deg <= 90.0:
        # Explain the problem.
        raise ValueError(f"sun elevation must be in (0, 90] degrees, got {sun_elevation_deg}")
    # Reflectance without the sun angle correction.
    rho = to_reflectance(dn, mult, add, nodata)
    # Divide by the sine of the sun elevation.
    return rho / np.sin(np.deg2rad(sun_elevation_deg))


# Landsat at-sensor brightness temperature in Kelvin.
def landsat_brightness_temperature(
    dn: NDArray[Any],  # Quantised calibrated values (Q_cal).
    ml: float,  # RADIANCE_MULT_BAND_x of the metadata.
    al: float,  # RADIANCE_ADD_BAND_x of the metadata.
    k1: float,  # K1_CONSTANT_BAND_x.
    k2: float,  # K2_CONSTANT_BAND_x.
    nodata: float | None = 0.0,  # Fill value of Level-1 products.
) -> NDArray[np.float64]:  # Temperature in Kelvin.
    # Spectral radiance.
    radiance = to_reflectance(dn, ml, al, nodata)
    # Non-positive radiance has no temperature.
    radiance = np.where(radiance > 0, radiance, np.nan)
    # Inverse Planck function with the band constants.
    return k2 / np.log(k1 / radiance + 1.0)


# =============================================================================
# End of module src/unbihexium/core/sensor.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
