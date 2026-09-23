# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/core/index.py
# Title       : Spectral index library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Spectral indices computed from surface reflectance bands. Every index is a
# SpectralIndex with its formula, required bands, parameters and reference,
# and is registered in IndexRegistry under its name (case-insensitive):
#
#   vegetation   NDVI, GNDVI, NDRE, EVI, EVI2, SAVI, MSAVI, OSAVI, ARVI,
#                VARI, SR, WDRVI, CIgreen, CIre
#   water        NDWI, MNDWI, NDMI, AWEInsh, AWEIsh, NDTI, NDCI
#   burn         NBR, NBR2 (and dNBR via difference_index / dnbr)
#   urban, soil  NDBI, BSI
#   snow         NDSI
#   moisture     MSI
#
# Bands are passed as a dictionary of arrays keyed by common band names
# (COASTAL, BLUE, GREEN, RED, REDEDGE1..3, NIR, NIR08, SWIR1, SWIR2) or, when
# a sensor is given, by the product names of that sensor (B04 or SR_B4, see
# unbihexium.core.sensor). Pixels where any input is NaN, infinite or equal
# to the no-data value, or where a denominator is zero, become NaN. Results
# are float64.
#
# The value_range of an index is its range for reflectances in [0, 1]; an
# infinite limit means the index is unbounded.
#
# Burn severity: classify_burn_severity maps dNBR = NBR(pre) - NBR(post) to
# the seven classes of Key and Benson (2006) with the limits -0.25, -0.1,
# 0.1, 0.27, 0.44 and 0.66.
#
# Usage
# -----
#   ndvi = compute_index("NDVI", {"NIR": nir, "RED": red})
#   evi = compute_index("EVI", {"B02": b2, "B04": b4, "B08": b8}, sensor="sentinel2")
#   severity = classify_burn_severity(dnbr(pre_bands, post_bands))
#
# References
# ----------
#   Rouse, J. W., Haas, R. H., Schell, J. A., Deering, D. W. (1974).
#     Monitoring vegetation systems in the Great Plains with ERTS. Third ERTS
#     Symposium, NASA SP-351, 309-317. (NDVI)
#   Jordan, C. F. (1969). Derivation of leaf-area index from quality of light
#     on the forest floor. Ecology 50(4), 663-666. (SR)
#   Gitelson, A. A., Kaufman, Y. J., Merzlyak, M. N. (1996). Use of a green
#     channel in remote sensing of global vegetation from EOS-MODIS. Remote
#     Sensing of Environment 58(3), 289-298. (GNDVI)
#   Gitelson, A. A., Merzlyak, M. N. (1994). Spectral reflectance changes
#     associated with autumn senescence of Aesculus hippocastanum L. and Acer
#     platanoides L. leaves. Journal of Plant Physiology 143(3), 286-292.
#     (NDRE)
#   Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., Ferreira,
#     L. G. (2002). Overview of the radiometric and biophysical performance of
#     the MODIS vegetation indices. Remote Sensing of Environment 83(1-2),
#     195-213. (EVI)
#   Jiang, Z., Huete, A. R., Didan, K., Miura, T. (2008). Development of a
#     two-band enhanced vegetation index without a blue band. Remote Sensing
#     of Environment 112(10), 3833-3845. (EVI2)
#   Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). Remote
#     Sensing of Environment 25(3), 295-309. (SAVI)
#   Qi, J., Chehbouni, A., Huete, A. R., Kerr, Y. H., Sorooshian, S. (1994).
#     A modified soil adjusted vegetation index. Remote Sensing of
#     Environment 48(2), 119-126. (MSAVI)
#   Rondeaux, G., Steven, M., Baret, F. (1996). Optimization of
#     soil-adjusted vegetation indices. Remote Sensing of Environment 55(2),
#     95-107. (OSAVI)
#   Kaufman, Y. J., Tanre, D. (1992). Atmospherically resistant vegetation
#     index (ARVI) for EOS-MODIS. IEEE Transactions on Geoscience and Remote
#     Sensing 30(2), 261-270. (ARVI)
#   Gitelson, A. A., Kaufman, Y. J., Stark, R., Rundquist, D. (2002). Novel
#     algorithms for remote estimation of vegetation fraction. Remote Sensing
#     of Environment 80(1), 76-87. (VARI)
#   Gitelson, A. A. (2004). Wide dynamic range vegetation index for remote
#     quantification of biophysical characteristics of vegetation. Journal
#     of Plant Physiology 161(2), 165-173. (WDRVI)
#   Gitelson, A. A., Gritz, Y., Merzlyak, M. N. (2003). Relationships between
#     leaf chlorophyll content and spectral reflectance and algorithms for
#     non-destructive chlorophyll assessment in higher plant leaves. Journal
#     of Plant Physiology 160(3), 271-282. (CIgreen, CIre)
#   McFeeters, S. K. (1996). The use of the Normalized Difference Water Index
#     (NDWI) in the delineation of open water features. International
#     Journal of Remote Sensing 17(7), 1425-1432. (NDWI)
#   Xu, H. (2006). Modification of normalised difference water index (NDWI)
#     to enhance open water features in remotely sensed imagery.
#     International Journal of Remote Sensing 27(14), 3025-3033. (MNDWI)
#   Gao, B.-C. (1996). NDWI, a normalized difference water index for remote
#     sensing of vegetation liquid water from space. Remote Sensing of
#     Environment 58(3), 257-266. (NDMI)
#   Feyisa, G. L., Meilby, H., Fensholt, R., Proud, S. R. (2014). Automated
#     Water Extraction Index: a new technique for surface water mapping using
#     Landsat imagery. Remote Sensing of Environment 140, 23-35. (AWEI)
#   Lacaux, J. P., Tourre, Y. M., Vignolles, C., Ndione, J. A., Lafaye, M.
#     (2007). Classification of ponds from high-spatial resolution remote
#     sensing: application to Rift Valley Fever epidemics in Senegal. Remote
#     Sensing of Environment 106(1), 66-74. (NDTI)
#   Mishra, S., Mishra, D. R. (2012). Normalized difference chlorophyll
#     index: a novel model for remote estimation of chlorophyll-a
#     concentration in turbid productive waters. Remote Sensing of
#     Environment 117, 394-406. (NDCI)
#   Key, C. H., Benson, N. C. (2006). Landscape assessment (LA). In FIREMON:
#     Fire Effects Monitoring and Inventory System. USDA Forest Service,
#     General Technical Report RMRS-GTR-164-CD. (NBR, dNBR)
#   Zha, Y., Gao, J., Ni, S. (2003). Use of normalized difference built-up
#     index in automatically mapping urban areas from TM imagery.
#     International Journal of Remote Sensing 24(3), 583-594. (NDBI)
#   Rikimaru, A., Roy, P. S., Miyatake, S. (2002). Tropical forest cover
#     density mapping. Tropical Ecology 43(1), 39-47. (BSI)
#   Hall, D. K., Riggs, G. A., Salomonson, V. V. (1995). Development of
#     methods for mapping global snow cover using moderate resolution imaging
#     spectroradiometer data. Remote Sensing of Environment 54(2), 127-140.
#     (NDSI)
#   Hunt, E. R., Rock, B. N. (1989). Detection of changes in leaf water
#     content using near- and middle-infrared reflectances. Remote Sensing of
#     Environment 30(1), 43-54. (MSI)
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Index functions and band dictionaries.
from collections.abc import Callable, Iterable, Mapping

# Index definitions.
from dataclasses import dataclass, field

# Index categories.
from enum import Enum

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Sensor band aliases.
from unbihexium.core.sensor import get_sensor

# Float arrays used by the formulas.
Array = NDArray[np.float64]

# Index formula: bands and parameters in, index values out.
IndexFunction = Callable[[Mapping[str, Array], Mapping[str, float]], Array]

# Unbounded limit of a value range.
INF = float("inf")

# Band names accepted for the common names regardless of the sensor.
GENERIC_ALIASES = {
    "RE1": "REDEDGE1",  # Red edge 1, about 705 nm.
    "RE2": "REDEDGE2",  # Red edge 2, about 740 nm.
    "RE3": "REDEDGE3",  # Red edge 3, about 783 nm.
    "REDEDGE": "REDEDGE1",  # First red edge band.
    "SWIR16": "SWIR1",  # Shortwave infrared at 1.6 um.
    "SWIR22": "SWIR2",  # Shortwave infrared at 2.2 um.
}  # End of the generic aliases.

# Upper limits of the burn severity classes of dNBR (Key and Benson, 2006).
BURN_SEVERITY_LIMITS = (-0.25, -0.1, 0.1, 0.27, 0.44, 0.66)

# Names of the burn severity classes, by class number.
BURN_SEVERITY_CLASSES = (
    "enhanced regrowth, high",  # 0: dNBR < -0.25.
    "enhanced regrowth, low",  # 1: -0.25 <= dNBR < -0.1.
    "unburned",  # 2: -0.1 <= dNBR < 0.1.
    "low severity",  # 3: 0.1 <= dNBR < 0.27.
    "moderate-low severity",  # 4: 0.27 <= dNBR < 0.44.
    "moderate-high severity",  # 5: 0.44 <= dNBR < 0.66.
    "high severity",  # 6: dNBR >= 0.66.
)  # End of the class names.


# Thematic groups of indices.
class IndexCategory(str, Enum):
    # Vegetation vigour and chlorophyll.
    VEGETATION = "vegetation"
    # Open water and water quality.
    WATER = "water"
    # Bare soil.
    SOIL = "soil"
    # Burned areas.
    BURN = "burn"
    # Built-up areas.
    URBAN = "urban"
    # Snow and ice.
    SNOW = "snow"
    # Vegetation water content.
    MOISTURE = "moisture"


# Quotient with NaN where the denominator is zero.
def ratio(numerator: Array, denominator: Array) -> Array:
    # Output filled with NaN.
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    # Divide only where the denominator is non-zero.
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


# Normalised difference (a - b) / (a + b).
def normalized_difference(a: Array, b: Array) -> Array:
    # Ratio of the difference and the sum.
    return ratio(a - b, a + b)


# Definition of a spectral index.
@dataclass
class SpectralIndex:
    # Short name, for example "NDVI".
    name: str
    # Formula in terms of the common band names.
    formula: str
    # Thematic group.
    category: IndexCategory
    # Common names of the required bands.
    bands_required: list[str]
    # Implementation of the formula.
    function: IndexFunction = field(repr=False)
    # Range for reflectances in [0, 1]; infinite limits mean unbounded.
    value_range: tuple[float, float] = (-1.0, 1.0)
    # What the index measures.
    description: str = ""
    # Publication that defines the index.
    reference: str = ""
    # Default values of the formula parameters.
    parameters: dict[str, float] = field(default_factory=dict)

    # Compute the index from a dictionary of bands.
    def compute(
        self,  # This object.
        bands: Mapping[str, Any],  # Band arrays by common or product name.
        sensor: str | None = None,  # Sensor whose product band names are used.
        nodata: float | None = None,  # Input value that marks missing pixels.
        **parameters: float,  # Overrides of the formula parameters.
    ) -> Array:  # Index values, NaN where undefined.
        # Unknown parameters are an error.
        unknown = sorted(set(parameters) - set(self.parameters))
        # Report them.
        if unknown:
            # Explain the problem with the known parameters.
            raise ValueError(f"{self.name} has no parameters {unknown}; known: {self.parameters}")
        # Parameters in effect.
        values = {**self.parameters, **{k: float(v) for k, v in parameters.items()}}
        # Required bands as float64 arrays.
        arrays = resolve_bands(bands, self.bands_required, sensor)
        # Pixels where any input is missing.
        invalid = np.zeros(next(iter(arrays.values())).shape, dtype=bool)
        # Visit the inputs.
        for array in arrays.values():
            # NaN and infinite values.
            invalid |= ~np.isfinite(array)
            # The no-data value.
            if nodata is not None:
                # Pixels equal to it.
                invalid |= array == nodata
        # Evaluate the formula; division warnings are handled by ratio.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Index values.
            result = np.asarray(self.function(arrays, values), dtype=np.float64)
        # Mark invalid inputs and undefined results.
        result[invalid | ~np.isfinite(result)] = np.nan
        # Return the index.
        return result

    # Plain dictionary for listings.
    def to_dict(self) -> dict[str, Any]:
        # One entry per descriptive field.
        return {
            "name": self.name,  # Name.
            "formula": self.formula,  # Formula.
            "category": self.category.value,  # Group.
            "bands_required": list(self.bands_required),  # Bands.
            "value_range": list(self.value_range),  # Range.
            "description": self.description,  # Description.
            "reference": self.reference,  # Reference.
            "parameters": dict(self.parameters),  # Parameters.
        }  # End of the dictionary.


# Map the keys of a band dictionary to common names and return the required bands.
def resolve_bands(
    bands: Mapping[str, Any],  # Band arrays by common or product name.
    required: Iterable[str],  # Common names needed.
    sensor: str | None = None,  # Sensor whose product band names are used.
) -> dict[str, Array]:  # Required bands as float64 arrays.
    # Product name aliases of the sensor.
    aliases: dict[str, str] = {}
    # Look the sensor up when given.
    if sensor is not None:
        # Sensor definition.
        model = get_sensor(sensor)
        # Unknown sensors are an error.
        if model is None:
            # Explain the problem.
            raise ValueError(f"unknown sensor {sensor!r}")
        # Its aliases.
        aliases = model.band_aliases()
    # Bands by common name.
    common: dict[str, Any] = {}
    # Visit the given bands.
    for key, value in bands.items():
        # Upper-case key.
        upper = str(key).upper()
        # Sensor alias, then generic alias, then the key itself.
        name = aliases.get(upper, GENERIC_ALIASES.get(upper, upper))
        # Keep the first band given for each common name.
        common.setdefault(name, value)
    # Required names.
    names = list(required)
    # Names that are not available.
    missing = [n for n in names if n not in common]
    # Report missing bands.
    if missing:
        # Explain the problem with the available names.
        raise ValueError(f"Missing bands: {missing}; available: {sorted(common)}")
    # Required bands as float64 arrays.
    arrays = {n: np.asarray(common[n], dtype=np.float64) for n in names}
    # Shapes of the bands.
    shapes = {a.shape for a in arrays.values()}
    # All bands must have the same shape.
    if len(shapes) > 1:
        # Explain the problem.
        raise ValueError(f"bands have different shapes: {sorted(shapes)}")
    # Return the bands.
    return arrays


# Registry of spectral indices by case-insensitive name.
class IndexRegistry:
    # Indices by upper-case name.
    _indices: dict[str, SpectralIndex] = {}

    # Register an index; an existing index of the same name is replaced.
    @classmethod
    def register(cls, index: SpectralIndex) -> SpectralIndex:
        # Store under the upper-case name.
        cls._indices[index.name.upper()] = index
        # Return the index.
        return index

    # Remove an index.
    @classmethod
    def unregister(cls, name: str) -> None:
        # Remove when present.
        cls._indices.pop(name.upper(), None)

    # Index by name, or None when unknown.
    @classmethod
    def get(cls, name: str) -> SpectralIndex | None:
        # Case-insensitive look-up.
        return cls._indices.get(name.upper())

    # Names of the registered indices.
    @classmethod
    def list_all(cls) -> list[str]:
        # Names with their original case.
        return [index.name for index in cls._indices.values()]

    # Indices of one category.
    @classmethod
    def by_category(cls, category: IndexCategory | str) -> list[SpectralIndex]:
        # Normalise the category.
        wanted = IndexCategory(category)
        # Matching indices.
        return [index for index in cls._indices.values() if index.category is wanted]

    # Indices that can be computed from the given common band names.
    @classmethod
    def available_for(cls, band_names: Iterable[str]) -> list[str]:
        # Upper-case names.
        have = {b.upper() for b in band_names}
        # Indices whose bands are all present.
        return [i.name for i in cls._indices.values() if set(i.bands_required) <= have]


# Create and register an index.
def _define(
    name: str,  # Short name.
    formula: str,  # Formula text.
    category: IndexCategory,  # Group.
    bands: list[str],  # Required bands.
    function: IndexFunction,  # Implementation.
    reference: str,  # Short reference.
    description: str,  # What the index measures.
    value_range: tuple[float, float] = (-1.0, 1.0),  # Range for reflectances.
    parameters: dict[str, float] | None = None,  # Default parameters.
) -> SpectralIndex:  # The registered index.
    # Build and register.
    return IndexRegistry.register(
        SpectralIndex(  # The new index.
            name=name,  # Name.
            formula=formula,  # Formula.
            category=category,  # Group.
            bands_required=bands,  # Bands.
            function=function,  # Implementation.
            value_range=value_range,  # Range.
            description=description,  # Description.
            reference=reference,  # Reference.
            parameters=dict(parameters or {}),  # Parameters.
        )  # End of the index.
    )  # End of the registration.


# Shorter names for the categories used below.
_VEG, _WAT, _BURN = IndexCategory.VEGETATION, IndexCategory.WATER, IndexCategory.BURN


# EVI with the MODIS coefficients as parameters.
def _evi(b: Mapping[str, Array], p: Mapping[str, float]) -> Array:
    # Denominator with the aerosol resistance terms.
    den = b["NIR"] + p["C1"] * b["RED"] - p["C2"] * b["BLUE"] + p["L"]
    # Gain times the ratio.
    return p["G"] * ratio(b["NIR"] - b["RED"], den)


# MSAVI in the closed form of Qi et al. (1994).
def _msavi(b: Mapping[str, Array], p: Mapping[str, float]) -> Array:
    # 2 * NIR + 1.
    t = 2.0 * b["NIR"] + 1.0
    # Discriminant; negative values give NaN.
    disc = t * t - 8.0 * (b["NIR"] - b["RED"])
    # Solution of the self-adjusting soil line.
    return (t - np.sqrt(np.where(disc >= 0, disc, np.nan))) / 2.0


# ARVI with the red-blue combination RB = RED - gamma * (BLUE - RED).
def _arvi(b: Mapping[str, Array], p: Mapping[str, float]) -> Array:
    # Atmospherically corrected red.
    rb = b["RED"] - p["gamma"] * (b["BLUE"] - b["RED"])
    # Normalised difference with NIR.
    return normalized_difference(b["NIR"], rb)


# AWEI with shadow removal.
def _aweish(b: Mapping[str, Array], p: Mapping[str, float]) -> Array:
    # Water-bright visible bands.
    visible = b["BLUE"] + 2.5 * b["GREEN"]
    # Water-dark infrared bands.
    infrared = 1.5 * (b["NIR"] + b["SWIR1"]) + 0.25 * b["SWIR2"]
    # Difference.
    return visible - infrared


# Bare soil index.
def _bsi(b: Mapping[str, Array], p: Mapping[str, float]) -> Array:
    # Soil-bright bands.
    soil = b["SWIR1"] + b["RED"]
    # Vegetation-bright bands.
    veg = b["NIR"] + b["BLUE"]
    # Normalised difference.
    return normalized_difference(soil, veg)


# Vegetation indices.
NDVI = _define(  # Define and register NDVI.
    "NDVI",  # Normalised Difference Vegetation Index.
    "(NIR - RED) / (NIR + RED)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: normalized_difference(b["NIR"], b["RED"]),  # Implementation.
    "Rouse et al. (1974)",  # Reference.
    "Green vegetation vigour and density",  # Description.
)  # End of NDVI.
GNDVI = _define(  # Define and register GNDVI.
    "GNDVI",  # Green NDVI.
    "(NIR - GREEN) / (NIR + GREEN)",  # Formula.
    _VEG,  # Group.
    ["NIR", "GREEN"],  # Bands.
    lambda b, p: normalized_difference(b["NIR"], b["GREEN"]),  # Implementation.
    "Gitelson et al. (1996)",  # Reference.
    "Chlorophyll concentration; saturates later than NDVI",  # Description.
)  # End of GNDVI.
NDRE = _define(  # Define and register NDRE.
    "NDRE",  # Normalised Difference Red Edge.
    "(NIR - REDEDGE1) / (NIR + REDEDGE1)",  # Formula.
    _VEG,  # Group.
    ["NIR", "REDEDGE1"],  # Bands.
    lambda b, p: normalized_difference(b["NIR"], b["REDEDGE1"]),  # Implementation.
    "Gitelson and Merzlyak (1994)",  # Reference.
    "Chlorophyll content of dense canopies",  # Description.
)  # End of NDRE.
EVI = _define(  # Define and register EVI.
    "EVI",  # Enhanced Vegetation Index.
    "G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED", "BLUE"],  # Bands.
    _evi,  # Implementation.
    "Huete et al. (2002)",  # Reference.
    "Vegetation with reduced soil and aerosol influence",  # Description.
    (-INF, INF),  # The denominator can vanish.
    {"G": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0},  # MODIS coefficients.
)  # End of EVI.
EVI2 = _define(  # Define and register EVI2.
    "EVI2",  # Two-band EVI.
    "2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: 2.5 * ratio(b["NIR"] - b["RED"], b["NIR"] + 2.4 * b["RED"] + 1.0),  # Formula.
    "Jiang et al. (2008)",  # Reference.
    "EVI without the blue band",  # Description.
    (-2.5 / 3.4, 1.25),  # Extremes at (NIR, RED) = (0, 1) and (1, 0).
)  # End of EVI2.
SAVI = _define(  # Define and register SAVI.
    "SAVI",  # Soil Adjusted Vegetation Index.
    "(1 + L) * (NIR - RED) / (NIR + RED + L)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: (1 + p["L"]) * ratio(b["NIR"] - b["RED"], b["NIR"] + b["RED"] + p["L"]),  # SAVI.
    "Huete (1988)",  # Reference.
    "Vegetation with soil brightness correction",  # Description.
    (-1.0, 1.0),  # Range for reflectances.
    {"L": 0.5},  # Soil adjustment for intermediate cover.
)  # End of SAVI.
MSAVI = _define(  # Define and register MSAVI.
    "MSAVI",  # Modified SAVI (MSAVI2).
    "(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    _msavi,  # Implementation.
    "Qi et al. (1994)",  # Reference.
    "SAVI with a self-adjusting soil factor",  # Description.
)  # End of MSAVI.
OSAVI = _define(  # Define and register OSAVI.
    "OSAVI",  # Optimised SAVI.
    "(NIR - RED) / (NIR + RED + 0.16)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: ratio(b["NIR"] - b["RED"], b["NIR"] + b["RED"] + 0.16),  # Implementation.
    "Rondeaux et al. (1996)",  # Reference.
    "SAVI with the soil factor 0.16 optimised for agriculture",  # Description.
    (-1.0 / 1.16, 1.0 / 1.16),  # Extremes at (NIR, RED) = (0, 1) and (1, 0).
)  # End of OSAVI.
ARVI = _define(  # Define and register ARVI.
    "ARVI",  # Atmospherically Resistant Vegetation Index.
    "(NIR - RB) / (NIR + RB), RB = RED - gamma * (BLUE - RED)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED", "BLUE"],  # Bands.
    _arvi,  # Implementation.
    "Kaufman and Tanre (1992)",  # Reference.
    "Vegetation with self-correction of aerosol effects",  # Description.
    (-INF, INF),  # RB can be negative.
    {"gamma": 1.0},  # Recommended value.
)  # End of ARVI.
VARI = _define(  # Define and register VARI.
    "VARI",  # Visible Atmospherically Resistant Index.
    "(GREEN - RED) / (GREEN + RED - BLUE)",  # Formula.
    _VEG,  # Group.
    ["GREEN", "RED", "BLUE"],  # Bands.
    lambda b, p: ratio(b["GREEN"] - b["RED"], b["GREEN"] + b["RED"] - b["BLUE"]),  # VARI.
    "Gitelson et al. (2002)",  # Reference.
    "Vegetation fraction from visible bands only",  # Description.
    (-INF, INF),  # The denominator can vanish.
)  # End of VARI.
SR = _define(  # Define and register SR.
    "SR",  # Simple Ratio.
    "NIR / RED",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: ratio(b["NIR"], b["RED"]),  # Implementation.
    "Jordan (1969)",  # Reference.
    "Leaf area index and biomass",  # Description.
    (0.0, INF),  # Unbounded above.
)  # End of SR.
WDRVI = _define(  # Define and register WDRVI.
    "WDRVI",  # Wide Dynamic Range Vegetation Index.
    "(alpha * NIR - RED) / (alpha * NIR + RED)",  # Formula.
    _VEG,  # Group.
    ["NIR", "RED"],  # Bands.
    lambda b, p: normalized_difference(p["alpha"] * b["NIR"], b["RED"]),  # Implementation.
    "Gitelson (2004)",  # Reference.
    "Vegetation fraction of dense canopies where NDVI saturates",  # Description.
    (-1.0, 1.0),  # Range for reflectances.
    {"alpha": 0.1},  # Weighting coefficient.
)  # End of WDRVI.
CIGREEN = _define(  # Define and register CIGREEN.
    "CIgreen",  # Green Chlorophyll Index.
    "NIR / GREEN - 1",  # Formula.
    _VEG,  # Group.
    ["NIR", "GREEN"],  # Bands.
    lambda b, p: ratio(b["NIR"], b["GREEN"]) - 1.0,  # Implementation.
    "Gitelson et al. (2003)",  # Reference.
    "Leaf chlorophyll content",  # Description.
    (-1.0, INF),  # Unbounded above.
)  # End of CIgreen.
CIRE = _define(  # Define and register CIRE.
    "CIre",  # Red Edge Chlorophyll Index.
    "NIR / REDEDGE1 - 1",  # Formula.
    _VEG,  # Group.
    ["NIR", "REDEDGE1"],  # Bands.
    lambda b, p: ratio(b["NIR"], b["REDEDGE1"]) - 1.0,  # Implementation.
    "Gitelson et al. (2003)",  # Reference.
    "Leaf chlorophyll content from the red edge",  # Description.
    (-1.0, INF),  # Unbounded above.
)  # End of CIre.

# Water indices.
NDWI = _define(  # Define and register NDWI.
    "NDWI",  # Normalised Difference Water Index.
    "(GREEN - NIR) / (GREEN + NIR)",  # Formula.
    _WAT,  # Group.
    ["GREEN", "NIR"],  # Bands.
    lambda b, p: normalized_difference(b["GREEN"], b["NIR"]),  # Implementation.
    "McFeeters (1996)",  # Reference.
    "Open water; positive over water",  # Description.
)  # End of NDWI.
MNDWI = _define(  # Define and register MNDWI.
    "MNDWI",  # Modified NDWI.
    "(GREEN - SWIR1) / (GREEN + SWIR1)",  # Formula.
    _WAT,  # Group.
    ["GREEN", "SWIR1"],  # Bands.
    lambda b, p: normalized_difference(b["GREEN"], b["SWIR1"]),  # Implementation.
    "Xu (2006)",  # Reference.
    "Open water with suppression of built-up areas",  # Description.
)  # End of MNDWI.
NDMI = _define(  # Define and register NDMI.
    "NDMI",  # Normalised Difference Moisture Index.
    "(NIR - SWIR1) / (NIR + SWIR1)",  # Formula.
    IndexCategory.MOISTURE,  # Group.
    ["NIR", "SWIR1"],  # Bands.
    lambda b, p: normalized_difference(b["NIR"], b["SWIR1"]),  # Implementation.
    "Gao (1996)",  # Reference.
    "Vegetation liquid water content",  # Description.
)  # End of NDMI.
AWEINSH = _define(  # Define and register AWEINSH.
    "AWEInsh",  # Automated Water Extraction Index, no shadow.
    "4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)",  # Formula.
    _WAT,  # Group.
    ["GREEN", "SWIR1", "NIR", "SWIR2"],  # Bands.
    lambda b, p: 4.0 * (b["GREEN"] - b["SWIR1"]) - (0.25 * b["NIR"] + 2.75 * b["SWIR2"]),  # AWEI.
    "Feyisa et al. (2014)",  # Reference.
    "Water in scenes without strong shadows; water > 0",  # Description.
    (-7.0, 4.0),  # Extremes for reflectances in [0, 1].
)  # End of AWEInsh.
AWEISH = _define(  # Define and register AWEISH.
    "AWEIsh",  # Automated Water Extraction Index, shadow.
    "BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2",  # Formula.
    _WAT,  # Group.
    ["BLUE", "GREEN", "NIR", "SWIR1", "SWIR2"],  # Bands.
    _aweish,  # Implementation.
    "Feyisa et al. (2014)",  # Reference.
    "Water in scenes with shadows and dark surfaces; water > 0",  # Description.
    (-3.25, 3.5),  # Extremes for reflectances in [0, 1].
)  # End of AWEIsh.
NDTI = _define(  # Define and register NDTI.
    "NDTI",  # Normalised Difference Turbidity Index.
    "(RED - GREEN) / (RED + GREEN)",  # Formula.
    _WAT,  # Group.
    ["RED", "GREEN"],  # Bands.
    lambda b, p: normalized_difference(b["RED"], b["GREEN"]),  # Implementation.
    "Lacaux et al. (2007)",  # Reference.
    "Turbidity of water bodies",  # Description.
)  # End of NDTI.
NDCI = _define(  # Define and register NDCI.
    "NDCI",  # Normalised Difference Chlorophyll Index.
    "(REDEDGE1 - RED) / (REDEDGE1 + RED)",  # Formula.
    _WAT,  # Group.
    ["REDEDGE1", "RED"],  # Bands.
    lambda b, p: normalized_difference(b["REDEDGE1"], b["RED"]),  # Implementation.
    "Mishra and Mishra (2012)",  # Reference.
    "Chlorophyll-a in turbid productive waters",  # Description.
)  # End of NDCI.

# Burn indices.
NBR = _define(  # Define and register NBR.
    "NBR",  # Normalised Burn Ratio.
    "(NIR - SWIR2) / (NIR + SWIR2)",  # Formula.
    _BURN,  # Group.
    ["NIR", "SWIR2"],  # Bands.
    lambda b, p: normalized_difference(b["NIR"], b["SWIR2"]),  # Implementation.
    "Key and Benson (2006)",  # Reference.
    "Burned areas; low after fire",  # Description.
)  # End of NBR.
NBR2 = _define(  # Define and register NBR2.
    "NBR2",  # Normalised Burn Ratio 2.
    "(SWIR1 - SWIR2) / (SWIR1 + SWIR2)",  # Formula.
    _BURN,  # Group.
    ["SWIR1", "SWIR2"],  # Bands.
    lambda b, p: normalized_difference(b["SWIR1"], b["SWIR2"]),  # Implementation.
    "USGS Landsat spectral indices product guide",  # Reference.
    "Post-fire recovery and vegetation water content",  # Description.
)  # End of NBR2.

# Urban, soil, snow and moisture indices.
NDBI = _define(  # Define and register NDBI.
    "NDBI",  # Normalised Difference Built-up Index.
    "(SWIR1 - NIR) / (SWIR1 + NIR)",  # Formula.
    IndexCategory.URBAN,  # Group.
    ["SWIR1", "NIR"],  # Bands.
    lambda b, p: normalized_difference(b["SWIR1"], b["NIR"]),  # Implementation.
    "Zha et al. (2003)",  # Reference.
    "Built-up areas; positive over built-up land",  # Description.
)  # End of NDBI.
BSI = _define(  # Define and register BSI.
    "BSI",  # Bare Soil Index.
    "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",  # Formula.
    IndexCategory.SOIL,  # Group.
    ["SWIR1", "RED", "NIR", "BLUE"],  # Bands.
    _bsi,  # Implementation.
    "Rikimaru et al. (2002)",  # Reference.
    "Bare soil against vegetation",  # Description.
)  # End of BSI.
NDSI = _define(  # Define and register NDSI.
    "NDSI",  # Normalised Difference Snow Index.
    "(GREEN - SWIR1) / (GREEN + SWIR1)",  # Formula.
    IndexCategory.SNOW,  # Group.
    ["GREEN", "SWIR1"],  # Bands.
    lambda b, p: normalized_difference(b["GREEN"], b["SWIR1"]),  # Implementation.
    "Hall et al. (1995)",  # Reference.
    "Snow cover; snow typically above 0.4",  # Description.
)  # End of NDSI.
MSI = _define(  # Define and register MSI.
    "MSI",  # Moisture Stress Index.
    "SWIR1 / NIR",  # Formula.
    IndexCategory.MOISTURE,  # Group.
    ["SWIR1", "NIR"],  # Bands.
    lambda b, p: ratio(b["SWIR1"], b["NIR"]),  # Implementation.
    "Hunt and Rock (1989)",  # Reference.
    "Plant water stress; higher values mean drier vegetation",  # Description.
    (0.0, INF),  # Unbounded above.
)  # End of MSI.


# Compute a registered index by name.
def compute_index(
    name: str,  # Index name, case-insensitive.
    bands: Mapping[str, Any],  # Band arrays.
    sensor: str | None = None,  # Sensor whose product band names are used.
    nodata: float | None = None,  # Input value that marks missing pixels.
    **parameters: float,  # Overrides of the formula parameters.
) -> Array:  # Index values.
    # Look the index up.
    index = IndexRegistry.get(name)
    # Unknown names are an error.
    if index is None:
        # Explain the problem with the known names.
        raise ValueError(f"Unknown index: {name}; available: {', '.join(IndexRegistry.list_all())}")
    # Compute.
    return index.compute(bands, sensor=sensor, nodata=nodata, **parameters)


# Compute several indices from the same bands.
def compute_indices(
    names: Iterable[str],  # Index names.
    bands: Mapping[str, Any],  # Band arrays.
    sensor: str | None = None,  # Sensor whose product band names are used.
    nodata: float | None = None,  # Input value that marks missing pixels.
) -> dict[str, Array]:  # Index values by name.
    # One entry per name.
    return {n: compute_index(n, bands, sensor=sensor, nodata=nodata) for n in names}


# Change of an index between two dates: index(pre) - index(post).
def difference_index(
    name: str,  # Index name.
    pre: Mapping[str, Any],  # Bands before the event.
    post: Mapping[str, Any],  # Bands after the event.
    sensor: str | None = None,  # Sensor whose product band names are used.
    nodata: float | None = None,  # Input value that marks missing pixels.
) -> Array:  # Differenced index.
    # Index before the event.
    before = compute_index(name, pre, sensor=sensor, nodata=nodata)
    # Index after the event.
    after = compute_index(name, post, sensor=sensor, nodata=nodata)
    # The two dates must share the grid.
    if before.shape != after.shape:
        # Explain the problem.
        raise ValueError(f"pre and post shapes differ: {before.shape} vs {after.shape}")
    # Difference.
    return before - after


# Differenced Normalised Burn Ratio, NBR(pre) - NBR(post).
def dnbr(
    pre: Mapping[str, Any],  # Bands before the fire.
    post: Mapping[str, Any],  # Bands after the fire.
    sensor: str | None = None,  # Sensor whose product band names are used.
    nodata: float | None = None,  # Input value that marks missing pixels.
) -> Array:  # dNBR, positive where vegetation burned.
    # Difference of NBR.
    return difference_index("NBR", pre, post, sensor=sensor, nodata=nodata)


# Burn severity classes 0 to 6 of dNBR; 255 where dNBR is NaN.
def classify_burn_severity(dnbr_values: Any) -> NDArray[np.uint8]:
    # Values as float64.
    values = np.asarray(dnbr_values, dtype=np.float64)
    # Class number: count of limits at or below each value.
    classes = np.digitize(values, BURN_SEVERITY_LIMITS).astype(np.uint8)
    # Mark missing values.
    classes[~np.isfinite(values)] = 255
    # Return the classes.
    return classes


# =============================================================================
# End of module src/unbihexium/core/index.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
