# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/indices/spectral.py
# Title       : Spectral and radar indices as array functions
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Band arithmetic indices for optical and radar imagery. Each function takes
# band arrays (surface reflectance in [0, 1] for the optical indices, linear
# backscatter for the radar ones) and returns a float64 array of the
# broadcast shape. Ratios return NaN where the denominator is zero or an
# input is NaN, so that no artificial values enter later statistics.
#
#   vegetation  ndvi, gndvi, ndre, evi, evi2, savi, osavi, msavi, arvi,
#               vari, kndvi, ci_green, ci_rededge
#   water       ndwi, mndwi, ndmi, awei_nsh, awei_sh
#   built-up    ndbi, bsi
#   snow        ndsi
#   fire        nbr, nbr2, dnbr, rdnbr, burn_severity
#   moisture    msi
#   radar       rvi (quad-pol radar vegetation index), cross_pol_ratio
#
# compute_index(name, **bands) evaluates an index by its name.
#
# References
# ----------
# Rouse, J. W., Haas, R. H., Schell, J. A., Deering, D. W. (1974).
#   Monitoring vegetation systems in the Great Plains with ERTS. NASA
#   SP-351, 309-317.
# Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). Remote
#   Sensing of Environment, 25(3), 295-309.
# Kaufman, Y. J., Tanre, D. (1992). Atmospherically resistant vegetation
#   index (ARVI) for EOS-MODIS. IEEE Transactions on Geoscience and Remote
#   Sensing, 30(2), 261-270.
# Qi, J., Chehbouni, A., Huete, A. R., Kerr, Y. H., Sorooshian, S. (1994). A
#   modified soil adjusted vegetation index. Remote Sensing of Environment,
#   48(2), 119-126.
# Rondeaux, G., Steven, M., Baret, F. (1996). Optimization of
#   soil-adjusted vegetation indices. Remote Sensing of Environment, 55(2),
#   95-107.
# Gitelson, A. A., Kaufman, Y. J., Merzlyak, M. N. (1996). Use of a green
#   channel in remote sensing of global vegetation from EOS-MODIS. Remote
#   Sensing of Environment, 58(3), 289-298.
# Barnes, E. M., et al. (2000). Coincident detection of crop water stress,
#   nitrogen status and canopy density using ground-based multispectral
#   data. Proc. 5th International Conference on Precision Agriculture.
# Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., Ferreira,
#   L. G. (2002). Overview of the radiometric and biophysical performance of
#   the MODIS vegetation indices. Remote Sensing of Environment, 83(1-2),
#   195-213.
# Gitelson, A. A., Stark, R., Grits, U., Rundquist, D., Kaufman, Y.,
#   Derry, D. (2002). Vegetation and soil lines in visible spectral space.
#   International Journal of Remote Sensing, 23(13), 2537-2562.
# Gitelson, A. A., Gritz, Y., Merzlyak, M. N. (2003). Relationships between
#   leaf chlorophyll content and spectral reflectance. Journal of Plant
#   Physiology, 160(3), 271-282.
# Jiang, Z., Huete, A. R., Didan, K., Miura, T. (2008). Development of a
#   two-band enhanced vegetation index without a blue band. Remote Sensing
#   of Environment, 112(10), 3833-3845.
# Camps-Valls, G., et al. (2021). A unified vegetation index for
#   quantifying the terrestrial biosphere. Science Advances, 7(9), eabc7447.
# McFeeters, S. K. (1996). The use of the normalized difference water index
#   (NDWI) in the delineation of open water features. International Journal
#   of Remote Sensing, 17(7), 1425-1432.
# Gao, B.-C. (1996). NDWI: a normalized difference water index for remote
#   sensing of vegetation liquid water from space. Remote Sensing of
#   Environment, 58(3), 257-266.
# Xu, H. (2006). Modification of normalised difference water index (NDWI)
#   to enhance open water features in remotely sensed imagery.
#   International Journal of Remote Sensing, 27(14), 3025-3033.
# Feyisa, G. L., Meilby, H., Fensholt, R., Proud, S. R. (2014). Automated
#   water extraction index. Remote Sensing of Environment, 140, 23-35.
# Zha, Y., Gao, J., Ni, S. (2003). Use of normalized difference built-up
#   index in automatically mapping urban areas from TM imagery.
#   International Journal of Remote Sensing, 24(3), 583-594.
# Rikimaru, A., Roy, P. S., Miyatake, S. (2002). Tropical forest cover
#   density mapping. Tropical Ecology, 43(1), 39-47.
# Hall, D. K., Riggs, G. A., Salomonson, V. V. (1995). Development of
#   methods for mapping global snow cover using MODIS data. Remote Sensing
#   of Environment, 54(2), 127-140.
# Key, C. H., Benson, N. C. (2006). Landscape assessment: ground measure of
#   severity, the Composite Burn Index, and remote sensing of severity, the
#   Normalized Burn Ratio. USDA Forest Service RMRS-GTR-164-CD.
# Miller, J. D., Thode, A. E. (2007). Quantifying burn severity in a
#   heterogeneous landscape with a relative version of the delta Normalized
#   Burn Ratio (dNBR). Remote Sensing of Environment, 109(1), 66-80.
# Hunt, E. R., Rock, B. N. (1989). Detection of changes in leaf water
#   content using near- and middle-infrared reflectances. Remote Sensing of
#   Environment, 30(1), 43-54.
# Kim, Y., van Zyl, J. J. (2009). A time-series approach to estimate soil
#   moisture using polarimetric radar data. IEEE Transactions on Geoscience
#   and Remote Sensing, 47(8), 2519-2527.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of callables in the index table.
from collections.abc import Callable

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Array type annotations.
from numpy.typing import NDArray

# Array-like input: arrays or scalars.
Band = NDArray[Any] | float


# Float64 view of a band.
def _f(band: Band) -> NDArray[np.float64]:
    # Convert to float64.
    return np.asarray(band, dtype=np.float64)


# Ratio num / den with NaN where the denominator is zero or not finite.
def safe_divide(num: Band, den: Band) -> NDArray[np.float64]:
    # Operands as float64 arrays, broadcast together.
    n, d = np.broadcast_arrays(_f(num), _f(den))
    # Only divide by finite, non-zero denominators.
    ok = np.isfinite(d) & (d != 0)
    # NaN elsewhere.
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=ok)


# Normalised difference (a - b) / (a + b).
def normalized_difference(a: Band, b: Band) -> NDArray[np.float64]:
    # Operands as float64.
    x, y = _f(a), _f(b)
    # Ratio of difference and sum.
    return safe_divide(x - y, x + y)


# Normalized Difference Vegetation Index (Rouse et al., 1974).
def ndvi(nir: Band, red: Band) -> NDArray[np.float64]:
    # (NIR - Red) / (NIR + Red).
    return normalized_difference(nir, red)


# Green NDVI (Gitelson et al., 1996).
def gndvi(nir: Band, green: Band) -> NDArray[np.float64]:
    # (NIR - Green) / (NIR + Green).
    return normalized_difference(nir, green)


# Normalized Difference Red Edge index (Barnes et al., 2000).
def ndre(nir: Band, red_edge: Band) -> NDArray[np.float64]:
    # (NIR - RedEdge) / (NIR + RedEdge).
    return normalized_difference(nir, red_edge)


# Enhanced Vegetation Index (Huete et al., 2002).
def evi(
    nir: Band,  # Near-infrared reflectance.
    red: Band,  # Red reflectance.
    blue: Band,  # Blue reflectance.
    g: float = 2.5,  # Gain factor.
    c1: float = 6.0,  # Aerosol coefficient of the red band.
    c2: float = 7.5,  # Aerosol coefficient of the blue band.
    l: float = 1.0,  # Canopy background adjustment.
) -> NDArray[np.float64]:  # EVI.
    # G (NIR - Red) / (NIR + C1 Red - C2 Blue + L).
    return safe_divide(g * (_f(nir) - _f(red)), _f(nir) + c1 * _f(red) - c2 * _f(blue) + l)


# Two-band EVI without the blue band (Jiang et al., 2008).
def evi2(nir: Band, red: Band) -> NDArray[np.float64]:
    # 2.5 (NIR - Red) / (NIR + 2.4 Red + 1).
    return safe_divide(2.5 * (_f(nir) - _f(red)), _f(nir) + 2.4 * _f(red) + 1.0)


# Soil Adjusted Vegetation Index (Huete, 1988).
def savi(nir: Band, red: Band, l: float = 0.5) -> NDArray[np.float64]:
    # (1 + L) (NIR - Red) / (NIR + Red + L).
    return safe_divide((1.0 + l) * (_f(nir) - _f(red)), _f(nir) + _f(red) + l)


# Optimised SAVI (Rondeaux et al., 1996).
def osavi(nir: Band, red: Band) -> NDArray[np.float64]:
    # (NIR - Red) / (NIR + Red + 0.16).
    return safe_divide(_f(nir) - _f(red), _f(nir) + _f(red) + 0.16)


# Modified SAVI, MSAVI2 (Qi et al., 1994).
def msavi(nir: Band, red: Band) -> NDArray[np.float64]:
    # Shorthand for 2 NIR + 1.
    a = 2.0 * _f(nir) + 1.0
    # (2 NIR + 1 - sqrt((2 NIR + 1)^2 - 8 (NIR - Red))) / 2; NaN for a negative radicand.
    radicand = a * a - 8.0 * (_f(nir) - _f(red))
    # Square root of non-negative values only.
    root = np.sqrt(np.where(radicand >= 0, radicand, np.nan))
    # Final index.
    return (a - root) / 2.0


# Atmospherically Resistant Vegetation Index (Kaufman and Tanre, 1992).
def arvi(nir: Band, red: Band, blue: Band, gamma: float = 1.0) -> NDArray[np.float64]:
    # Atmospherically corrected red: Red - gamma (Blue - Red).
    rb = _f(red) - gamma * (_f(blue) - _f(red))
    # (NIR - RB) / (NIR + RB).
    return normalized_difference(nir, rb)


# Visible Atmospherically Resistant Index (Gitelson et al., 2002).
def vari(green: Band, red: Band, blue: Band) -> NDArray[np.float64]:
    # (Green - Red) / (Green + Red - Blue).
    return safe_divide(_f(green) - _f(red), _f(green) + _f(red) - _f(blue))


# Kernel NDVI with the RBF kernel and sigma = (NIR + Red) / 2 (Camps-Valls et al., 2021).
def kndvi(nir: Band, red: Band) -> NDArray[np.float64]:
    # With this sigma, kNDVI = tanh(NDVI^2).
    return np.tanh(ndvi(nir, red) ** 2)


# Green chlorophyll index (Gitelson et al., 2003).
def ci_green(nir: Band, green: Band) -> NDArray[np.float64]:
    # NIR / Green - 1.
    return safe_divide(nir, green) - 1.0


# Red-edge chlorophyll index (Gitelson et al., 2003).
def ci_rededge(nir: Band, red_edge: Band) -> NDArray[np.float64]:
    # NIR / RedEdge - 1.
    return safe_divide(nir, red_edge) - 1.0


# Normalized Difference Water Index for open water (McFeeters, 1996).
def ndwi(green: Band, nir: Band) -> NDArray[np.float64]:
    # (Green - NIR) / (Green + NIR).
    return normalized_difference(green, nir)


# Modified NDWI (Xu, 2006).
def mndwi(green: Band, swir1: Band) -> NDArray[np.float64]:
    # (Green - SWIR1) / (Green + SWIR1).
    return normalized_difference(green, swir1)


# Normalized Difference Moisture Index, the NDWI of Gao (1996).
def ndmi(nir: Band, swir1: Band) -> NDArray[np.float64]:
    # (NIR - SWIR1) / (NIR + SWIR1).
    return normalized_difference(nir, swir1)


# Automated Water Extraction Index without shadows (Feyisa et al., 2014).
def awei_nsh(green: Band, nir: Band, swir1: Band, swir2: Band) -> NDArray[np.float64]:
    # 4 (Green - SWIR1) - (0.25 NIR + 2.75 SWIR2).
    return 4.0 * (_f(green) - _f(swir1)) - (0.25 * _f(nir) + 2.75 * _f(swir2))


# Automated Water Extraction Index with shadows (Feyisa et al., 2014).
def awei_sh(blue: Band, green: Band, nir: Band, swir1: Band, swir2: Band) -> NDArray[np.float64]:
    # Blue + 2.5 Green - 1.5 (NIR + SWIR1) - 0.25 SWIR2.
    return _f(blue) + 2.5 * _f(green) - 1.5 * (_f(nir) + _f(swir1)) - 0.25 * _f(swir2)


# Normalized Difference Built-up Index (Zha et al., 2003).
def ndbi(swir1: Band, nir: Band) -> NDArray[np.float64]:
    # (SWIR1 - NIR) / (SWIR1 + NIR).
    return normalized_difference(swir1, nir)


# Bare Soil Index (Rikimaru et al., 2002).
def bsi(blue: Band, red: Band, nir: Band, swir1: Band) -> NDArray[np.float64]:
    # ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).
    return normalized_difference(_f(swir1) + _f(red), _f(nir) + _f(blue))


# Normalized Difference Snow Index (Hall et al., 1995).
def ndsi(green: Band, swir1: Band) -> NDArray[np.float64]:
    # (Green - SWIR1) / (Green + SWIR1).
    return normalized_difference(green, swir1)


# Normalized Burn Ratio with the long SWIR band (Key and Benson, 2006).
def nbr(nir: Band, swir: Band) -> NDArray[np.float64]:
    # (NIR - SWIR2) / (NIR + SWIR2).
    return normalized_difference(nir, swir)


# Normalized Burn Ratio 2 from the two SWIR bands.
def nbr2(swir1: Band, swir2: Band) -> NDArray[np.float64]:
    # (SWIR1 - SWIR2) / (SWIR1 + SWIR2).
    return normalized_difference(swir1, swir2)


# Differenced NBR, pre-fire minus post-fire (Key and Benson, 2006).
def dnbr(nbr_pre: Band, nbr_post: Band) -> NDArray[np.float64]:
    # Positive values indicate burning.
    return _f(nbr_pre) - _f(nbr_post)


# Relative dNBR, dNBR / sqrt(|NBR_pre|) (Miller and Thode, 2007).
def rdnbr(nbr_pre: Band, nbr_post: Band) -> NDArray[np.float64]:
    # Unitless form; multiply by 1000 for the scale of Miller and Thode.
    return safe_divide(dnbr(nbr_pre, nbr_post), np.sqrt(np.abs(_f(nbr_pre))))


# Lower dNBR limits of the burn severity classes of Key and Benson (2006).
BURN_SEVERITY_BREAKS = (-0.25, -0.1, 0.1, 0.27, 0.44, 0.66)

# Names of the classes 0 to 6 returned by burn_severity.
BURN_SEVERITY_CLASSES = (
    "enhanced regrowth, high",  # dNBR < -0.25.
    "enhanced regrowth, low",  # -0.25 <= dNBR < -0.1.
    "unburned",  # -0.1 <= dNBR < 0.1.
    "low severity",  # 0.1 <= dNBR < 0.27.
    "moderate-low severity",  # 0.27 <= dNBR < 0.44.
    "moderate-high severity",  # 0.44 <= dNBR < 0.66.
    "high severity",  # dNBR >= 0.66.
)  # End of the class names.


# Burn severity class 0 to 6 from dNBR, -1 for NaN.
def burn_severity(dnbr_values: Band) -> NDArray[np.int64]:
    # dNBR as float64.
    d = _f(dnbr_values)
    # Class index from the breaks.
    classes = np.digitize(d, BURN_SEVERITY_BREAKS, right=False).astype(np.int64)
    # Undefined pixels.
    return np.where(np.isfinite(d), classes, -1)


# Moisture Stress Index (Hunt and Rock, 1989).
def msi(swir: Band, nir: Band) -> NDArray[np.float64]:
    # SWIR1 / NIR; higher values mean drier vegetation.
    return safe_divide(swir, nir)


# Radar Vegetation Index of quad-pol backscatter (Kim and van Zyl, 2009).
def rvi(sigma_hh: Band, sigma_hv: Band, sigma_vv: Band) -> NDArray[np.float64]:
    # 8 HV / (HH + VV + 2 HV), linear backscatter.
    return safe_divide(8.0 * _f(sigma_hv), _f(sigma_hh) + _f(sigma_vv) + 2.0 * _f(sigma_hv))


# Cross-polarisation ratio of linear backscatter, for example VH / VV.
def cross_pol_ratio(sigma_cross: Band, sigma_co: Band) -> NDArray[np.float64]:
    # Linear ratio.
    return safe_divide(sigma_cross, sigma_co)


# Index functions by lower-case name.
INDEX_FUNCTIONS: dict[str, Callable[..., NDArray[Any]]] = {
    "ndvi": ndvi,  # Vegetation.
    "gndvi": gndvi,  # Green vegetation.
    "ndre": ndre,  # Red edge.
    "evi": evi,  # Enhanced vegetation.
    "evi2": evi2,  # Two-band EVI.
    "savi": savi,  # Soil adjusted.
    "osavi": osavi,  # Optimised soil adjusted.
    "msavi": msavi,  # Modified soil adjusted.
    "arvi": arvi,  # Atmospherically resistant.
    "vari": vari,  # Visible atmospherically resistant.
    "kndvi": kndvi,  # Kernel NDVI.
    "ci_green": ci_green,  # Green chlorophyll.
    "ci_rededge": ci_rededge,  # Red-edge chlorophyll.
    "ndwi": ndwi,  # Open water.
    "mndwi": mndwi,  # Modified water.
    "ndmi": ndmi,  # Moisture.
    "awei_nsh": awei_nsh,  # Water extraction, no shadow.
    "awei_sh": awei_sh,  # Water extraction, shadow.
    "ndbi": ndbi,  # Built-up.
    "bsi": bsi,  # Bare soil.
    "ndsi": ndsi,  # Snow.
    "nbr": nbr,  # Burn ratio.
    "nbr2": nbr2,  # Burn ratio 2.
    "msi": msi,  # Moisture stress.
    "rvi": rvi,  # Radar vegetation.
}  # End of the index table.


# Evaluate an index by name with bands passed as keyword arguments.
def compute_index(name: str, **bands: Band) -> NDArray[Any]:
    # Look up the function.
    function = INDEX_FUNCTIONS.get(name.lower())
    # Unknown index.
    if function is None:
        # Report the known names.
        raise ValueError(f"unknown index {name!r}; known: {sorted(INDEX_FUNCTIONS)}")
    # Call with the named bands.
    try:
        # Evaluate the index.
        return function(**bands)
    # Missing or unexpected band names.
    except TypeError as exc:
        # Report them as a value problem.
        raise ValueError(f"bad bands for index {name!r}: {exc}") from exc


# =============================================================================
# End of module src/unbihexium/indices/spectral.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
