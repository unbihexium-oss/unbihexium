# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/indices/__init__.py
# Title       : Spectral and radar indices
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Array functions for band-ratio indices of optical and radar imagery
# (vegetation, water, built-up, snow, fire and radar vegetation indices).
# The functions live in unbihexium.indices.spectral, which lists the
# formulas and references. They work on plain NumPy arrays and return NaN
# where a ratio is undefined.
# =============================================================================

# Index functions.
from unbihexium.indices.spectral import (
    BURN_SEVERITY_BREAKS,  # dNBR class limits.
    BURN_SEVERITY_CLASSES,  # dNBR class names.
    INDEX_FUNCTIONS,  # Index functions by name.
    arvi,  # Atmospherically resistant vegetation index.
    awei_nsh,  # Water extraction index, no shadow.
    awei_sh,  # Water extraction index, shadow.
    bsi,  # Bare soil index.
    burn_severity,  # dNBR severity classes.
    ci_green,  # Green chlorophyll index.
    ci_rededge,  # Red-edge chlorophyll index.
    compute_index,  # Index by name.
    cross_pol_ratio,  # Radar cross-polarisation ratio.
    dnbr,  # Differenced burn ratio.
    evi,  # Enhanced vegetation index.
    evi2,  # Two-band EVI.
    gndvi,  # Green NDVI.
    kndvi,  # Kernel NDVI.
    mndwi,  # Modified water index.
    msavi,  # Modified soil adjusted vegetation index.
    msi,  # Moisture stress index.
    nbr,  # Normalized burn ratio.
    nbr2,  # Normalized burn ratio 2.
    ndbi,  # Built-up index.
    ndmi,  # Moisture index.
    ndre,  # Red-edge index.
    ndsi,  # Snow index.
    ndvi,  # Vegetation index.
    ndwi,  # Water index.
    normalized_difference,  # Generic normalised difference.
    osavi,  # Optimised soil adjusted vegetation index.
    rdnbr,  # Relative differenced burn ratio.
    rvi,  # Radar vegetation index.
    safe_divide,  # Division with NaN for zero denominators.
    savi,  # Soil adjusted vegetation index.
    vari,  # Visible atmospherically resistant index.
)  # End of the index imports.

# Public names of the package.
__all__ = [
    "BURN_SEVERITY_BREAKS",  # dNBR class limits.
    "BURN_SEVERITY_CLASSES",  # dNBR class names.
    "INDEX_FUNCTIONS",  # Index functions by name.
    "arvi",  # Atmospherically resistant vegetation index.
    "awei_nsh",  # Water extraction index, no shadow.
    "awei_sh",  # Water extraction index, shadow.
    "bsi",  # Bare soil index.
    "burn_severity",  # dNBR severity classes.
    "ci_green",  # Green chlorophyll index.
    "ci_rededge",  # Red-edge chlorophyll index.
    "compute_index",  # Index by name.
    "cross_pol_ratio",  # Radar cross-polarisation ratio.
    "dnbr",  # Differenced burn ratio.
    "evi",  # Enhanced vegetation index.
    "evi2",  # Two-band EVI.
    "gndvi",  # Green NDVI.
    "kndvi",  # Kernel NDVI.
    "mndwi",  # Modified water index.
    "msavi",  # Modified soil adjusted vegetation index.
    "msi",  # Moisture stress index.
    "nbr",  # Normalized burn ratio.
    "nbr2",  # Normalized burn ratio 2.
    "ndbi",  # Built-up index.
    "ndmi",  # Moisture index.
    "ndre",  # Red-edge index.
    "ndsi",  # Snow index.
    "ndvi",  # Vegetation index.
    "ndwi",  # Water index.
    "normalized_difference",  # Generic normalised difference.
    "osavi",  # Optimised soil adjusted vegetation index.
    "rdnbr",  # Relative differenced burn ratio.
    "rvi",  # Radar vegetation index.
    "safe_divide",  # Division with NaN for zero denominators.
    "savi",  # Soil adjusted vegetation index.
    "vari",  # Visible atmospherically resistant index.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/indices/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
