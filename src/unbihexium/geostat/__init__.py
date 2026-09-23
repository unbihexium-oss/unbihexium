# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/geostat/__init__.py
# Title       : Geostatistics and spatial statistics
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Analysis and interpolation of point data:
#
#   variogram  empirical semivariograms and model fitting (spherical,
#              exponential, gaussian, Matern, linear, power)
#   kriging    ordinary and universal kriging with variances, local
#              neighbourhoods and cross-validation; inverse distance
#              weighting
#   spatial    spatial weights, Moran's I, Geary's C, Getis-Ord Gi* and
#              local Moran's I
# =============================================================================

# Interpolation.
from unbihexium.geostat.kriging import (
    KrigingResult,  # Predictions and variances.
    OrdinaryKriging,  # Constant unknown mean.
    UniversalKriging,  # Polynomial trend.
    idw,  # Inverse distance weighting.
)  # End of the kriging imports.

# Spatial autocorrelation.
from unbihexium.geostat.spatial import (
    GearysC,  # Geary's C from coordinates.
    LocalStatisticResult,  # Local statistic record.
    MoransI,  # Moran's I from coordinates.
    SpatialAutocorrelationResult,  # Global statistic record.
    contiguity_weights,  # Grid contiguity weights.
    distance_band_weights,  # Distance band weights.
    gearys_c,  # Geary's C from weights.
    getis_ord_gi_star,  # Hot spot z-scores.
    grid_morans_i,  # Moran's I of a raster.
    knn_weights,  # Nearest-neighbour weights.
    local_morans_i,  # LISA.
    morans_i,  # Moran's I from weights.
    row_standardize,  # Row standardisation.
    weight_sums,  # S0, S1, S2.
)  # End of the spatial imports.

# Semivariograms.
from unbihexium.geostat.variogram import (
    Variogram,  # Estimation and fitting.
    VariogramModel,  # Model families.
    VariogramResult,  # Fitted parameters.
    empirical_variogram,  # Binned estimate.
    variogram_function,  # Model evaluation.
)  # End of the variogram imports.

# Public names of the package.
__all__ = [
    "GearysC",  # Geary's C from coordinates.
    "KrigingResult",  # Predictions and variances.
    "LocalStatisticResult",  # Local statistic record.
    "MoransI",  # Moran's I from coordinates.
    "OrdinaryKriging",  # Constant unknown mean.
    "SpatialAutocorrelationResult",  # Global statistic record.
    "UniversalKriging",  # Polynomial trend.
    "Variogram",  # Estimation and fitting.
    "VariogramModel",  # Model families.
    "VariogramResult",  # Fitted parameters.
    "contiguity_weights",  # Grid contiguity weights.
    "distance_band_weights",  # Distance band weights.
    "empirical_variogram",  # Binned estimate.
    "gearys_c",  # Geary's C from weights.
    "getis_ord_gi_star",  # Hot spot z-scores.
    "grid_morans_i",  # Moran's I of a raster.
    "idw",  # Inverse distance weighting.
    "knn_weights",  # Nearest-neighbour weights.
    "local_morans_i",  # LISA.
    "morans_i",  # Moran's I from weights.
    "row_standardize",  # Row standardisation.
    "variogram_function",  # Model evaluation.
    "weight_sums",  # S0, S1, S2.
]  # End of the public names.

# =============================================================================
# End of module src/unbihexium/geostat/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
