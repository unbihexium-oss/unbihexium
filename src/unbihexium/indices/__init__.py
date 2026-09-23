# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Spectral indices computation module.

This module provides functions for computing common spectral indices
from multispectral satellite imagery.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ndvi(nir: NDArray, red: NDArray, epsilon: float = 1e-8) -> NDArray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-infrared band array
        red: Red band array
        epsilon: Small value to avoid division by zero

    Returns:
        NDVI array with values in range [-1, 1]
    """
    return (nir - red) / (nir + red + epsilon)


def ndwi(green: NDArray, nir: NDArray, epsilon: float = 1e-8) -> NDArray:
    """Compute Normalized Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR)

    Args:
        green: Green band array
        nir: Near-infrared band array
        epsilon: Small value to avoid division by zero

    Returns:
        NDWI array with values in range [-1, 1]
    """
    return (green - nir) / (green + nir + epsilon)


def evi(
    nir: NDArray,
    red: NDArray,
    blue: NDArray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
) -> NDArray:
    """Compute Enhanced Vegetation Index.

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Args:
        nir: Near-infrared band array
        red: Red band array
        blue: Blue band array
        g: Gain factor (default 2.5)
        c1: Coefficient 1 (default 6.0)
        c2: Coefficient 2 (default 7.5)
        l: Canopy background adjustment (default 1.0)

    Returns:
        EVI array
    """
    return g * (nir - red) / (nir + c1 * red - c2 * blue + l)


def savi(nir: NDArray, red: NDArray, l: float = 0.5, epsilon: float = 1e-8) -> NDArray:
    """Compute Soil Adjusted Vegetation Index.

    SAVI = ((NIR - Red) * (1 + L)) / (NIR + Red + L)

    Args:
        nir: Near-infrared band array
        red: Red band array
        l: Soil brightness correction factor (default 0.5)
        epsilon: Small value to avoid division by zero

    Returns:
        SAVI array
    """
    return ((nir - red) * (1 + l)) / (nir + red + l + epsilon)


def nbr(nir: NDArray, swir: NDArray, epsilon: float = 1e-8) -> NDArray:
    """Compute Normalized Burn Ratio.

    NBR = (NIR - SWIR) / (NIR + SWIR)

    Args:
        nir: Near-infrared band array
        swir: Shortwave infrared band array
        epsilon: Small value to avoid division by zero

    Returns:
        NBR array with values in range [-1, 1]
    """
    return (nir - swir) / (nir + swir + epsilon)


def msi(swir: NDArray, nir: NDArray, epsilon: float = 1e-8) -> NDArray:
    """Compute Moisture Stress Index.

    MSI = SWIR / NIR

    Args:
        swir: Shortwave infrared band array
        nir: Near-infrared band array
        epsilon: Small value to avoid division by zero

    Returns:
        MSI array (higher values indicate water stress)
    """
    return swir / (nir + epsilon)


def dnbr(nbr_pre: NDArray, nbr_post: NDArray) -> NDArray:
    """Compute differenced Normalized Burn Ratio.

    dNBR = NBR_pre - NBR_post

    Args:
        nbr_pre: Pre-fire NBR array
        nbr_post: Post-fire NBR array

    Returns:
        dNBR array (positive values indicate burn severity)
    """
    return nbr_pre - nbr_post


__all__ = [
    "ndvi",
    "ndwi",
    "evi",
    "savi",
    "nbr",
    "msi",
    "dnbr",
]
