"""Spectral indices for vegetation and environmental analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class IndexCategory(str, Enum):
    """Categories of spectral indices."""

    VEGETATION = "vegetation"
    WATER = "water"
    SOIL = "soil"
    BURN = "burn"
    URBAN = "urban"


@dataclass
class SpectralIndex:
    """Spectral index definition."""

    name: str
    formula: str
    category: IndexCategory
    bands_required: list[str]
    value_range: tuple[float, float] = (-1.0, 1.0)
    description: str = ""

    def compute(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        """Compute the index from band arrays."""
        missing = [b for b in self.bands_required if b not in bands]
        if missing:
            msg = f"Missing bands: {missing}"
            raise ValueError(msg)
        return self._calculate(bands)

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        raise NotImplementedError


class NDVI(SpectralIndex):
    """Normalized Difference Vegetation Index."""

    def __init__(self) -> None:
        super().__init__(
            name="NDVI",
            formula="(NIR - RED) / (NIR + RED)",
            category=IndexCategory.VEGETATION,
            bands_required=["NIR", "RED"],
            description="Measures vegetation health and density",
        )

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        nir = bands["NIR"].astype(np.float64)
        red = bands["RED"].astype(np.float64)
        denom = nir + red
        denom = np.where(denom == 0, 1e-10, denom)
        return (nir - red) / denom


class NDWI(SpectralIndex):
    """Normalized Difference Water Index."""

    def __init__(self) -> None:
        super().__init__(
            name="NDWI",
            formula="(GREEN - NIR) / (GREEN + NIR)",
            category=IndexCategory.WATER,
            bands_required=["GREEN", "NIR"],
            description="Identifies water bodies and moisture content",
        )

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        green = bands["GREEN"].astype(np.float64)
        nir = bands["NIR"].astype(np.float64)
        denom = green + nir
        denom = np.where(denom == 0, 1e-10, denom)
        return (green - nir) / denom


class NBR(SpectralIndex):
    """Normalized Burn Ratio."""

    def __init__(self) -> None:
        super().__init__(
            name="NBR",
            formula="(NIR - SWIR2) / (NIR + SWIR2)",
            category=IndexCategory.BURN,
            bands_required=["NIR", "SWIR2"],
            description="Identifies burned areas",
        )

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        nir = bands["NIR"].astype(np.float64)
        swir2 = bands["SWIR2"].astype(np.float64)
        denom = nir + swir2
        denom = np.where(denom == 0, 1e-10, denom)
        return (nir - swir2) / denom


class EVI(SpectralIndex):
    """Enhanced Vegetation Index."""

    def __init__(self, g: float = 2.5, c1: float = 6.0, c2: float = 7.5, l: float = 1.0) -> None:
        super().__init__(
            name="EVI",
            formula="G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)",
            category=IndexCategory.VEGETATION,
            bands_required=["NIR", "RED", "BLUE"],
            description="Enhanced vegetation index with atmospheric correction",
        )
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.l = l

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        nir = bands["NIR"].astype(np.float64)
        red = bands["RED"].astype(np.float64)
        blue = bands["BLUE"].astype(np.float64)
        denom = nir + self.c1 * red - self.c2 * blue + self.l
        denom = np.where(denom == 0, 1e-10, denom)
        return self.g * (nir - red) / denom


class SAVI(SpectralIndex):
    """Soil Adjusted Vegetation Index."""

    def __init__(self, l: float = 0.5) -> None:
        super().__init__(
            name="SAVI",
            formula="((NIR - RED) / (NIR + RED + L)) * (1 + L)",
            category=IndexCategory.VEGETATION,
            bands_required=["NIR", "RED"],
            description="Vegetation index with soil brightness correction",
        )
        self.l = l

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        nir = bands["NIR"].astype(np.float64)
        red = bands["RED"].astype(np.float64)
        denom = nir + red + self.l
        denom = np.where(denom == 0, 1e-10, denom)
        return ((nir - red) / denom) * (1 + self.l)


class MSI(SpectralIndex):
    """Moisture Stress Index."""

    def __init__(self) -> None:
        super().__init__(
            name="MSI",
            formula="SWIR1 / NIR",
            category=IndexCategory.VEGETATION,
            bands_required=["SWIR1", "NIR"],
            value_range=(0.0, 3.0),
            description="Identifies plant water stress",
        )

    def _calculate(self, bands: dict[str, NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        swir1 = bands["SWIR1"].astype(np.float64)
        nir = bands["NIR"].astype(np.float64)
        nir = np.where(nir == 0, 1e-10, nir)
        return swir1 / nir


class IndexRegistry:
    """Registry of spectral indices."""

    _indices: dict[str, SpectralIndex] = {}

    @classmethod
    def register(cls, index: SpectralIndex) -> None:
        cls._indices[index.name] = index

    @classmethod
    def get(cls, name: str) -> SpectralIndex | None:
        return cls._indices.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._indices.keys())

    @classmethod
    def by_category(cls, category: IndexCategory) -> list[SpectralIndex]:
        return [idx for idx in cls._indices.values() if idx.category == category]


# Register built-in indices
IndexRegistry.register(NDVI())
IndexRegistry.register(NDWI())
IndexRegistry.register(NBR())
IndexRegistry.register(EVI())
IndexRegistry.register(SAVI())
IndexRegistry.register(MSI())


def compute_index(
    name: str,
    bands: dict[str, NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    """Compute a spectral index by name."""
    index = IndexRegistry.get(name)
    if index is None:
        msg = f"Unknown index: {name}"
        raise ValueError(msg)
    return index.compute(bands)
