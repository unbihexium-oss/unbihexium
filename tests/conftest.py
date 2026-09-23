# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/conftest.py
# Title       : Shared pytest fixtures
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, NumPy and rasterio
# =============================================================================
#
# Abstract
# --------
# Fixtures used by several test modules:
#
#   sample_raster_data   seeded three-band float32 array of 256 x 256 pixels
#   tmp_geotiff          the same array written as a georeferenced GeoTIFF
#   sample_bands         seeded reflectance bands keyed by name for indices
#   isolated_cache       a temporary model cache for tests that build models
#
# Every fixture uses its own seeded generator, so the values do not depend on
# the order in which tests run.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Iterator type of generator fixtures.
from collections.abc import Iterator

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Test framework.
import pytest

# Array type annotations.
from numpy.typing import NDArray


# Seeded three-band raster array of 256 x 256 pixels.
@pytest.fixture
def sample_raster_data() -> NDArray[np.floating[Any]]:
    # Independent generator with a fixed seed.
    rng = np.random.default_rng(42)
    # Uniform reflectances in [0, 1).
    return rng.random((3, 256, 256), dtype=np.float32)


# The sample array written as a GeoTIFF in geographic coordinates.
@pytest.fixture
def tmp_geotiff(tmp_path: Path, sample_raster_data: NDArray[np.floating[Any]]) -> Iterator[Path]:
    # Imported here so that tests without GeoTIFFs do not need rasterio.
    import rasterio  # GeoTIFF writing.
    from rasterio.transform import from_bounds  # Transform of a bounding box.

    # Destination file.
    path = tmp_path / "test.tif"
    # Unit square covered by 256 x 256 pixels.
    transform = from_bounds(0, 0, 1, 1, 256, 256)
    # Write the file.
    with rasterio.open(
        path,  # File.
        "w",  # Write mode.
        driver="GTiff",  # Format.
        height=256,  # Rows.
        width=256,  # Columns.
        count=3,  # Bands.
        dtype="float32",  # Data type.
        crs="EPSG:4326",  # Geographic coordinates.
        transform=transform,  # Pixel grid.
    ) as dst:  # Dataset handle.
        # Write the bands.
        dst.write(sample_raster_data)
    # Hand the path to the test.
    yield path
    # Remove the file afterwards.
    path.unlink(missing_ok=True)


# Seeded reflectance bands keyed by name.
@pytest.fixture
def sample_bands() -> dict[str, NDArray[np.floating[Any]]]:
    # Independent generator with a fixed seed.
    rng = np.random.default_rng(42)
    # Upper bound of each band's reflectance.
    scales = {"RED": 0.5, "NIR": 0.8, "GREEN": 0.4, "BLUE": 0.3, "SWIR1": 0.6, "SWIR2": 0.5}
    # One band per name.
    return {name: rng.random((256, 256), dtype=np.float32) * s for name, s in scales.items()}


# Temporary model cache, so that tests never write to the user's cache.
@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Cache root inside the test directory.
    root = tmp_path / "cache"
    # Point the model store at it.
    monkeypatch.setenv("UNBIHEXIUM_CACHE", str(root))
    # Return the root.
    return root


# =============================================================================
# End of module tests/conftest.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
