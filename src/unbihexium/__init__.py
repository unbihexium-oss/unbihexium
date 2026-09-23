# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/__init__.py
# Title       : Earth observation, geospatial, remote sensing and SAR library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# Top-level package. Importing it is cheap: only the version is loaded, and
# the subpackages are imported by the caller when needed:
#
#   core       rasters, vectors, tiles, spectral indices, sensors, scenes,
#              products, models, pipelines and provenance
#   ai         task APIs, inference and training of the model zoo
#   zoo        model catalogue, weights and export
#   cli        the `unbihexium` command line
#
# Usage
# -----
#   import unbihexium
#   print(unbihexium.__version__)
#   from unbihexium.core import Raster, compute_index
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Version string and tuple.
from unbihexium._version import __version__, __version_tuple__

# Public names of the package.
__all__ = [
    "__version__",  # Version string.
    "__version_tuple__",  # Version as a tuple of integers.
]

# =============================================================================
# End of module src/unbihexium/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
