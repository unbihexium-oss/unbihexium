# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/_version.py
# Title       : Version of the package
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Single source of the package version. The string follows Semantic
# Versioning 2.0.0 (MAJOR.MINOR.PATCH) and must equal the "version" field of
# pyproject.toml; the tuple holds the same three numbers as integers so that
# callers can compare versions without parsing the string.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Version string, identical to the version in pyproject.toml.
__version__ = "1.0.1"

# Major, minor and patch numbers of the version string.
__version_tuple__ = (1, 0, 1)

# =============================================================================
# End of module src/unbihexium/_version.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
