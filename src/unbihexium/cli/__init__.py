# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/cli/__init__.py
# Title       : Command line package
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires click and rich
# =============================================================================
#
# Abstract
# --------
# Exposes the click group of the `unbihexium` command as `main` (the console
# script entry point) and `cli` (the name used by earlier releases).
# =============================================================================

# Root command group under both names.
from unbihexium.cli.main import cli, main

# Public names of the package.
__all__ = [
    "cli",  # Name used by earlier releases.
    "main",  # Console script entry point.
]  # End of the export list.

# =============================================================================
# End of module src/unbihexium/cli/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
