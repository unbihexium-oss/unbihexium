# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/config/__init__.py
# Title       : Configuration of the library
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires PyYAML
# =============================================================================
#
# Abstract
# --------
# Public interface of the configuration (see settings.py): the Config
# dataclass with its model, processing and serving sections, layered
# loading from defaults, YAML and UNBIHEXIUM_* environment variables, and
# the cached process-wide settings.
#
# Usage
# -----
#   from unbihexium.config import load_config
#   config = load_config("unbihexium.yaml")
#   config.processing.tile_size
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Configuration classes and loaders.
from unbihexium.config.settings import (
    CONFIG_ENV,  # Variable with the path of a YAML file.
    ENV_PREFIX,  # Prefix of the environment variables.
    Config,  # Complete configuration.
    ModelConfig,  # Model section.
    ProcessingConfig,  # Processing section.
    ServingConfig,  # Serving section.
    get_default_config,  # Defaults.
    get_settings,  # Cached process-wide settings.
    load_config,  # Layered loading.
    reset_settings,  # Forget the cached settings.
)  # End of the settings imports.

# Public names of the package.
__all__ = [
    "CONFIG_ENV",  # Variable with the path of a YAML file.
    "ENV_PREFIX",  # Prefix of the environment variables.
    "Config",  # Complete configuration.
    "ModelConfig",  # Model section.
    "ProcessingConfig",  # Processing section.
    "ServingConfig",  # Serving section.
    "get_default_config",  # Defaults.
    "get_settings",  # Cached process-wide settings.
    "load_config",  # Layered loading.
    "reset_settings",  # Forget the cached settings.
]  # End of the export list.


# =============================================================================
# End of module src/unbihexium/config/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
