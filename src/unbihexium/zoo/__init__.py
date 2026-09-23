# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/__init__.py
# Title       : Model zoo: catalogue, registry and local model store
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14; building models requires PyTorch
# =============================================================================
#
# Abstract
# --------
# The model zoo offers 130 model families in four size variants (520
# models). Every model is a starter model with a trainable architecture and
# deterministic, verifiable starter weights; see catalog.yaml for what each
# model does and what training data it needs. Listing and describing models
# works without PyTorch; building, training and exporting models requires
# unbihexium[torch].
#
# Usage
# -----
#   from unbihexium.zoo import list_models, load_model
#   for entry in list_models(task="detection", variant="tiny"):
#       print(entry.model_id, entry.spec.description)
#   model = load_model("ship_detector_base")
# =============================================================================

# Catalogue of model families, tasks and variants.
from unbihexium.zoo.catalog import (
    CatalogError,  # Raised for invalid catalogue entries or names.
    ModelSpec,  # Catalogue entry of a model family.
    Task,  # Task enumeration.
    Variant,  # Size variant enumeration.
    VariantSpec,  # Hyperparameters of a size variant.
    all_model_ids,  # Every model id of the zoo.
    catalog_version,  # Version of the catalogue.
    get_spec,  # Look up a model family.
    get_variant,  # Look up a size variant.
    list_specs,  # List model families.
    parse_model_id,  # Split a model id into family and variant.
)  # End of the catalogue imports.

# Registry of model entries.
from unbihexium.zoo.registry import (
    ModelZooEntry,  # Description of one model.
    get_model,  # Look up a model entry.
    list_models,  # List model entries.
    register_model,  # Register a user model.
    unregister_model,  # Remove a user model.
)  # End of the registry imports.

# Local model store.
from unbihexium.zoo.store import (
    clear_cache,  # Remove cached models.
    download_model,  # Obtain a model and return its checkpoint path.
    ensure_model,  # Obtain a model in the cache and return its directory.
    get_cache_dir,  # Root directory of the model cache.
    get_cached_model_path,  # Checkpoint path of a cached model.
    is_model_cached,  # Whether a model is cached.
    list_cached,  # Model ids present in the cache.
    load_model,  # Load a model into memory.
    model_dir,  # Cache directory of a model.
    verify_model,  # Verify a cached model.
)  # End of the store imports.

# File checksums.
from unbihexium.zoo.verify import (
    VerificationError,  # Raised when a checksum does not match.
    compute_sha256,  # SHA-256 of a file.
    read_sha256_file,  # Read a sha256sum file.
    verify_directory,  # Verify the files listed in a sha256sum file.
    verify_file,  # Verify one file.
    write_sha256_file,  # Write a sha256sum file.
)  # End of the checksum imports.

# Names exported by `from unbihexium.zoo import *`.
__all__ = [
    "CatalogError",  # Raised for invalid catalogue entries or names.
    "ModelSpec",  # Catalogue entry of a model family.
    "ModelZooEntry",  # Description of one model.
    "Task",  # Task enumeration.
    "Variant",  # Size variant enumeration.
    "VariantSpec",  # Hyperparameters of a size variant.
    "VerificationError",  # Raised when a checksum does not match.
    "all_model_ids",  # Every model id of the zoo.
    "catalog_version",  # Version of the catalogue.
    "clear_cache",  # Remove cached models.
    "compute_sha256",  # SHA-256 of a file.
    "download_model",  # Obtain a model and return its checkpoint path.
    "ensure_model",  # Obtain a model in the cache and return its directory.
    "get_cache_dir",  # Root directory of the model cache.
    "get_cached_model_path",  # Checkpoint path of a cached model.
    "get_model",  # Look up a model entry.
    "get_spec",  # Look up a model family.
    "get_variant",  # Look up a size variant.
    "is_model_cached",  # Whether a model is cached.
    "list_cached",  # Model ids present in the cache.
    "list_models",  # List model entries.
    "list_specs",  # List model families.
    "load_model",  # Load a model into memory.
    "model_dir",  # Cache directory of a model.
    "parse_model_id",  # Split a model id into family and variant.
    "read_sha256_file",  # Read a sha256sum file.
    "register_model",  # Register a user model.
    "unregister_model",  # Remove a user model.
    "verify_directory",  # Verify the files listed in a sha256sum file.
    "verify_file",  # Verify one file.
    "verify_model",  # Verify a cached model.
    "write_sha256_file",  # Write a sha256sum file.
]  # End of the export list.

# =============================================================================
# End of module src/unbihexium/zoo/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
