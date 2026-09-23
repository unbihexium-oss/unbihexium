# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model Zoo package for unbihexium."""

from unbihexium.zoo.cache import (
    ModelCache,
    get_cache_dir,
    get_cached_model_path,
    is_model_cached,
)
from unbihexium.zoo.downloader import download_model
from unbihexium.zoo.registry import (
    ModelZooEntry,
    get_model,
    list_models,
    register_model,
)
from unbihexium.zoo.verify import (
    VerificationError,
    compute_sha256,
    verify_file,
    verify_model,
)

__all__ = [
    "ModelCache",
    "ModelZooEntry",
    "VerificationError",
    "compute_sha256",
    "download_model",
    "get_cache_dir",
    "get_cached_model_path",
    "get_model",
    "is_model_cached",
    "list_models",
    "register_model",
    "verify_file",
    "verify_model",
]
