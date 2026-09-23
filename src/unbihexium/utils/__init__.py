# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/__init__.py
# Title       : Common utilities: hashing, logging, timing, seeding, tiling
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; count_parameters
#               needs a PyTorch module
# =============================================================================
#
# Abstract
# --------
# Small helpers shared by the rest of the library:
#
#   hashing   compute_sha256, sha256_bytes, array_digest, json_digest
#   log       get_logger, configure_logging
#   timing    Timer, timed
#   seeding   set_seed, derive_seed, spawn_generators
#   tiling    tile_starts, tile_windows, tile_image, merge_tiles
#   files     ensure_dir, atomic_write_bytes, atomic_write_text,
#             bytes_to_human
#
# count_parameters counts the parameters of a PyTorch module without
# importing PyTorch itself.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Type of loosely structured values.
from typing import Any

# File system helpers.
from unbihexium.utils.files import (
    atomic_write_bytes,  # Atomic binary write.
    atomic_write_text,  # Atomic text write.
    bytes_to_human,  # Byte counts as text.
    ensure_dir,  # Create directories.
)  # End of the file helper imports.

# Digests.
from unbihexium.utils.hashing import (
    array_digest,  # Digest of an array.
    canonical_json,  # Canonical JSON text.
    compute_sha256,  # Digest of a file.
    json_digest,  # Digest of a JSON document.
    sha256_bytes,  # Digest of bytes.
)  # End of the digest imports.

# Logging.
from unbihexium.utils.log import (
    configure_logging,  # Install a handler.
    get_logger,  # Library logger.
    parse_level,  # Level names to numbers.
)  # End of the logging imports.

# Random number generation.
from unbihexium.utils.seeding import (
    derive_seed,  # Child seeds.
    set_seed,  # Seed every generator.
    spawn_generators,  # Independent generators.
)  # End of the seeding imports.

# Tiling.
from unbihexium.utils.tiling import (
    merge_tiles,  # Mosaic of tiles.
    tile_image,  # Tile generator.
    tile_starts,  # Starts along one axis.
    tile_windows,  # Tile windows.
)  # End of the tiling imports.

# Timing.
from unbihexium.utils.timing import (
    Timer,  # Elapsed-time measurement.
    timed,  # Timing decorator.
)  # End of the timing imports.


# Number of parameters of a PyTorch module.
def count_parameters(model: Any, trainable_only: bool = True) -> int:
    # Parameters to count: trainable ones, or all of them.
    params = [p for p in model.parameters() if p.requires_grad or not trainable_only]
    # Sum of their element counts.
    return int(sum(p.numel() for p in params))


# Public names of the package.
__all__ = [
    "Timer",  # Elapsed-time measurement.
    "array_digest",  # Digest of an array.
    "atomic_write_bytes",  # Atomic binary write.
    "atomic_write_text",  # Atomic text write.
    "bytes_to_human",  # Byte counts as text.
    "canonical_json",  # Canonical JSON text.
    "compute_sha256",  # Digest of a file.
    "configure_logging",  # Install a handler.
    "count_parameters",  # Parameters of a PyTorch module.
    "derive_seed",  # Child seeds.
    "ensure_dir",  # Create directories.
    "get_logger",  # Library logger.
    "json_digest",  # Digest of a JSON document.
    "merge_tiles",  # Mosaic of tiles.
    "parse_level",  # Level names to numbers.
    "set_seed",  # Seed every generator.
    "sha256_bytes",  # Digest of bytes.
    "spawn_generators",  # Independent generators.
    "tile_image",  # Tile generator.
    "tile_starts",  # Starts along one axis.
    "tile_windows",  # Tile windows.
    "timed",  # Timing decorator.
]  # End of the export list.


# =============================================================================
# End of module src/unbihexium/utils/__init__.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
