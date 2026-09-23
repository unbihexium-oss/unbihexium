# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/hashing.py
# Title       : Content digests of files, bytes, arrays and documents
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy
# =============================================================================
#
# Abstract
# --------
# SHA-256 digests (FIPS PUB 180-4) used for provenance records, cache keys
# and model verification:
#
#   compute_sha256   digest of a file, read in chunks of constant memory
#   sha256_bytes     digest of an in-memory byte string
#   array_digest     digest of an array that covers its dtype, shape and
#                    values, so that two arrays with the same bytes but a
#                    different layout never collide
#   json_digest      digest of a JSON document in canonical form (sorted
#                    keys, no insignificant whitespace), independent of the
#                    key order of the dictionaries
#
# References
# ----------
# National Institute of Standards and Technology (2015). Secure Hash
# Standard (SHS). FIPS PUB 180-4. doi:10.6028/NIST.FIPS.180-4
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# SHA-256 implementation.
import hashlib

# Canonical JSON serialisation.
import json

# Represent file paths.
from pathlib import Path

# Type of loosely structured values.
from typing import Any

# Arrays.
import numpy as np

# Default read size of file digests (1 MiB).
CHUNK_SIZE = 1 << 20


# SHA-256 hex digest of a file.
def compute_sha256(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    # Reject chunk sizes that would never make progress.
    if chunk_size < 1:
        # Explain the problem.
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # Running digest.
    digest = hashlib.sha256()
    # Read the file in binary mode.
    with open(path, "rb") as handle:
        # Feed the digest chunk by chunk.
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            # Add the chunk.
            digest.update(chunk)
    # Hexadecimal digest.
    return digest.hexdigest()


# SHA-256 hex digest of a byte string.
def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    # Digest of the bytes.
    return hashlib.sha256(bytes(data)).hexdigest()


# SHA-256 hex digest of an array, covering dtype, shape and values.
def array_digest(array: Any) -> str:
    # C-contiguous copy so that the byte order of the values is defined.
    values = np.ascontiguousarray(np.asarray(array))
    # Running digest.
    digest = hashlib.sha256()
    # The data type, including the byte order (for example "<f4").
    digest.update(values.dtype.str.encode("ascii"))
    # The shape separates arrays with identical bytes.
    digest.update(repr(tuple(int(n) for n in values.shape)).encode("ascii"))
    # The values in memory order.
    digest.update(values.tobytes(order="C"))
    # Hexadecimal digest.
    return digest.hexdigest()


# Canonical JSON text of a document: sorted keys, compact separators.
def canonical_json(document: Any) -> str:
    # NaN and infinity are not valid JSON and are rejected.
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


# SHA-256 hex digest of a JSON document in canonical form.
def json_digest(document: Any) -> str:
    # Digest of the UTF-8 encoded canonical text.
    return sha256_bytes(canonical_json(document).encode("utf-8"))


# =============================================================================
# End of module src/unbihexium/utils/hashing.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
