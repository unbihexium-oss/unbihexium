# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/zoo/verify.py
# Title       : File checksums and model digest verification
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Two levels of integrity checking are provided:
#
#   file level    SHA-256 of files, and the sha256sum-compatible model.sha256
#                 file written next to every cached model
#   weight level  the digest of the weights (unbihexium.ai.models.init),
#                 compared with the published digest of the model zoo; this
#                 is independent of how the checkpoint file was serialised
#
# The weight-level check needs PyTorch and is performed by
# unbihexium.zoo.store.verify_model.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# SHA-256 of files.
import hashlib

# Represent file paths.
from pathlib import Path

# Size of the blocks read while hashing, 1 MiB.
BLOCK_SIZE = 1 << 20


# Raised when a file or model does not match its expected checksum.
class VerificationError(RuntimeError):
    # No behaviour beyond RuntimeError; the class exists for precise handling.
    pass


# Compute the SHA-256 of a file.
def compute_sha256(path: str | Path) -> str:
    # Incremental hash.
    hasher = hashlib.sha256()
    # Read the file in binary mode.
    with Path(path).open("rb") as fh:
        # Read blocks until the end of the file.
        for block in iter(lambda: fh.read(BLOCK_SIZE), b""):
            # Feed each block to the hash.
            hasher.update(block)
    # Hexadecimal digest.
    return hasher.hexdigest()


# Write a sha256sum-compatible checksum file; an unchanged file is not rewritten.
def write_sha256_file(
    directory: str | Path,  # Directory holding the files.
    names: list[str],  # File names to include.
    filename: str = "model.sha256",  # Name of the checksum file.
) -> Path:  # Path of the checksum file.
    # Normalise the directory.
    directory = Path(directory)
    # One "<digest>  <name>" line per existing file, in sorted order.
    present = [n for n in sorted(names) if (directory / n).is_file()]
    # One checksum line per file.
    lines = [f"{compute_sha256(directory / n)}  {n}" for n in present]
    # Path of the checksum file.
    target = directory / filename
    # Contents with a final newline.
    text = "\n".join(lines) + "\n"
    # Write only when the contents changed, so read-only stores stay usable.
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        # Write the file.
        target.write_text(text, encoding="utf-8", newline="\n")
    # Return the path.
    return target


# Read a sha256sum-compatible checksum file into {name: digest}.
def read_sha256_file(path: str | Path) -> dict[str, str]:
    # Mapping of file names to digests.
    result: dict[str, str] = {}
    # Parse every line.
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # Split into digest and name; sha256sum uses two separator characters.
        parts = line.strip().split(maxsplit=1)
        # Ignore blank and malformed lines.
        if len(parts) == 2:
            # A leading "*" marks binary mode in sha256sum output.
            result[parts[1].lstrip("*")] = parts[0].lower()
    # Return the mapping.
    return result


# Verify a file against an expected SHA-256 digest.
def verify_file(path: str | Path, expected: str) -> bool:
    # Missing files fail verification.
    if not Path(path).is_file():
        # The file does not exist.
        return False
    # Compare case-insensitively.
    return compute_sha256(path) == expected.lower()


# Verify every file listed in a model.sha256 file.
def verify_directory(directory: str | Path, filename: str = "model.sha256") -> dict[str, bool]:
    # Normalise the directory.
    directory = Path(directory)
    # Expected digests.
    expected = read_sha256_file(directory / filename)
    # Check each listed file.
    return {name: verify_file(directory / name, digest) for name, digest in expected.items()}


# =============================================================================
# End of module src/unbihexium/zoo/verify.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
