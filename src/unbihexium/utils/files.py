# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/files.py
# Title       : File system helpers
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
#   ensure_dir          create a directory and its parents
#   atomic_write_bytes  write a file through a temporary file in the same
#   atomic_write_text   directory and os.replace, so that readers never see
#                       a partially written file (os.replace is atomic on
#                       POSIX and Windows when source and target share a
#                       file system)
#   bytes_to_human      byte count with binary (IEC 80000-13) prefixes
#
# References
# ----------
# International Electrotechnical Commission (2008). IEC 80000-13:2008,
# Quantities and units, Part 13: Information science and technology.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# File descriptors and fsync.
import os

# Temporary files next to the target.
import tempfile

# Represent file paths.
from pathlib import Path

# Binary prefixes of IEC 80000-13.
BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")


# Create a directory with its parents and return it.
def ensure_dir(path: str | Path) -> Path:
    # Path object.
    path = Path(path)
    # Create it, tolerating an existing directory.
    path.mkdir(parents=True, exist_ok=True)
    # Return the path.
    return path


# Write bytes atomically.
def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    # Destination.
    path = Path(path)
    # Directory of the destination.
    ensure_dir(path.parent)
    # Temporary file on the same file system.
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    # Write, then replace; remove the temporary file on failure.
    try:
        # Open the descriptor as a file.
        with os.fdopen(fd, "wb") as handle:
            # Write the content.
            handle.write(data)
            # Flush Python's buffer.
            handle.flush()
            # Flush the operating system buffer.
            os.fsync(handle.fileno())
        # Atomic rename over the destination.
        Path(tmp).replace(path)
    # Clean up after any error.
    except BaseException:
        # Remove the temporary file if it still exists.
        Path(tmp).unlink(missing_ok=True)
        # Propagate the error.
        raise
    # Return the destination.
    return path


# Write text atomically.
def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    # Encode and write.
    return atomic_write_bytes(path, text.encode(encoding))


# Byte count as text with binary prefixes, for example 1536 -> "1.5 KiB".
def bytes_to_human(size: float, precision: int = 1) -> str:
    # Negative sizes are not meaningful.
    if size < 0:
        # Explain the problem.
        raise ValueError(f"size must not be negative, got {size}")
    # Work on a float copy.
    value = float(size)
    # Divide by 1024 until the value is below 1024 or the prefixes run out.
    for unit in BINARY_UNITS[:-1]:
        # Small enough for this unit.
        if value < 1024:
            # Whole bytes are shown without decimals.
            return f"{int(value)} B" if unit == "B" else f"{value:.{precision}f} {unit}"
        # Next prefix.
        value /= 1024
    # Largest prefix.
    return f"{value:.{precision}f} {BINARY_UNITS[-1]}"


# =============================================================================
# End of module src/unbihexium/utils/files.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
