# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_utils.py
# Title       : Tests of hashing, logging, timing, seeding, tiling and files
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest
# =============================================================================
#
# Abstract
# --------
# Checks the utilities against known values: the SHA-256 digests of the
# empty string and of "abc" from FIPS 180-4, tile starts computed by hand,
# exact reconstruction of an image from overlapping tiles, weighted means of
# overlaps, IEC byte prefixes, reproducible seeding and timers driven by a
# fake clock.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Standard logging.
import logging

# Python's own generator.
import random

# In-memory log output.
from io import StringIO

# Represent file paths.
from pathlib import Path

# Arrays.
import numpy as np

# Test framework.
import pytest

# Utilities under test.
from unbihexium.utils import (
    Timer,  # Timing.
    array_digest,  # Array digests.
    atomic_write_text,  # Atomic writes.
    bytes_to_human,  # Byte counts.
    compute_sha256,  # File digests.
    configure_logging,  # Logging setup.
    derive_seed,  # Child seeds.
    get_logger,  # Loggers.
    json_digest,  # Document digests.
    merge_tiles,  # Mosaics.
    parse_level,  # Log levels.
    set_seed,  # Seeding.
    sha256_bytes,  # Byte digests.
    spawn_generators,  # Parallel generators.
    tile_image,  # Tiles.
    tile_starts,  # Tile starts.
    tile_windows,  # Tile windows.
)  # End of the utility imports.

# SHA-256 of the empty string (FIPS 180-4 test vector).
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# SHA-256 of "abc" (FIPS 180-4 test vector).
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# Digests match the published test vectors and cover layout and key order.
def test_hashing(tmp_path: Path) -> None:
    # Byte strings.
    assert sha256_bytes(b"") == EMPTY_SHA256 and sha256_bytes(b"abc") == ABC_SHA256
    # File read in chunks of one byte.
    (tmp_path / "abc.txt").write_bytes(b"abc")
    # Same digest as the bytes.
    assert compute_sha256(tmp_path / "abc.txt", chunk_size=1) == ABC_SHA256
    # Same bytes, different shapes.
    a = np.arange(6, dtype=np.int32)
    # Digests differ because the shape is included.
    assert array_digest(a) != array_digest(a.reshape(2, 3))
    # Non-contiguous views hash like their contiguous copies.
    assert array_digest(a.reshape(2, 3).T) == array_digest(np.ascontiguousarray(a.reshape(2, 3).T))
    # Key order does not matter.
    assert json_digest({"a": 1, "b": [1, 2]}) == json_digest({"b": [1, 2], "a": 1})
    # NaN is not JSON.
    with pytest.raises(ValueError):
        # Rejected.
        json_digest({"x": float("nan")})


# Tile starts cover the axis with full-size tiles.
def test_tile_starts_and_windows() -> None:
    # Axis of 10 with tiles of 4 and overlap 1: steps of 3, last tile ends at 10.
    assert tile_starts(10, 4, 1) == [0, 3, 6]
    # Axis of 11: an extra tile ends at the border.
    assert tile_starts(11, 4, 1) == [0, 3, 6, 7]
    # Axes smaller than a tile hold one tile.
    assert tile_starts(3, 8) == [0]
    # Overlaps must be smaller than the tile.
    with pytest.raises(ValueError):
        # Overlap equals the tile.
        tile_starts(10, 4, 4)
    # Windows of a 5 x 7 image with tiles of 4 and no overlap.
    assert tile_windows(5, 7, 4) == [(0, 0, 4, 4), (0, 3, 4, 4), (1, 0, 4, 4), (1, 3, 4, 4)]


# Tiles merge back into the original image.
def test_tile_and_merge() -> None:
    # Image with distinct values.
    image = np.arange(3 * 20 * 23, dtype=np.float32).reshape(3, 20, 23)
    # Overlapping tiles.
    tiles = list(tile_image(image, tile_size=8, overlap=3))
    # Every tile has the full size.
    assert all(t.shape == (3, 8, 8) for t, _, _ in tiles)
    # Mosaic of the tiles.
    merged = merge_tiles([t for t, _, _ in tiles], [(r, c) for _, r, c in tiles], image.shape)
    # Exact reconstruction.
    np.testing.assert_array_equal(merged, image)
    # Two overlapping one-pixel tiles with weights 1 and 3 average to (1*2 + 3*6) / 4 = 5.
    mean = merge_tiles([np.full((1, 1), 2.0)], [(0, 0)], (1, 2))
    # The uncovered pixel is NaN.
    assert mean[0, 0] == 2.0 and np.isnan(mean[0, 1])
    # Weighted overlap.
    weighted = merge_tiles(
        [np.full((1, 1), 2.0), np.full((1, 1), 6.0)],  # Values 2 and 6.
        [(0, 0), (0, 0)],  # Same pixel.
        (1, 1),  # One-pixel mosaic.
        weights=np.ones((1, 1)),  # Uniform weights.
    )  # End of the merge.
    # Mean of 2 and 6.
    assert weighted[0, 0] == 4.0
    # Tiles outside the mosaic are errors.
    with pytest.raises(ValueError):
        # Tile at column 1 of a one-column mosaic.
        merge_tiles([np.zeros((1, 1))], [(0, 1)], (1, 1))


# Binary prefixes and atomic writes.
def test_files(tmp_path: Path) -> None:
    # Bytes, kibibytes and mebibytes.
    assert bytes_to_human(512) == "512 B" and bytes_to_human(1536) == "1.5 KiB"
    # 3 MiB.
    assert bytes_to_human(3 * 1024**2) == "3.0 MiB"
    # Negative sizes are errors.
    with pytest.raises(ValueError):
        # Negative size.
        bytes_to_human(-1)
    # Atomic write into a new directory.
    target = atomic_write_text(tmp_path / "a" / "b.txt", "hello")
    # Content.
    assert target.read_text(encoding="utf-8") == "hello"
    # No temporary files left.
    assert [p.name for p in target.parent.iterdir()] == ["b.txt"]


# Seeds reproduce sequences and derived seeds are independent of order.
def test_seeding() -> None:
    # First run.
    first = set_seed(42)
    # Values of the first run.
    values = (random.random(), first.random())
    # Second run with the same seed.
    second = set_seed(42)
    # Same values.
    assert values == (random.random(), second.random())
    # Derived seeds depend on the key.
    assert derive_seed(1, "tile", 3) == derive_seed(1, "tile", 3) != derive_seed(1, "tile", 4)
    # Keys are separated: ("1", "2") differs from ("12",).
    assert derive_seed(0, "1", "2") != derive_seed(0, "12")
    # Seeds must fit NumPy.
    with pytest.raises(ValueError):
        # Too large.
        set_seed(2**32)
    # Spawned generators give different streams.
    a, b = spawn_generators(7, 2)
    # First draws differ.
    assert a.random() != b.random()


# Timers with a fake clock, and library logging.
def test_timer_and_logging() -> None:
    # Fake clock values.
    ticks = iter([10.0, 10.5, 12.0, 13.0])
    # Timer with the fake clock.
    timer = Timer("block", clock=lambda: next(ticks))
    # Start at 10.
    timer.start()
    # Lap at 10.5.
    assert timer.lap("read") == pytest.approx(0.5)
    # Lap at 12.
    assert timer.lap("predict") == pytest.approx(1.5)
    # Stop at 13: three seconds in total.
    assert timer.stop() == pytest.approx(3.0)
    # Laps by name.
    assert [name for name, _ in timer.laps] == ["read", "predict"]
    # Level names and numbers.
    assert parse_level("info") == logging.INFO and parse_level("10") == logging.DEBUG
    # Unknown names are errors.
    with pytest.raises(ValueError):
        # No such level.
        parse_level("loud")
    # Logging into a buffer.
    buffer = StringIO()
    # One handler, even after two calls.
    configure_logging("INFO", stream=buffer)
    # Second call replaces the handler.
    logger = configure_logging("INFO", stream=buffer)
    # Library children propagate to it.
    get_logger("io").info("written")
    # One line with the UTC time stamp and the logger name.
    lines = buffer.getvalue().splitlines()
    # Exactly one record.
    assert len(lines) == 1 and lines[0].endswith("unbihexium.io: written") and "Z INFO" in lines[0]
    # Remove the handler again.
    logger.handlers.clear()


# =============================================================================
# End of module tests/unit/test_utils.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
