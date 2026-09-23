# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : fuzz/fuzz_geojson.py
# Title       : Fuzz target for GeoJSON validation, bounds and orientation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; atheris to fuzz
# =============================================================================
#
# Abstract
# --------
# Feeds arbitrary bytes, parsed as JSON, to the GeoJSON functions of
# unbihexium.io.geojson, which read untrusted documents (files, REST
# requests and STAC items). The contract checked for every input:
#
#   geojson_problems  never raises and returns a list of strings
#   geojson_bounds    on a valid document returns four finite numbers with
#                     min <= max, or raises ValueError when there are no
#                     positions
#   rewind            on a valid document returns a document that is still
#                     valid, and rewinding twice gives the same result
#
# Any other exception is a bug and makes the fuzzer report a crash.
#
# Usage
# -----
#   python fuzz/fuzz_geojson.py [corpus directory] [libFuzzer options]
#
# ClusterFuzzLite builds and runs the target (see .clusterfuzzlite/). The
# unit tests call test_one_input directly on a fixed corpus.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the input bytes.
import json

# Check that the bounds are finite.
import math

# Command line arguments of the fuzzer.
import sys

# GeoJSON functions under test.
from unbihexium.io.geojson import geojson_bounds, geojson_problems, rewind


# Parse bytes as JSON, or None when they are not a JSON document.
def parse_json(data: bytes) -> object | None:
    # Parse the text; invalid UTF-8 and invalid JSON are not interesting.
    try:
        # JSON value.
        return json.loads(data.decode("utf-8"))
    # Rejected inputs, including documents nested too deeply for json.
    except (UnicodeDecodeError, ValueError, RecursionError):
        # Nothing to test.
        return None


# Check the contract of the GeoJSON functions on one input.
def test_one_input(data: bytes) -> None:
    # Parsed document.
    obj = parse_json(data)
    # Only JSON documents reach the library.
    if obj is None:
        # Nothing to test.
        return
    # Problems of the document; must never raise.
    problems = geojson_problems(obj)
    # A list of messages.
    assert isinstance(problems, list) and all(isinstance(p, str) for p in problems)
    # The remaining functions require a valid document.
    if problems:
        # Nothing more to test.
        return
    # Bounds of the positions.
    try:
        # Box.
        west, south, east, north = geojson_bounds(obj)
    # Documents without positions, for example a Feature with a null geometry.
    except ValueError:
        # Allowed by the contract.
        pass
    # A box was computed.
    else:
        # Finite numbers.
        assert all(math.isfinite(v) for v in (west, south, east, north))
        # Ordered corners.
        assert west <= east and south <= north
    # Oriented copy.
    once = rewind(obj)
    # Still valid.
    assert geojson_problems(once) == []
    # Rewinding is idempotent.
    assert rewind(once) == once


# Run the target under atheris, the coverage-guided fuzzer for Python.
def main() -> None:
    # Imported here so that the unit tests do not need atheris.
    import atheris  # Fuzzing engine.

    # Instrument the loaded library code for coverage feedback.
    atheris.instrument_all()
    # Pass the libFuzzer options and the target.
    atheris.Setup(sys.argv, test_one_input)
    # Fuzz until a crash or the time limit.
    atheris.Fuzz()


# Start the fuzzer when the file is executed as a script.
if __name__ == "__main__":
    # Run it.
    main()

# =============================================================================
# End of module fuzz/fuzz_geojson.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
