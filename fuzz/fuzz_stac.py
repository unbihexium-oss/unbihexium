# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : fuzz/fuzz_stac.py
# Title       : Fuzz target for STAC item and time parsing
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; atheris to fuzz
# =============================================================================
#
# Abstract
# --------
# Feeds arbitrary bytes to the parsers of unbihexium.io.stac, which read
# documents returned by remote STAC APIs and catalogues. The contract
# checked for every input:
#
#   parse_datetime        raises only ValueError and returns a time with a
#                         zone
#   parse_datetime_range  raises only ValueError and returns an ordered
#                         interval
#   STACItem.from_dict    raises only ValueError on a malformed item; an
#                         item it accepts survives a round trip through
#                         to_dict with the same id, box and time
#
# The input text is used as a time and, when it is a JSON object, as an
# item. Any other exception is a bug and makes the fuzzer report a crash.
#
# Usage
# -----
#   python fuzz/fuzz_stac.py [corpus directory] [libFuzzer options]
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse the input bytes.
import json

# Command line arguments of the fuzzer.
import sys

# STAC functions under test.
from unbihexium.io.stac import STACItem, parse_datetime, parse_datetime_range


# Check the time parsers on one text.
def check_times(text: str) -> None:
    # Single instant.
    try:
        # Parsed time.
        instant = parse_datetime(text)
    # Malformed times are rejected with ValueError.
    except ValueError:
        # Allowed by the contract.
        pass
    # A time was parsed.
    else:
        # It carries a zone.
        assert instant.tzinfo is not None
    # Interval.
    try:
        # Parsed ends.
        start, end = parse_datetime_range(text)
    # Malformed intervals are rejected with ValueError.
    except ValueError:
        # Allowed by the contract.
        pass
    # An interval was parsed.
    else:
        # Closed intervals are ordered.
        assert start is None or end is None or start <= end


# Check the item parser on one JSON object.
def check_item(obj: dict) -> None:
    # Parse the item.
    try:
        # Item record.
        item = STACItem.from_dict(obj)
    # Malformed items are rejected with ValueError.
    except ValueError:
        # Allowed by the contract.
        return
    # Parse the serialised item again.
    again = STACItem.from_dict(item.to_dict())
    # Same identity, box and time.
    assert (again.id, again.bbox, again.datetime) == (item.id, item.bbox, item.datetime)


# Check the contract of the STAC parsers on one input.
def test_one_input(data: bytes) -> None:
    # Text of the input; invalid UTF-8 is not interesting.
    try:
        # Decoded text.
        text = data.decode("utf-8")
    # Not UTF-8.
    except UnicodeDecodeError:
        # Nothing to test.
        return
    # Times.
    check_times(text)
    # Items are JSON objects.
    try:
        # JSON value.
        obj = json.loads(text)
    # Invalid JSON, including documents nested too deeply for json.
    except (ValueError, RecursionError):
        # Nothing more to test.
        return
    # Only objects can be items.
    if isinstance(obj, dict):
        # Item parser.
        check_item(obj)


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
# End of module fuzz/fuzz_stac.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
