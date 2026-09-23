# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_fuzz_targets.py
# Title       : Regression tests for the fuzz targets and their corpus
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest and NumPy
# =============================================================================
#
# Abstract
# --------
# Runs the fuzz targets of fuzz/ on every file of their seed corpus in
# fuzz/corpus/<target>/, without atheris. The corpus holds valid examples
# and, in the files named regression_*, the inputs that crashed the targets
# before the fixes:
#
#   regression_degenerate_ring    rewind flipped rings of zero area on every
#                                 call
#   regression_collinear_ring     rewind flipped rings of collinear points,
#                                 whose area is only rounding error
#   regression_mixed_dimensions   rings mixing 2-D and 3-D positions broke
#                                 the shoelace area
#   regression_type_not_string    a non-string "type" raised TypeError
#   regression_huge_integer       integers beyond the float range raised
#                                 OverflowError
#   regression_short_bbox         a bbox of three numbers raised IndexError
#   regression_wrong_member_types links, assets or times of the wrong JSON
#                                 type raised AttributeError
#
# A random sample of bytes also runs through each target, so that the
# contract (only the documented exceptions) is checked beyond the corpus.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Load the targets from their files.
import importlib.util

# Represent file paths.
from pathlib import Path

# Type of a loaded module.
from types import ModuleType

# Seeded random bytes.
import numpy as np

# Test framework.
import pytest

# Directory of the fuzz targets.
FUZZ = Path(__file__).resolve().parents[2] / "fuzz"

# Target names; each has fuzz/fuzz_<name>.py and fuzz/corpus/<name>/.
TARGETS = ("geojson", "stac")


# Import fuzz/fuzz_<name>.py as a module.
def load_target(name: str) -> ModuleType:
    # Module specification from the file.
    spec = importlib.util.spec_from_file_location(f"fuzz_{name}", FUZZ / f"fuzz_{name}.py")
    # Module object.
    module = importlib.util.module_from_spec(spec)
    # Execute the module; main() only runs when the file is a script.
    spec.loader.exec_module(module)
    # Return it.
    return module


# Every corpus file, as (target name, path).
CORPUS = [(name, path) for name in TARGETS for path in sorted((FUZZ / "corpus" / name).iterdir())]


# Every corpus file satisfies the contract of its target.
@pytest.mark.parametrize(("name", "path"), CORPUS, ids=[f"{n}/{p.name}" for n, p in CORPUS])
def test_corpus(name: str, path: Path) -> None:
    # Run the target; any exception outside the contract fails the test.
    load_target(name).test_one_input(path.read_bytes())


# Every regression input of the bugs found so far is in the corpus.
def test_regressions_present() -> None:
    # Names of the regression files.
    names = {p.stem for _, p in CORPUS if p.name.startswith("regression_")}
    # Seven bugs, seven inputs.
    assert len(names) == 7


# Random bytes and mutated corpus files satisfy the contract.
@pytest.mark.parametrize("name", TARGETS)
def test_random_inputs(name: str) -> None:
    # Target module.
    target = load_target(name)
    # Seeded generator.
    rng = np.random.default_rng(0)
    # Corpus files of the target.
    seeds = [p.read_bytes() for n, p in CORPUS if n == name]
    # A few hundred inputs.
    for _ in range(300):
        # A corpus file with a few random bytes replaced.
        data = bytearray(seeds[int(rng.integers(len(seeds)))])
        # Replace up to three bytes.
        for _ in range(int(rng.integers(1, 4))):
            # Position and new value.
            data[int(rng.integers(len(data)))] = int(rng.integers(32, 127))
        # Run the target.
        target.test_one_input(bytes(data))


# =============================================================================
# End of module tests/unit/test_fuzz_targets.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
