# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/utils/seeding.py
# Title       : Reproducible random number generation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires NumPy; seeds PyTorch when it
#               is installed and requested
# =============================================================================
#
# Abstract
# --------
# set_seed() seeds every random number generator a workflow may use: the
# Python `random` module, the legacy NumPy global generator and, when it is
# already imported or explicitly requested, PyTorch (CPU and CUDA). It
# returns a fresh numpy.random.Generator (PCG64) for new code, which should
# pass generators explicitly instead of relying on global state.
#
# derive_seed() derives independent child seeds from a base seed and a key
# (for example a tile index or a worker id) with SHA-256, so that results do
# not depend on the order in which tiles or workers run. spawn_generators()
# uses NumPy's SeedSequence spawning, which guarantees statistically
# independent streams.
#
# References
# ----------
# O'Neill, M. E. (2014). PCG: A family of simple fast space-efficient
# statistically good algorithms for random number generation. Technical
# Report HMC-CS-2014-0905, Harvey Mudd College.
# Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011).
# Parallel random numbers: as easy as 1, 2, 3. Proceedings of SC11, ACM.
# doi:10.1145/2063384.2063405
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Seed derivation.
import hashlib

# Python's own generator.
import random

# Detect an imported PyTorch.
import sys

# Arrays and generators.
import numpy as np

# Largest seed accepted by the legacy NumPy generator (2**32 - 1).
MAX_SEED = 2**32 - 1


# Validate a seed and return it as an int.
def _check_seed(seed: int) -> int:
    # Seeds must be integers (booleans are rejected).
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        # Explain the problem.
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}")
    # Seeds must fit the legacy NumPy generator.
    if not 0 <= int(seed) <= MAX_SEED:
        # Explain the range.
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    # Plain integer.
    return int(seed)


# Seed Python, NumPy and optionally PyTorch; return a NumPy generator.
def set_seed(
    seed: int,  # Seed in [0, 2**32 - 1].
    torch: bool | None = None,  # True: seed PyTorch; None: only when already imported.
    deterministic: bool = False,  # Ask PyTorch for deterministic algorithms.
) -> np.random.Generator:  # Fresh PCG64 generator seeded with `seed`.
    # Validate the seed.
    seed = _check_seed(seed)
    # Python's generator.
    random.seed(seed)
    # Legacy NumPy global generator, still used by some dependencies.
    np.random.seed(seed)
    # Seed PyTorch when requested or already in use.
    if torch or (torch is None and "torch" in sys.modules):
        # Imported lazily because PyTorch is optional and slow to import.
        import torch as _torch

        # CPU and every CUDA device.
        _torch.manual_seed(seed)
        # Deterministic kernels, which may be slower.
        if deterministic:
            # Error out on non-deterministic operations instead of silently varying.
            _torch.use_deterministic_algorithms(True, warn_only=True)
    # Generator for new code.
    return np.random.default_rng(seed)


# Child seed derived from a base seed and keys, independent of call order.
def derive_seed(seed: int, *keys: object) -> int:
    # Validate the base seed.
    seed = _check_seed(seed)
    # Text of the base seed and the keys, separated so that ("1", "2") != ("12",).
    text = "\x1f".join([str(seed), *(repr(k) for k in keys)])
    # First eight bytes of the SHA-256 digest.
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    # Reduce to the accepted seed range.
    return int.from_bytes(digest, "big") % (MAX_SEED + 1)


# Independent generators for parallel workers.
def spawn_generators(seed: int, count: int) -> list[np.random.Generator]:
    # At least one generator.
    if count < 1:
        # Explain the problem.
        raise ValueError(f"count must be positive, got {count}")
    # Root of the seed tree.
    root = np.random.SeedSequence(_check_seed(seed))
    # One generator per child sequence.
    return [np.random.default_rng(child) for child in root.spawn(count)]


# =============================================================================
# End of module src/unbihexium/utils/seeding.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
