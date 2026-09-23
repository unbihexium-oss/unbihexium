# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate the example notebooks.

Every notebook must be valid nbformat 4 and must be committed without outputs
or execution counts, so that no data, credentials or local paths leak into the
repository and diffs stay reviewable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def main(root: str = "examples/notebooks") -> int:
    failures = 0
    notebooks = sorted(Path(root).glob("*.ipynb"))
    for path in notebooks:
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
        except Exception as exc:  # noqa: BLE001 - report every validation problem
            print(f"::error file={path}::Invalid notebook: {exc}")
            failures += 1
            continue
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                print(f"::error file={path}::Cell {index} contains outputs or an execution count; clear outputs before committing")
                failures += 1
    if failures:
        print(f"{failures} problem(s) found in {len(notebooks)} notebooks.")
        return 1
    print(f"All {len(notebooks)} notebooks are valid and have no outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
