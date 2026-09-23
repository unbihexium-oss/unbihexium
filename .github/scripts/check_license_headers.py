# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check that every tracked source file carries the MPL-2.0 Exhibit A notice.

The notice must appear at the top of the file, after an optional shebang and
encoding line. The script also checks that LICENSE.txt is the MPL-2.0 text.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

NOTICE = (
    "# This Source Code Form is subject to the terms of the Mozilla Public\n"
    "# License, v. 2.0. If a copy of the MPL was not distributed with this\n"
    "# file, You can obtain one at https://mozilla.org/MPL/2.0/.\n"
)
PATTERNS = ["*.py", "*.bash", "*.sh"]
HEAD_LINES = 5


def tracked(patterns: list[str]) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *patterns], check=True, capture_output=True).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def main() -> int:
    failures = 0

    licence = Path("LICENSE.txt").read_text(encoding="utf-8")
    if not licence.startswith("Mozilla Public License Version 2.0"):
        print("::error file=LICENSE.txt::LICENSE.txt does not contain the Mozilla Public License 2.0 text")
        failures += 1

    files = tracked(PATTERNS)
    for path in files:
        head = "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:HEAD_LINES])
        if NOTICE not in head:
            print(f"::error file={path},line=1::Missing MPL-2.0 notice at the top of the file")
            failures += 1

    if failures:
        print(f"{failures} licence header problem(s) found in {len(files)} source files.")
        return 1
    print(f"All {len(files)} source files carry the MPL-2.0 notice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
