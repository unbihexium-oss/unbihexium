# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Enforce the repository writing rules on every tracked text file.

The repository is written in English only and must not contain emojis or
em dashes. The check reports every offending line and exits non-zero if any
violation is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "emoji or pictographic symbol",
        re.compile(
            "["
            "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, supplemental symbols
            "\u2600-\u27bf"  # miscellaneous symbols and dingbats
            "\u2b00-\u2bff"  # miscellaneous symbols and arrows (stars, circles)
            "\ufe0f"  # emoji variation selector
            "\u200d"  # zero width joiner used in emoji sequences
            "]"
        ),
    ),
    ("em dash or horizontal bar", re.compile("[\u2014\u2015]")),
    ("Turkish-specific letter (the repository is English only)", re.compile("[\u011e\u011f\u0130\u0131\u015e\u015f]")),
]

# Files whose content is defined by a third party and must stay verbatim.
EXCLUDED = {"LICENSE.txt", "LICENSES/MPL-2.0.txt"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def main() -> int:
    violations = 0
    for path in tracked_files():
        if path.as_posix() in EXCLUDED or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary file
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in RULES:
                match = pattern.search(line)
                if match:
                    violations += 1
                    char = match.group(0)
                    print(f"::error file={path},line={lineno}::{name}: U+{ord(char):04X}")
    if violations:
        print(f"{violations} text policy violation(s) found.")
        return 1
    print("Text policy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
