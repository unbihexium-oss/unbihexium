# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_text_policy.py
# Title       : Repository text policy check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Enforces the repository writing rules on every tracked text file. The
# repository is written in English only and must not contain emojis or em
# dashes. The check reports every offending line with the code point it found
# and exits with a non-zero status if any violation exists. Binary files and
# files that are not valid UTF-8 are skipped.
#
# Usage
# -----
#   python .github/scripts/check_text_policy.py
#
# The script must run from the repository root, because it lists files with
# `git ls-files`.
#
# Exit status
# -----------
#   0  no violation was found
#   1  at least one violation was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Character class patterns for the forbidden characters.
import re

# Run `git ls-files` to list the tracked files.
import subprocess

# Set the exit status of the script.
import sys

# Represent and read file paths.
from pathlib import Path

# Each rule pairs a human-readable name with a pattern of forbidden characters.
RULES: list[tuple[str, re.Pattern[str]]] = [
    # Rule 1: emojis and pictographic symbols.
    (
        # Name printed in the error message.
        "emoji or pictographic symbol",
        # One character class assembled from the code point ranges below.
        re.compile(
            # Start of the character class.
            "["
            "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, supplemental symbols
            "\u2600-\u27bf"  # miscellaneous symbols and dingbats
            "\u2b00-\u2bff"  # miscellaneous symbols and arrows (stars, circles)
            "\ufe0f"  # emoji variation selector
            "\u200d"  # zero width joiner used in emoji sequences
            # End of the character class.
            "]"
        ),  # End of the emoji pattern.
    ),  # End of rule 1.
    # Rule 2: the em dash (U+2014) and the horizontal bar (U+2015).
    ("em dash or horizontal bar", re.compile("[\u2014\u2015]")),
    # Rule 3: letters used only in Turkish, which signal non-English text.
    ("Turkish-specific letter (the repository is English only)", re.compile("[\u011e\u011f\u0130\u0131\u015e\u015f]")),
]  # End of the rule list.

# Files whose content is defined by a third party and must stay verbatim.
EXCLUDED = {"LICENSE.txt"}


# Return every file tracked by git.
def tracked_files() -> list[Path]:
    # -z separates the file names with NUL bytes, which is safe for any name.
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    # Split the output at NUL bytes and drop the empty trailing entry.
    return [Path(p) for p in out.decode().split("\0") if p]


# Check every tracked file and return the exit status.
def main() -> int:
    # Number of violations found so far.
    violations = 0
    # Check each tracked file in turn.
    for path in tracked_files():
        # Skip excluded files and paths that are not regular files (for
        # example a deleted file that is still in the index).
        if path.as_posix() in EXCLUDED or not path.is_file():
            # Continue with the next file.
            continue
        # Read the raw bytes to detect binary content before decoding.
        data = path.read_bytes()
        # A NUL byte in the first 8 KiB marks a binary file.
        if b"\0" in data[:8192]:
            continue  # binary file
        # Decode the file as UTF-8 text.
        try:
            # Text files in the repository are UTF-8.
            text = data.decode("utf-8")
        # Files that are not valid UTF-8 are not text files for this check.
        except UnicodeDecodeError:
            # Continue with the next file.
            continue
        # Inspect every line with its 1-based line number.
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Apply every rule to the line.
            for name, pattern in RULES:
                # Find the first forbidden character, if any.
                match = pattern.search(line)
                # Report the line when a forbidden character is present.
                if match:
                    # Count the violation.
                    violations += 1
                    # The offending character.
                    char = match.group(0)
                    # Annotate the file and line with the rule and code point.
                    print(f"::error file={path},line={lineno}::{name}: U+{ord(char):04X}")
    # Summarise and fail when any violation was found.
    if violations:
        # Print the number of violations.
        print(f"{violations} text policy violation(s) found.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm that the check passed.
    print("Text policy check passed.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main())

# =============================================================================
# End of module .github/scripts/check_text_policy.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
