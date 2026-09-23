# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_config_style.py
# Title       : Documentation style check for configuration and data files
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Applies the documentation style of the Python sources (see
# check_python_style.py) to every other tracked text file that can hold `#`
# comments: YAML, TOML, INI, the Dockerfile, the Makefile, shell scripts, lock
# files and the Git, Docker and editor configuration files.
#
#   header     the MPL-2.0 notice followed by the academic header block with
#              the "Project", "File", "Title", "Author", "Affiliation",
#              "Copyright", "Licence" and "Format" fields and an "Abstract"
#              section; the File field must name the file; only a shebang or
#              a Dockerfile parser directive may precede the notice
#   footer     the closing block "End of file <path>"
#   coverage   every line that holds content has a comment on the same line
#              or on the line directly above it
#
# Same-line comments are accepted only in formats whose parsers strip them
# (YAML, TOML, shell, pip requirements and the model checksum list). In the
# other formats a trailing `#` would become part of the value, so the comment
# must stand on the line above. Lines that continue the previous line (after
# a trailing backslash), lines with only closing brackets, the contents of
# YAML block scalars and TOML multi-line strings (prose, except shell scripts
# under a `run` key) and the continuation lines of INI values are covered by
# the comment of the line that opens them.
#
# Files that cannot hold comments are not checked: JSON, NumPy arrays, the
# empty py.typed marker, the Markdown and notebook documentation (own rules)
# and the licence and notice texts, which are reproduced verbatim.
#
# Usage
# -----
#   python .github/scripts/check_config_style.py [paths ...]
#
# Exit status
# -----------
#   0 when every file follows the style, 1 otherwise.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Regular expressions for block scalars, keys and brackets.
import re

# List the tracked files.
import subprocess

# Command line arguments and exit status.
import sys

# Represent file paths.
from pathlib import Path

# Header fields that every file must carry, in this order.
HEADER_FIELDS = (
    "Project",  # Project name.
    "File",  # Repository path of the file.
    "Title",  # One-line title.
    "Author",  # Author and e-mail address.
    "Affiliation",  # Institution of the author.
    "Copyright",  # Copyright holder and years.
    "Licence",  # Licence of the file.
    "Format",  # File format and the tool that reads it.
)  # End of the header fields.

# First line of the MPL-2.0 notice.
NOTICE_LINE = "# This Source Code Form is subject to the terms of the Mozilla Public"

# Files and patterns that are not checked, with the reason in the comment.
SKIPPED = (
    re.compile(r"\.(md|ipynb)$"),  # Documentation, own rules.
    re.compile(r"\.py$"),  # Python, checked by check_python_style.py.
    re.compile(r"\.json$"),  # JSON has no comment syntax.
    re.compile(r"\.npy$"),  # Binary NumPy arrays.
    re.compile(r"^LICENSE\.txt$"),  # Licence text, reproduced verbatim.
    re.compile(r"^LICENSES/"),  # Licence texts for REUSE, verbatim.
    re.compile(r"^NOTICE$"),  # Legal notice text, reproduced verbatim.
    re.compile(r"(^|/)py\.typed$"),  # Empty PEP 561 marker.
)  # End of the skipped patterns.


# Syntax family of a file: yaml, toml, ini, shell, inline or line.
def kind_of(path: str) -> str:
    # Base name of the file.
    name = path.rsplit("/", 1)[-1]
    # YAML files, including the Citation File Format.
    if name.endswith((".yml", ".yaml", ".cff")):
        # Same-line comments allowed, block scalars exempt.
        return "yaml"
    # TOML files.
    if name.endswith(".toml"):
        # Same-line comments allowed, multi-line strings exempt.
        return "toml"
    # INI files read by configparser.
    if name.endswith(".ini"):
        # Comments on their own line, value continuations exempt.
        return "ini"
    # Shell scripts.
    if name.endswith((".sh", ".bash")):
        # Same-line comments allowed.
        return "shell"
    # pip requirements and the model checksum list.
    if name.startswith("requirements") or name == "checksums.txt":
        # Same-line comments allowed.
        return "inline"
    # Dockerfile, Makefile and the Git, Docker and editor files.
    return "line"


# True when the line is a comment and nothing else.
def is_comment(text: str) -> bool:
    # Leading whitespace does not matter.
    return text.lstrip().startswith("#")


# Position of a same-line comment outside quotes, or -1.
def inline_comment(text: str, shell: bool = True) -> int:
    # Currently open quote character, if any.
    quote = ""
    # Walk over the characters.
    for i, char in enumerate(text):
        # Text before the character, without trailing blanks.
        before = text[:i].rstrip()
        # Inside a quoted string only its closing quote matters.
        if quote:
            # The string ends here.
            if char == quote:
                # Leave the string.
                quote = ""
        # A quote opens a string; in YAML and TOML only at the start of a value,
        # so that an apostrophe inside a plain scalar is text.
        elif char in "\"'" and (shell or not before or before[-1] in ":[{,-="):
            # Remember which quote closes it.
            quote = char
        # A hash after whitespace starts a comment.
        elif char == "#" and i > 0 and text[i - 1] in " \t":
            # Position of the comment.
            return i
    # No comment on this line.
    return -1


# Lines with only closing brackets are punctuation.
CLOSING_LINE = re.compile(r"^\s*[)\]}]+,?\s*$")

# A YAML key or sequence item that opens a block scalar.
BLOCK_SCALAR = re.compile(r"^(\s*)(?:-\s+)?([\w.-]+\s*:\s*)?[|>][0-9+-]*\s*(#.*)?$")


# Line numbers covered by YAML block scalars; shell blocks stay checked.
def yaml_exempt(lines: list[str]) -> tuple[set[int], set[int]]:
    # Prose lines that need no comment.
    prose: set[int] = set()
    # Shell lines checked like a shell script.
    shell: set[int] = set()
    # Index of the current line.
    i = 0
    # Walk over the lines.
    while i < len(lines):
        # Does the line open a block scalar?
        match = BLOCK_SCALAR.match(lines[i])
        # Only lines ending in a block indicator open one.
        if match and not is_comment(lines[i]):
            # Indentation of the key.
            indent = len(match.group(1))
            # Scripts are code; everything else is prose.
            target = shell if (match.group(2) or "").split(":")[0].strip() == "run" else prose
            # First content line.
            j = i + 1
            # The scalar ends at the first non-blank line indented no deeper.
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
                # Line numbers are one-based.
                target.add(j + 1)
                # Next line.
                j += 1
            # Continue after the scalar.
            i = j
        # Not a block scalar.
        else:
            # Next line.
            i += 1
    # Both sets.
    return prose, shell


# Line numbers inside TOML multi-line strings.
def toml_exempt(lines: list[str]) -> set[int]:
    # Lines inside a string.
    inside: set[int] = set()
    # Delimiter of the open string, if any.
    open_delim = ""
    # Walk over the lines.
    for number, text in enumerate(lines, start=1):
        # Content line of an open string.
        if open_delim:
            # Exempt it.
            inside.add(number)
            # The string closes on this line.
            if open_delim in text:
                # No open string any more.
                open_delim = ""
            # Next line.
            continue
        # Check both delimiters.
        for delim in ('"""', "'''"):
            # An odd count opens a string that continues on the next line.
            if text.count(delim) % 2 == 1:
                # Remember the delimiter.
                open_delim = delim
    # Exempt lines.
    return inside


# Line numbers that continue an INI value on an indented line.
def ini_exempt(lines: list[str]) -> set[int]:
    # Continuation lines.
    inside: set[int] = set()
    # Is a value open that may continue?
    in_value = False
    # Walk over the lines.
    for number, text in enumerate(lines, start=1):
        # Blank and comment lines do not end a value.
        if not text.strip() or is_comment(text):
            # Next line.
            continue
        # An indented line continues the open value.
        if in_value and text[0] in " \t":
            # Exempt it.
            inside.add(number)
        # A section header or a new key.
        else:
            # A key opens a value; a section header does not.
            in_value = not text.startswith("[")
    # Exempt lines.
    return inside


# Line numbers without a comment on the line or on the line above.
def uncommented_lines(path: str, lines: list[str]) -> list[int]:
    # Syntax family.
    kind = kind_of(path)
    # Lines covered by the syntax of the format.
    exempt: set[int] = set()
    # Lines that follow shell rules inside another format.
    shell: set[int] = set()
    # YAML block scalars.
    if kind == "yaml":
        # Prose and scripts.
        exempt, shell = yaml_exempt(lines)
    # TOML multi-line strings.
    elif kind == "toml":
        # Prose.
        exempt = toml_exempt(lines)
    # INI value continuations.
    elif kind == "ini":
        # Continuations.
        exempt = ini_exempt(lines)
    # Formats whose parsers strip same-line comments.
    inline_ok = kind in {"yaml", "toml", "shell", "inline"}
    # Offending lines.
    missing = []
    # Walk over the lines.
    for number, text in enumerate(lines, start=1):
        # Blank lines, comment lines and exempt lines need nothing.
        if not text.strip() or is_comment(text) or number in exempt:
            # Next line.
            continue
        # Previous line, if any.
        previous = lines[number - 2] if number > 1 else ""
        # A comment on the line above covers this line.
        if is_comment(previous):
            # Covered.
            continue
        # A continuation line is covered by the line that opens it.
        if previous.rstrip().endswith("\\") and not is_comment(previous):
            # Covered.
            continue
        # Closing brackets are punctuation.
        if CLOSING_LINE.match(text):
            # Covered.
            continue
        # A same-line comment, where the format allows one.
        if (inline_ok or number in shell) and inline_comment(text, kind == "shell" or number in shell) >= 0:
            # Covered.
            continue
        # A Makefile rule may carry its help text after "##".
        if path.endswith("Makefile") and re.match(r"^[\w.-]+:.*##", text):
            # Covered.
            continue
        # Nothing covers the line.
        missing.append(number)
    # Offending lines.
    return missing


# Problems of one file.
def check_file(path: Path) -> list[str]:
    # Repository path with forward slashes.
    name = path.as_posix()
    # File contents.
    lines = path.read_text(encoding="utf-8").splitlines()
    # Problems found.
    problems = []
    # Lines before the notice: a shebang or a Dockerfile parser directive.
    start = 1 if lines and (lines[0].startswith("#!") or lines[0].startswith("# syntax=")) else 0
    # The notice opens the file.
    if len(lines) <= start or lines[start] != NOTICE_LINE:
        # Report it.
        problems.append(f"{name}:{start + 1}: MPL-2.0 notice missing at the top")
    # Header block.
    head = "\n".join(lines[:60])
    # Every header field.
    for field in HEADER_FIELDS:
        # Field in the "# Name  : value" layout.
        if not re.search(rf"^# {field}\s+:", head, re.MULTILINE):
            # Report it.
            problems.append(f"{name}:1: header field {field!r} missing")
    # The File field names the file.
    if not re.search(rf"^# File\s+: {re.escape(name)}$", head, re.MULTILINE):
        # Report it.
        problems.append(f"{name}:1: header File field must be {name}")
    # The abstract follows the header.
    if not re.search(r"^# Abstract$", head, re.MULTILINE):
        # Report it.
        problems.append(f"{name}:1: header section 'Abstract' missing")
    # Footer block.
    tail = "\n".join(lines[-6:])
    # Closing line of the footer.
    if f"# End of file {name}" not in tail or "# Cite the project as described in CITATION.cff." not in tail:
        # Report it.
        problems.append(f"{name}:{len(lines)}: footer 'End of file {name}' missing")
    # Line coverage.
    for number in uncommented_lines(name, lines):
        # Report each line.
        problems.append(f"{name}:{number}: line without a comment")
    # All problems.
    return problems


# Tracked files that the style applies to.
def default_files() -> list[Path]:
    # Every tracked file.
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout  # noqa: S607
    # Names of the tracked files.
    names = [p for p in out.decode().split("\0") if p]
    # Keep the files that are not skipped.
    return [Path(p) for p in names if not any(pattern.search(p) for pattern in SKIPPED)]


# Check the given files, or every tracked file, and report the problems.
def main(argv: list[str]) -> int:
    # Files to check.
    files = [Path(a) for a in argv] or default_files()
    # Problems of all files.
    problems = [problem for path in files for problem in check_file(path)]
    # Report in the GitHub Actions annotation format.
    for problem in problems:
        # Split "path:line: message".
        location, _, message = problem.partition(": ")
        # Split "path:line".
        file, _, line = location.rpartition(":")
        # Annotation.
        print(f"::error file={file},line={line}::{message}")
    # Summary.
    print(f"Checked {len(files)} configuration files: {len(problems)} style problem(s).")
    # Exit status.
    return 1 if problems else 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status of the check.
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module .github/scripts/check_config_style.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
