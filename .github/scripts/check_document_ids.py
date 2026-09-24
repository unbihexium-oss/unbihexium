# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_document_ids.py
# Title       : Document identifier and register check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Checks the document identifiers against the rules of the Document Register
# (docs/document_register.md, UBX-DOC-302):
#
#   format     every controlled Markdown document has a control table whose
#              Document row holds an identifier UBX-DOC-SNN with a series
#              digit S from 1 to 9
#   unique     no identifier is used twice, and none is withdrawn
#   register   Section 3 of the register lists exactly the controlled
#              documents, each under the series of its identifier, with the
#              title (first-level heading) and the file of the document
#   header     the Title field of the header comment equals the heading
#   contents   the identifiers and titles in docs/toc.md match the documents
#
# Controlled documents are the tracked Markdown files except those under
# .github/ and the generated model cards (model_zoo/cards/ and
# model_zoo/MODEL_CARDS.md).
#
# Usage
# -----
#   python .github/scripts/check_document_ids.py
#
# The script must run from the repository root, because it lists files with
# `git ls-files` and reads the register and the table of contents from there.
#
# Exit status
# -----------
#   0  every rule holds
#   1  at least one problem was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Resolve relative links without touching the file system.
import posixpath

# Parse identifiers, headings and table rows.
import re

# Run `git ls-files` to list the tracked files.
import subprocess

# Set the exit status of the script.
import sys

# Represent and read file paths.
from pathlib import Path

# The register that defines the scheme and lists every document.
REGISTER = "docs/document_register.md"

# The table of contents of docs/.
TOC = "docs/toc.md"

# Directories whose Markdown files are not controlled documents.
EXCLUDED_DIRS = (
    ".github/",  # Pull request template and other GitHub files.
    "model_zoo/cards/",  # Model cards generated from the catalogue.
)  # End of the excluded directories.

# Single Markdown files that are not controlled documents.
EXCLUDED_FILES = {"model_zoo/MODEL_CARDS.md"}

# Identifier format: prefix, series digit 1 to 9 and two digits.
ID_FORMAT = re.compile(r"UBX-DOC-[1-9][0-9]{2}")

# Document row of a control table.
DOCUMENT_ROW = re.compile(r"^\| Document \| (.+?) \|$", re.MULTILINE)

# Title field of the header comment.
TITLE_FIELD = re.compile(r"^Title +: (.+)$", re.MULTILINE)

# First-level heading.
HEADING = re.compile(r"^# (.+)$", re.MULTILINE)

# Series subsection of Section 3 of the register.
SERIES_HEADING = re.compile(r"^### 3\.\d+ Series (\d): ")

# Entry of the register: identifier, linked title and file.
REGISTER_ROW = re.compile(r"^\| (UBX-DOC-\S+) \| \[(.+)\]\(([^)]+)\) \| `([^`]+)` \|$")

# Entry of Section 4 of the register: a withdrawn identifier.
WITHDRAWN_ROW = re.compile(r"^\| (UBX-DOC-\S+) \| ")

# Entry of the table of contents: linked title and identifier.
TOC_ROW = re.compile(r"^\| \[(.+)\]\(([^)]+\.md)\) \| (UBX-DOC-\S+) \|$")


# Tracked Markdown files that are controlled documents.
def controlled_documents() -> list[str]:
    # -z separates the file names with NUL bytes, which is safe for any name.
    command = ["git", "ls-files", "-z", "--", "*.md"]
    # Run git and keep its output.
    out = subprocess.run(command, check=True, capture_output=True).stdout
    # Split the output at NUL bytes and drop the empty trailing entry.
    names = [name for name in out.decode().split("\0") if name]
    # Keep the files outside the exclusions.
    return sorted(
        name  # Path relative to the repository root.
        for name in names  # Every tracked Markdown file.
        if not name.startswith(EXCLUDED_DIRS) and name not in EXCLUDED_FILES  # Not excluded.
    )  # End of the controlled documents.


# Identifier, heading and header title of a document; None where missing.
def document_fields(path: str) -> tuple[str | None, str | None, str | None]:
    # Text of the document.
    text = Path(path).read_text(encoding="utf-8")
    # First-level heading.
    heading = HEADING.search(text)
    # Without a heading there is no control table either.
    if heading is None:
        # Nothing found.
        return None, None, None
    # The control table follows the heading, before the first section.
    head = text[heading.end() :].split("\n## ", 1)[0]
    # Document row of the control table.
    row = DOCUMENT_ROW.search(head)
    # Title field of the header comment.
    title = TITLE_FIELD.search(text[: heading.start()])
    # The three values.
    return (
        row.group(1) if row else None,  # Identifier.
        heading.group(1),  # Heading.
        title.group(1) if title else None,  # Header title.
    )  # End of the fields.


# Lines of the register grouped by the number of their "## N." section.
def register_sections(text: str) -> dict[str, list[str]]:
    # Lines by section number.
    sections: dict[str, list[str]] = {}
    # Section of the current line; lines before the first section are skipped.
    current = ""
    # Walk over the lines.
    for line in text.splitlines():
        # A second-level heading starts a new section.
        if line.startswith("## "):
            # Number before the dot, or empty for unnumbered sections.
            current = line[3:].split(".", 1)[0] if line[3:4].isdigit() else ""
            # Next line.
            continue
        # Collect the line.
        sections.setdefault(current, []).append(line)
    # Every section with its lines.
    return sections


# Run all checks and return the exit status.
def main() -> int:
    # Problems found so far.
    problems: list[str] = []

    # Report a problem as a GitHub annotation and remember it.
    def report(path: str, message: str) -> None:
        # Annotation on the file.
        print(f"::error file={path}::{message}")
        # Count it.
        problems.append(message)

    # Identifier and heading of every controlled document, by path.
    documents: dict[str, tuple[str, str]] = {}
    # Read every controlled document.
    for path in controlled_documents():
        # Fields of the document.
        ident, heading, title = document_fields(path)
        # A controlled document needs a control table.
        if ident is None or heading is None:
            # Report the missing table.
            report(path, "no document control table with a Document row after the heading")
            # Next document.
            continue
        # The identifier must have the register format.
        if not ID_FORMAT.fullmatch(ident):
            # Report the format error.
            report(path, f"identifier {ident} does not match UBX-DOC-SNN with S from 1 to 9")
        # The header title must equal the heading.
        if title is not None and title != heading:
            # Report the mismatch.
            report(path, f"header Title {title!r} differs from the heading {heading!r}")
        # Remember the document.
        documents[path] = (ident, heading)

    # First document of every identifier.
    owners: dict[str, str] = {}
    # Look for identifiers used twice.
    for path, (ident, _) in documents.items():
        # A second use is a problem.
        if ident in owners:
            # Report both files.
            report(path, f"identifier {ident} is also used by {owners[ident]}")
        # Otherwise remember the owner.
        else:
            # First use.
            owners[ident] = path

    # Sections of the register.
    sections = register_sections(Path(REGISTER).read_text(encoding="utf-8"))
    # Register entries by file: identifier and title.
    listed: dict[str, tuple[str, str]] = {}
    # Series digit of the current subsection of Section 3.
    series = None
    # Walk over Section 3.
    for line in sections.get("3", []):
        # A series subsection.
        heading = SERIES_HEADING.match(line)
        # Remember its digit.
        if heading:
            # Series of the following entries.
            series = heading.group(1)
            # Next line.
            continue
        # An entry of the register.
        row = REGISTER_ROW.match(line)
        # Other lines are prose.
        if row is None:
            # Next line.
            continue
        # Identifier, title, link and file of the entry.
        ident, title, link, file = row.groups()
        # The identifier must belong to the series of its subsection.
        if series is None or not ident.startswith(f"UBX-DOC-{series}"):
            # Report the misplaced entry.
            report(REGISTER, f"{ident} is listed under series {series}")
        # The link must point to the listed file.
        if posixpath.normpath(posixpath.join("docs", link)) != file:
            # Report the wrong link.
            report(REGISTER, f"the link of {ident} points to {link}, not to {file}")
        # A file must be listed once.
        if file in listed:
            # Report the duplicate.
            report(REGISTER, f"{file} is listed more than once")
        # Remember the entry.
        listed[file] = (ident, title)
    # Withdrawn identifiers from Section 4.
    withdrawn = {m.group(1) for m in map(WITHDRAWN_ROW.match, sections.get("4", [])) if m}

    # Compare the documents with the register.
    for path, (ident, heading) in documents.items():
        # A withdrawn identifier must not be in use.
        if ident in withdrawn:
            # Report the reuse.
            report(path, f"identifier {ident} is withdrawn")
        # Every document must be listed.
        if path not in listed:
            # Report the missing entry.
            report(REGISTER, f"{path} ({ident}) is missing from Section 3")
            # Next document.
            continue
        # Identifier and title of the entry.
        entry_ident, entry_title = listed[path]
        # The identifiers must agree.
        if entry_ident != ident:
            # Report the mismatch.
            report(REGISTER, f"{path} is listed as {entry_ident}, but its table says {ident}")
        # The titles must agree.
        if entry_title != heading:
            # Report the mismatch.
            report(REGISTER, f"{path} is listed as {entry_title!r}, but its heading is {heading!r}")
    # Every entry must be a controlled document.
    for file in sorted(set(listed) - set(documents)):
        # Report the stale entry.
        report(REGISTER, f"{file} is listed but is not a controlled document")

    # Compare the table of contents with the documents.
    for line in Path(TOC).read_text(encoding="utf-8").splitlines():
        # An entry of the table of contents.
        row = TOC_ROW.match(line)
        # Other lines are prose.
        if row is None:
            # Next line.
            continue
        # Title, link and identifier of the entry.
        title, link, ident = row.groups()
        # File of the entry, relative to the repository root.
        path = posixpath.normpath(posixpath.join("docs", link))
        # The file must be a controlled document.
        if path not in documents:
            # Report the unknown file.
            report(TOC, f"{link} is not a controlled document")
            # Next line.
            continue
        # Identifier and heading of the document.
        doc_ident, doc_heading = documents[path]
        # The identifier and title must equal those of the document.
        if (ident, title) != (doc_ident, doc_heading):
            # Report the mismatch.
            report(TOC, f"{link} is listed as {ident} {title!r}, not {doc_ident} {doc_heading!r}")

    # Summarise and fail when any problem was found.
    if problems:
        # Print the number of problems.
        print(f"{len(problems)} document identifier problem(s) found.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm that every rule holds.
    print(f"{len(documents)} controlled documents carry valid identifiers that match the register.")
    # Zero exit status marks success.
    return 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main())

# =============================================================================
# End of module .github/scripts/check_document_ids.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
