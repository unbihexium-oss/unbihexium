# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/check_python_style.py
# Title       : Python documentation style check
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only
# =============================================================================
#
# Abstract
# --------
# Enforces the documentation style of the Python sources of the repository
# (see CONTRIBUTING.md, "Python documentation style"):
#
#   header     the MPL-2.0 notice followed by the academic header block with
#              the "Project", "Module", "Title", "Author", "Affiliation",
#              "Copyright", "Licence" and "Python" fields; the Module field
#              must name the file
#   footer     the closing block "End of module <path>"
#   comments   `#` comments only: no docstrings in modules, classes or
#              functions
#   coverage   every line that holds code has a comment on the same line or
#              on the line directly above it; lines with only closing
#              brackets, and string continuation lines inside brackets, are
#              punctuation and prose rather than code
#
# Usage
# -----
#   python .github/scripts/check_python_style.py [paths ...]
#
# Without arguments, every tracked Python file listed in the STYLE_ROOTS
# directories is checked.
#
# Exit status
# -----------
#   0  every checked file follows the style
#   1  at least one violation was found
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Parse modules to find docstrings.
import ast

# Treat source text as a stream for the tokenizer.
import io

# Regular expressions for string and bracket lines.
import re

# Run `git ls-files` to list the tracked files.
import subprocess

# Access the command line arguments and set the exit status.
import sys

# Split source files into tokens to find code and comments.
import tokenize

# Represent file paths.
from pathlib import Path

# Directories whose Python files must follow the style. Directories are added
# here as they are converted to the style.
STYLE_ROOTS = (
    ".github/scripts/",  # Repository check scripts.
    "scripts/",  # Maintenance scripts.
    "examples/scripts/",  # Example scripts.
    "examples/serving/",  # Serving example.
    "src/unbihexium/zoo/",  # Model zoo.
    "src/unbihexium/ai/",  # Task APIs, inference, training and architectures.
    "src/unbihexium/cli/",  # Command line interface.
    "tests/unit/test_ai.py",  # Tests of the task APIs.
    "tests/unit/test_ai_data.py",  # Tests of the datasets.
    "tests/unit/test_ai_utils.py",  # Tests of decoding, metrics and transforms.
    "tests/unit/test_cli.py",  # Tests of the command line.
    "tests/unit/test_models.py",  # Tests of the architectures.
    "tests/unit/test_training.py",  # Tests of training.
    "tests/unit/test_zoo_catalog.py",  # Tests of the catalogue.
    "tests/unit/test_zoo_store.py",  # Tests of the model store.
)  # End of the directory list.

# Header fields that must appear in every file.
HEADER_FIELDS = (
    "Project",  # Project name.
    "Module",  # Repository path of the file.
    "Title",  # One-line title.
    "Author",  # Author and e-mail address.
    "Affiliation",  # Institution of the author.
    "Copyright",  # Copyright holder and years.
    "Licence",  # Licence of the file.
    "Python",  # Supported Python versions and requirements.
)  # End of the header fields.

# Token types that do not count as code.
NON_CODE = {
    tokenize.COMMENT,  # Comments.
    tokenize.NL,  # Line breaks inside statements and blank lines.
    tokenize.NEWLINE,  # Ends of statements.
    tokenize.INDENT,  # Indentation increases.
    tokenize.DEDENT,  # Indentation decreases.
    tokenize.ENDMARKER,  # End of the file.
}  # End of the non-code token types.


# Return every node that may carry a docstring and does.
def docstring_lines(tree: ast.AST) -> list[int]:
    # Line numbers of docstrings found.
    found = []
    # Visit every node of the syntax tree.
    for node in ast.walk(tree):
        # Only modules, classes and functions have docstrings.
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # A docstring is a string constant as the first statement.
            if node.body and isinstance(node.body[0], ast.Expr):
                # The expression of the first statement.
                value = node.body[0].value
                # Only string constants are docstrings.
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    # Record the line of the docstring.
                    found.append(node.body[0].lineno)
    # Return the line numbers.
    return found


# A line that holds only a string literal, as in implicit concatenation.
STRING_LINE = re.compile(r"""^\s*[rRbBfFuU]{0,2}("[^"]*"|'[^']*')\s*[,)\]}]*\s*$""")

# A line that holds only closing brackets and an optional comma.
CLOSING_LINE = re.compile(r"^\s*[)\]}]+,?\s*$")


# Whether a line starts a function or class definition.
def _is_definition(text: str) -> bool:
    # Definitions start with def, async def or class.
    return text.lstrip().startswith(("def ", "async def ", "class "))


# Whether the line follows a covered decorator that spans several lines.
def _after_decorator(lines: list[str], line: int, covered: set[int]) -> bool:
    # Walk back over the lines of the decorator.
    for previous in range(line - 1, 0, -1):
        # Text of the earlier line.
        text = lines[previous - 1].strip()
        # Blank and comment lines end the search.
        if not text or text.startswith("#"):
            # No decorator directly above.
            return False
        # The first line of a decorator decides.
        if text.startswith("@"):
            # Covered decorators cover their definition.
            return previous in covered
    # The start of the file was reached.
    return False


# Return the numbers of code lines without a comment.
def uncommented_lines(source: str) -> list[int]:
    # Lines that contain code tokens.
    code: set[int] = set()
    # Lines that contain a comment.
    comments: set[int] = set()
    # Bracket depth at the start of each line.
    depth_at: dict[int, int] = {}
    # Current bracket depth while scanning tokens.
    depth = 0
    # Tokenise the source.
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        # Remember the depth at the first token of each line.
        depth_at.setdefault(token.start[0], depth)
        # Record comment lines.
        if token.type == tokenize.COMMENT:
            # Line of the comment.
            comments.add(token.start[0])
        # Record lines where code tokens start.
        elif token.type not in NON_CODE:
            # Line of the code token.
            code.add(token.start[0])
        # Track the bracket depth.
        if token.type == tokenize.OP and token.string in "([{":
            # Opening bracket.
            depth += 1
        # Closing brackets reduce the depth.
        elif token.type == tokenize.OP and token.string in ")]}":
            # Closing bracket.
            depth -= 1
    # Physical lines of the file.
    lines = source.splitlines()
    # Lines whose comment rule is satisfied.
    covered: set[int] = set()
    # Visit code lines in order so that decorator chains propagate.
    for line in sorted(code):
        # Text of the line.
        text = lines[line - 1]
        # An inline comment covers the line.
        if line in comments or ((line - 1) in comments and (line - 1) not in code) or ((line - 1) in covered and lines[line - 2].lstrip().startswith("@")) or (_is_definition(text) and _after_decorator(lines, line, covered)) or CLOSING_LINE.match(text) or (depth_at.get(line, 0) > 0 and STRING_LINE.match(text)):
            # Mark the line as covered.
            covered.add(line)
    # A covered first decorator covers every decorator and the definition
    # of its chain, even when multi-line decorators sit in between.
    for group in _decorator_groups(source):
        # The chain is documented by the comment above its first decorator.
        if group[0] in covered:
            # Cover the start of every decorator and the definition line.
            covered.update(group)
    # Code lines that are not covered.
    return sorted(code - covered)


# Start lines of the decorators and the definition of decorated functions and classes.
def _decorator_groups(source: str) -> list[list[int]]:
    # Chains found in the file.
    groups = []
    # Visit every node of the syntax tree.
    for node in ast.walk(ast.parse(source)):
        # Decorated functions and classes.
        definition = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        # Only nodes with decorators form a chain.
        if isinstance(node, definition) and node.decorator_list:
            # Decorator start lines followed by the definition line.
            groups.append([d.lineno for d in node.decorator_list] + [node.lineno])
    # Return the chains.
    return groups


# Check one file and return its problems.
def check_file(path: Path) -> list[str]:
    # Problems found in this file.
    problems = []
    # Read the file.
    source = path.read_text(encoding="utf-8")
    # The first 60 lines hold the header.
    head = "\n".join(source.splitlines()[:60])
    # Every header field must be present.
    for name in HEADER_FIELDS:
        # Fields are written as "# Name <padding>: value".
        if f"# {name}" not in head:
            # Report the missing field.
            problems.append(f"{path}:1: header field {name!r} missing")
    # The Module field must name the file.
    if f": {path.as_posix()}" not in head:
        # Report the wrong or missing module path.
        problems.append(f"{path}:1: header Module field must be {path.as_posix()}")
    # The footer must close the file.
    if f"# End of module {path.as_posix()}" not in source:
        # Report the missing footer.
        problems.append(f"{path}:1: footer 'End of module {path.as_posix()}' missing")
    # Parse the file to find docstrings.
    for line in docstring_lines(ast.parse(source)):
        # Report each docstring.
        problems.append(f"{path}:{line}: docstring found; use # comments")
    # Find code lines without comments.
    for line in uncommented_lines(source):
        # Report each uncommented line.
        problems.append(f"{path}:{line}: code line without a comment")
    # Return the problems.
    return problems


# Return the tracked Python files under the style roots.
def default_files() -> list[Path]:
    # List tracked Python files.
    command = ["git", "ls-files", "-z", "--", "*.py"]
    # Run git and capture its output.
    out = subprocess.run(command, check=True, capture_output=True).stdout  # noqa: S603
    # Keep files under the style roots.
    return [Path(p) for p in out.decode().split("\0") if p and p.startswith(STYLE_ROOTS)]


# Check the files and return the exit status.
def main(argv: list[str]) -> int:
    # Explicit paths or the default set.
    files = [Path(a) for a in argv] or default_files()
    # All problems found.
    problems = [problem for path in files for problem in check_file(path)]
    # Report each problem as a GitHub Actions annotation.
    for problem in problems:
        # Split "path:line: message" into its parts.
        location, _, message = problem.partition(": ")
        # File and line of the problem.
        file, _, line = location.rpartition(":")
        # Annotation in the job log.
        print(f"::error file={file},line={line}::{message}")
    # Summarise.
    print(f"Checked {len(files)} Python files: {len(problems)} style problem(s).")
    # Fail when any problem was found.
    return 1 if problems else 0


# Run the check when the file is executed as a script.
if __name__ == "__main__":
    # Pass the command line arguments without the script name.
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module .github/scripts/check_python_style.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
