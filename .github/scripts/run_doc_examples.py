# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : .github/scripts/run_doc_examples.py
# Title       : Execution of the code examples of the documentation
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, standard library only (the examples
#               need Unbihexium with its extras installed)
# =============================================================================
#
# Abstract
# --------
# Runs the code examples of the controlled Markdown documents against the
# installed package, so that a change of the library that breaks an example
# fails a check:
#
#   python   every fenced block with the info string "python"
#   bash     every fenced block with the info string "bash" whose commands
#            are all `unbihexium` commands (optionally preceded by variable
#            assignments), except `unbihexium serve`, which does not return
#
# The blocks of one document run in order, in one Python namespace and one
# temporary working directory, with UNBIHEXIUM_CACHE pointing into it, in a
# separate process per document. Two comments on the line before an opening
# fence change what happens to the block:
#
#   <!-- doc-example: skip (reason) -->   the block does not run, for example
#                                          because it needs network access, a
#                                          running server or the repository
#                                          checkout; the reason is required
#   <!-- doc-example: write NAME -->      the content of the block, of any
#                                          language, is written to the file
#                                          NAME when the examples reach it
#
# Usage
# -----
#   python .github/scripts/run_doc_examples.py [--list] [documents ...]
#
# Without documents, every controlled Markdown document is used. --list
# prints the blocks that would run and the skipped ones, and runs nothing.
#
# Exit status
# -----------
#   0  every example ran without an error
#   1  at least one example failed or a skip comment has no reason
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Command line options.
import argparse

# Environment of the example processes.
import os

# Parse fences, skip comments and commands.
import re

# Split shell lines into words.
import shlex

# Run git, the example processes and the shell blocks.
import subprocess

# Interpreter path and exit status.
import sys

# Temporary working directories.
import tempfile

# Duration of each document.
import time

# Tracebacks of failed Python blocks.
import traceback

# Represent and read file paths.
from pathlib import Path

# Directories whose Markdown files are not controlled documents.
EXCLUDED_DIRS = (".github/", "model_zoo/cards/")

# Fenced code block: info string and content, with the line of the fence.
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

# Skip comment on the line before a fence.
SKIP = re.compile(r"^<!-- doc-example: skip(?: \((.+)\))? -->$")

# Write comment on the line before a fence, with a relative file name.
WRITE = re.compile(r"^<!-- doc-example: write ([\w./-]+) -->$")

# Variable assignment before a command, for example UNBIHEXIUM_CONFIG=x.yaml.
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Seconds that the examples of one document may take.
TIMEOUT = 900


# One code block of a document.
class Block:
    # Store the fields of the block.
    def __init__(
        self,  # The block.
        line: int,  # Line of the opening fence.
        language: str,  # "python", "bash" or "file".
        code: str,  # Content.
        skip: str | None,  # Skip reason, if any.
        target: str = "",  # File name of a write comment.
    ) -> None:  # The constructor returns nothing.
        # Line of the opening fence, one-based.
        self.line = line
        # Info string: "python" or "bash".
        self.language = language
        # Content of the block.
        self.code = code
        # Reason of a skip comment, "" for a comment without reason, None otherwise.
        self.skip = skip
        # File that the block is written to, for write comments.
        self.target = target


# Controlled Markdown documents, as tracked by git.
def controlled_documents() -> list[str]:
    # List the tracked Markdown files.
    command = ["git", "ls-files", "-z", "--", "*.md"]
    # Run git and keep its output.
    out = subprocess.run(command, check=True, capture_output=True).stdout
    # Drop the empty trailing entry and the excluded directories.
    return sorted(n for n in out.decode().split("\0") if n and not n.startswith(EXCLUDED_DIRS))


# Logical lines of a shell block: continuations joined, comments dropped.
def shell_commands(code: str) -> list[str]:
    # Join backslash continuations.
    joined = re.sub(r"\\\n", " ", code)
    # Stripped lines.
    lines = (line.strip() for line in joined.splitlines())
    # Keep them without comments.
    return [s for s in lines if s and not s.startswith("#")]


# Whether a shell block consists of runnable unbihexium commands only.
def runnable_shell(code: str) -> bool:
    # Logical command lines.
    commands = shell_commands(code)
    # An empty block has nothing to run.
    if not commands:
        # Not runnable.
        return False
    # Check every command.
    for command in commands:
        # Words of the command; unparsable lines are not run.
        try:
            # Split like a POSIX shell.
            words = shlex.split(command)
        # Unbalanced quotes and similar.
        except ValueError:
            # Not runnable.
            return False
        # Drop leading variable assignments.
        while words and ASSIGNMENT.match(words[0]):
            # Next word.
            words = words[1:]
        # Only unbihexium commands run, and never the server.
        if not words or words[0] != "unbihexium" or "serve" in words[1:2]:
            # Not runnable.
            return False
    # Every command qualifies.
    return True


# Python and runnable shell blocks of a document, with their skip comments.
def document_blocks(path: str) -> list[Block]:
    # Text of the document.
    text = Path(path).read_text(encoding="utf-8")
    # Blocks found so far.
    blocks: list[Block] = []
    # Walk over the fenced blocks.
    for match in FENCE.finditer(text):
        # Info string and content.
        language, code = match.group(1), match.group(2)
        # Line of the opening fence.
        line = text.count("\n", 0, match.start()) + 1
        # Lines before the fence, without trailing blank lines.
        before = text[: match.start()].rstrip("\n").splitlines()
        # Last line before the fence.
        previous = before[-1].strip() if before else ""
        # Write comment directly above the fence, if any.
        write = WRITE.match(previous)
        # Blocks of any language can be written to a file.
        if write:
            # Remember the file block.
            blocks.append(Block(line, "file", code, None, write.group(1)))
            # Next block.
            continue
        # Otherwise only Python and runnable shell blocks count.
        if language != "python" and not (language == "bash" and runnable_shell(code)):
            # Next block.
            continue
        # Skip comment directly above the fence, if any.
        comment = SKIP.match(previous)
        # Reason of the skip; an empty string marks a comment without reason.
        skip = (comment.group(1) or "") if comment else None
        # Remember the block.
        blocks.append(Block(line, language, code, skip))
    # Every block of the document.
    return blocks


# Run the blocks of one document in this process; used by the child process.
def run_document(path: str, name: str) -> int:
    # Absolute path, since the working directory is a temporary directory.
    source = Path(path).resolve()
    # Namespace shared by the Python blocks of the document.
    namespace: dict[str, object] = {"__name__": "__main__"}
    # Run every block that is not skipped.
    for block in document_blocks(str(source)):
        # Skipped blocks do not run.
        if block.skip is not None:
            # Next block.
            continue
        # File blocks are written to the working directory.
        if block.language == "file":
            # Target file.
            target = Path(block.target)
            # Create its directory.
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write the content with LF line endings.
            target.write_text(block.code, encoding="utf-8", newline="\n")
            # Next block.
            continue
        # Python blocks run in the shared namespace.
        if block.language == "python":
            # Execute the block and report the first error.
            try:
                # Compile with the document and line as file name for tracebacks.
                exec(compile(block.code, f"{name}:{block.line}", "exec"), namespace)
            # Any error of the example.
            except BaseException:
                # Annotation on the fence.
                print(f"::error file={name},line={block.line}::Python example failed")
                # Full traceback in the log.
                traceback.print_exc()
                # Stop at the first failure of the document.
                return 1
        # Shell blocks run with bash, stopping at the first failing command.
        else:
            # Command that stops at the first failing line.
            command = ["bash", "-e", "-c", block.code]
            # Run the commands.
            result = subprocess.run(command, capture_output=True, text=True)
            # A non-zero exit status is a failure.
            if result.returncode:
                # Annotation on the fence.
                where = f"file={name},line={block.line}"
                # Report the exit status.
                print(f"::error {where}::shell example exited with {result.returncode}")
                # Output of the commands.
                print(result.stdout[-4000:] + result.stderr[-4000:])
                # Stop at the first failure of the document.
                return 1
    # Every block ran.
    return 0


# Run the examples of every document in a separate process.
def main(argv: list[str]) -> int:
    # Command line parser.
    parser = argparse.ArgumentParser(description="Run the code examples of the documentation")
    # Only list the blocks.
    parser.add_argument("--list", action="store_true", help="list the blocks and run nothing")
    # Internal: run one document in this process.
    parser.add_argument("--child", help=argparse.SUPPRESS)
    # Internal: name of that document in the annotations.
    parser.add_argument("--name", help=argparse.SUPPRESS)
    # Documents to use; all controlled documents by default.
    parser.add_argument("documents", nargs="*", help="Markdown documents")
    # Parse the options.
    args = parser.parse_args(argv)
    # The child process runs one document.
    if args.child:
        # Exit status of the document.
        return run_document(args.child, args.name or args.child)
    # Documents to check.
    documents = args.documents or controlled_documents()
    # Failures and problems found.
    failures = 0
    # Walk over the documents.
    for path in documents:
        # Blocks of the document.
        blocks = document_blocks(path)
        # Documents without examples need no process.
        if not blocks:
            # Next document.
            continue
        # A skip comment needs a reason.
        for block in blocks:
            # Report comments without reason.
            if block.skip == "":
                # Annotation on the fence.
                where = f"file={path},line={block.line}"
                # Report it.
                print(f"::error {where}::doc-example skip comment without a reason")
                # Count it.
                failures += 1
        # Number of blocks that run.
        active = sum(block.skip is None and not block.target for block in blocks)
        # Listing mode prints and runs nothing.
        if args.list:
            # One line per block.
            for block in blocks:
                # Run or write, or skip with the reason.
                state = f"write {block.target}" if block.target else "run"
                # Skipped blocks show their reason.
                state = f"skip ({block.skip})" if block.skip is not None else state
                # Print the block.
                print(f"{path}:{block.line} {block.language} {state}")
            # Next document.
            continue
        # Temporary working directory of the document.
        with tempfile.TemporaryDirectory(prefix="doc-examples-") as work:
            # Environment with a private model store.
            env = dict(os.environ, UNBIHEXIUM_CACHE=str(Path(work) / "cache"))
            # Start time.
            start = time.monotonic()
            # Run the document in a child process.
            try:
                # Same interpreter, this script, one document.
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve())]  # This script.
                    + ["--child", str(Path(path).resolve()), "--name", path],  # One document.
                    cwd=work,  # Working directory of the examples.
                    env=env,  # Private model store.
                    timeout=TIMEOUT,  # Upper bound per document.
                )  # End of the child process.
                # Exit status of the child.
                status = result.returncode
            # Documents that take too long.
            except subprocess.TimeoutExpired:
                # Report the timeout.
                print(f"::error file={path}::examples did not finish within {TIMEOUT} s")
                # Count as a failure.
                status = 1
        # Count failed documents.
        failures += status != 0
        # Duration of the document.
        elapsed = time.monotonic() - start
        # Result of the document.
        state = "failed" if status else "ok"
        # Print it.
        print(f"{path}: {active} of {len(blocks)} blocks, {elapsed:.1f} s, {state}")
    # Listing mode has nothing more to report.
    if args.list:
        # Exit status of the skip comment checks.
        return 1 if failures else 0
    # Summarise and set the exit status.
    if failures:
        # Number of failures.
        print(f"{failures} documentation example problem(s) found.")
        # Non-zero exit status fails the CI job.
        return 1
    # Confirm success.
    print("Every documentation example ran without an error.")
    # Zero exit status marks success.
    return 0


# Run the examples when the file is executed as a script.
if __name__ == "__main__":
    # Exit with the status returned by main().
    sys.exit(main(sys.argv[1:]))

# =============================================================================
# End of module .github/scripts/run_doc_examples.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
