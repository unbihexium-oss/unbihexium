# Repository Instructions for AI Coding Assistants

These instructions apply to any AI coding assistant that proposes changes to this repository.

## Language and Style

- Write everything in English: code, comments, documentation, commit messages, issues and pull requests.
- Do not use emojis or em dashes anywhere. Use commas, colons, parentheses or separate sentences instead.
- Follow the existing style of the surrounding code and documentation.

## Licensing

- The project is licensed under the Mozilla Public License 2.0 (`LICENSE.txt`).
- Every new Python module or shell script starts with the MPL-2.0 notice, after the shebang if there is one:

  ```text
  # This Source Code Form is subject to the terms of the Mozilla Public
  # License, v. 2.0. If a copy of the MPL was not distributed with this
  # file, You can obtain one at https://mozilla.org/MPL/2.0/.
  ```

- Do not add dependencies under GPL, AGPL, SSPL or non-commercial licences.

## Code

- Python 3.10 to 3.14 must be supported. Use type hints and Google style docstrings for public APIs.
- Run `ruff check src/`, `ruff format --check src/` and `pytest tests/` before proposing changes.
- Never commit secrets, credentials, personal data or notebook outputs.

## Commits and Pull Requests

- Commit messages follow `type(scope): description`, for example `fix(io): handle missing nodata values`.
- Pull request titles follow the same format and are checked by the PR Title workflow.
- Answer every question of the pull request template, including the AI usage declaration.
