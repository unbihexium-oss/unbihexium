<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : AUTHORS.md
Title       : Authors and Contributors
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Authors and Contributors

| Field | Value |
| --- | --- |
| Document | UBX-DOC-105 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This document records the people and organisations who created Unbihexium and those who have contributed to it, and explains how authorship, copyright and citation relate to one another. It is intended for contributors who want to be credited, for researchers who cite the software and for anyone who needs to establish who holds rights in the code. The list is compiled from the version history of the repository; the history itself remains the authoritative record of individual contributions.

## Contents

- [1. Copyright holder](#1-copyright-holder)
- [2. Lead author](#2-lead-author)
- [3. Contributors](#3-contributors)
- [4. Automated contributions](#4-automated-contributions)
- [5. Authorship and citation](#5-authorship-and-citation)
- [6. Verifying the list](#6-verifying-the-list)
- [7. Adding yourself](#7-adding-yourself)

## 1. Copyright holder

Copyright notices in the repository read "2025-2026 Unbihexium OSS Foundation and contributors". [CITATION.cff](CITATION.cff) lists the Unbihexium OSS Foundation as the copyright holder, with the contact address <yunus.z.imanov@helsinki.fi> and the GitHub organisation <https://github.com/unbihexium-oss>. The project does not use a contributor licence agreement; contributors keep the copyright of their contributions and license them under the Mozilla Public License 2.0 (see [LICENSE.txt](LICENSE.txt) and [CONTRIBUTING.md](CONTRIBUTING.md)).

## 2. Lead author

| Name | Affiliation | Contact | GitHub |
| --- | --- | --- | --- |
| Olaf Yunus Laitinen Imanov | University of Helsinki | <yunus.z.imanov@helsinki.fi> | [@olaflaitinen](https://github.com/olaflaitinen) |

Olaf Yunus Laitinen Imanov created the project and wrote most of its code, documentation, tests, model zoo tooling and CI configuration. He is the lead maintainer (see [MAINTAINERS.md](MAINTAINERS.md)) and the author named in [CITATION.cff](CITATION.cff) and [codemeta.json](codemeta.json).

## 3. Contributors

The following people have contributed to Unbihexium, in alphabetical order:

| Name | GitHub | Contributions |
| --- | --- | --- |
| Exstaa | [@Exstaa](https://github.com/Exstaa) | Portable detection of the `model_zoo` path; Git LFS set-up instructions (December 2025) |
| Olaf Yunus Laitinen Imanov | [@olaflaitinen](https://github.com/olaflaitinen) | Lead author (Section 2) |

## 4. Automated contributions

Dependabot (`dependabot[bot]`) opens pull requests that update the pip dependencies, the GitHub Actions and the Docker base image according to `.github/dependabot.yml`. Its commits are part of the history but are not listed as authorship.

## 5. Authorship and citation

Being listed in this file acknowledges a contribution to the software. It does not by itself make a person an author in the citation metadata. The authors given in [CITATION.cff](CITATION.cff) and [codemeta.json](codemeta.json) are decided by the maintainers, taking into account the size and nature of the contributions; contributors who believe they should be included are invited to raise it by e-mail to <yunus.z.imanov@helsinki.fi>. How to cite the software is described in [CITATION.md](CITATION.md).

## 6. Verifying the list

The complete list of commit authors can be produced from the version history:

```bash
git shortlog --summary --numbered --email HEAD
```

The file [.mailmap](.mailmap) maps the e-mail addresses under which the lead author has committed to one identity, so that the command reports each person once. Contributions that are not commits, such as reviews, issue reports and discussions, are visible on GitHub but not in this output.

## 7. Adding yourself

Contributors may add themselves to the table in Section 3 in the pull request that contains their first contribution, or in a later one. Entries give the name the person wishes to be credited under, their GitHub account and a short description of the contribution, and keep the alphabetical order. If you commit under several e-mail addresses, you may also add a line to [.mailmap](.mailmap). Contributors who prefer not to be listed can ask for their entry to be removed at any time.

<!--
=============================================================================
End of file AUTHORS.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
