<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : CITATION.md
Title       : Citing Unbihexium
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Citing Unbihexium

| Field | Value |
| --- | --- |
| Document | UBX-DOC-104 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0, development versions from the main branch, and the metadata in CITATION.cff and codemeta.json |

## Abstract

This document explains how to cite Unbihexium in publications, theses, reports and software documentation. It gives ready-to-use references for the latest release, explains how to cite an unreleased version from the main branch, how to report the use of model zoo models, and which works the library depends on should also be cited. It is written for researchers and authors who use the software. The citation data are taken without change from the machine-readable files [CITATION.cff](CITATION.cff), in the Citation File Format 1.2.0 [1], and [codemeta.json](codemeta.json), in CodeMeta 3.0 [2]; where this document and those files differ, the files prevail and the difference is a defect to be reported. The practice described here follows the software citation principles of the FORCE11 Software Citation Working Group [3].

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Citation metadata](#2-citation-metadata)
- [3. Citing a release](#3-citing-a-release)
- [4. Citing a development version](#4-citing-a-development-version)
- [5. Reporting the use of models](#5-reporting-the-use-of-models)
- [6. Citing the underlying software](#6-citing-the-underlying-software)
- [7. Machine-readable metadata](#7-machine-readable-metadata)
- [8. Corrections](#8-corrections)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

This document covers citation of the Unbihexium software. Unbihexium has no associated journal article or conference paper, so the software itself is the citable work.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [4] and RFC 8174 [5] when, and only when, they appear in capitals. They describe the practice the project asks of citing authors; they do not add conditions to the licence, which permits use without citation.

## 2. Citation metadata

The table lists every citation field recorded in [CITATION.cff](CITATION.cff) and the corresponding value in [codemeta.json](codemeta.json).

| Field | CITATION.cff | codemeta.json |
| --- | --- | --- |
| Title | Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python | `name`: Unbihexium |
| Type | software | SoftwareSourceCode |
| Author 1 | Unbihexium OSS Foundation (organisation, copyright holder) | Unbihexium OSS Foundation (Organization) |
| Author 2 | Olaf Yunus Laitinen Imanov, University of Helsinki | Olaf Yunus Laitinen Imanov, University of Helsinki (Person) |
| Contact | Olaf Yunus Laitinen Imanov, <yunus.z.imanov@helsinki.fi> | Maintainer: Olaf Yunus Laitinen Imanov, <yunus.z.imanov@helsinki.fi> |
| Version | 2.0.1 | `version` and `softwareVersion`: 2.0.1 |
| Release date | `date-released`: 2026-09-24 | `datePublished`: 2025-12-18 (first publication); `dateModified`: 2026-09-24 (last change of the metadata) |
| Licence | MPL-2.0 | <https://spdx.org/licenses/MPL-2.0> |
| Repository | <https://github.com/unbihexium-oss/unbihexium> | `codeRepository`: same address |
| Package | <https://pypi.org/project/unbihexium/> | `installUrl`: same address |
| Release identifier | <https://github.com/unbihexium-oss/unbihexium/releases/tag/v2.0.1> | Not recorded |
| DOI | None | None |
| ORCID of the personal author | None | None |

Unbihexium has no DOI. Until one is assigned, the release page URL of the version used is the most specific persistent reference. Neither file records an ORCID identifier for the personal author; references therefore give the name and affiliation only.

## 3. Citing a release

### 3.1 Recommended reference

Cite the exact version you used. For version 2.0.1, the latest release, the recommended reference is:

> Unbihexium OSS Foundation, & Laitinen Imanov, O. Y. (2026). *Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python* (Version 2.0.1) [Computer software]. <https://github.com/unbihexium-oss/unbihexium/releases/tag/v2.0.1>

This reference follows the APA 7th edition pattern for software [6]; adapt it to the style your publisher requires, keeping the authors, title, version, year and URL.

### 3.2 BibTeX

For BibLaTeX, which supports the `@software` entry type:

```bibtex
@software{unbihexium_2_0_1,
  author  = {{Unbihexium OSS Foundation} and Laitinen Imanov, Olaf Yunus},
  title   = {Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python},
  version = {2.0.1},
  date    = {2026-09-24},
  url     = {https://github.com/unbihexium-oss/unbihexium/releases/tag/v2.0.1},
  license = {MPL-2.0},
}
```

For classic BibTeX styles, which do not know `@software`, use `@misc` with `year = {2026}`, `month = {sep}` and `note = {Version 2.0.1}` instead of `version` and `date`. The double braces around the organisation name keep it from being split into given and family names.

### 3.3 Other formats

GitHub shows a "Cite this repository" button on the repository page that exports APA and BibTeX from [CITATION.cff](CITATION.cff). The same file can be converted to other formats with cffconvert [7], for example `cffconvert -f ris`, `cffconvert -f endnote` or `cffconvert -f zenodo`, and reference managers such as Zotero import it directly.

### 3.4 In-text mention

When the software is mentioned in the text, give its name and version, for example "processed with Unbihexium 2.0.1 [ref]". Authors SHOULD also state the Python version and the operating system in the methods section when numerical results are reported.

## 4. Citing a development version

Between releases, the main branch can contain unreleased changes (listed under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md)) while still reporting the version of the last release, because the version number changes only at a release (see [VERSIONING.md](VERSIONING.md)). If you used an installation from the main branch or from any commit that is not a tagged release, you SHOULD cite the commit instead of the release:

> Unbihexium OSS Foundation, & Laitinen Imanov, O. Y. (2026). *Unbihexium: Earth Observation, Geospatial, Remote Sensing and SAR Library for Python* (Development version, commit 0123abc) [Computer software]. <https://github.com/unbihexium-oss/unbihexium/tree/0123abc>

Replace `0123abc` with the full or abbreviated commit hash of the checkout you installed from (`git rev-parse HEAD`) and the year with the year of that commit. The version number alone (`unbihexium --version`) does not distinguish a development installation from the release.

## 5. Reporting the use of models

The model zoo contains 520 models (130 families in four size variants). They are untrained starter models: each has a complete architecture and deterministic starter weights, but has not been trained on Earth observation data. The exception is the 28 models of the 7 spectral index families, which compute published formulas and need no training. Results obtained with a starter model have no meaning until the model is trained.

When a publication reports results of a model trained or run with Unbihexium, authors SHOULD report:

- the Unbihexium version or commit, as in [Section 3](#3-citing-a-release) or [Section 4](#4-citing-a-development-version);
- the model identifier, for example `ship_detector_base`, and the catalogue version, which `unbihexium.zoo.catalog_version()` returns (2.0.0 at the time of review);
- the SHA-256 digest of the checkpoint or ONNX file that produced the results;
- the training data, their licence and citation, and the validation data and accuracy measures used.

For spectral index models, cite the original publication of the index formula; the references are given in the source code of `unbihexium.indices` and `unbihexium.core.index`.

## 6. Citing the underlying software

Unbihexium reads and writes rasters through GDAL (via rasterio), transforms coordinates through PROJ (via pyproj) and runs exported models with ONNX Runtime. [CITATION.cff](CITATION.cff) lists these works under `references` with the citations their projects recommend. Where your results depend directly on them, you SHOULD cite them as well:

- GDAL/OGR contributors. *GDAL/OGR Geospatial Data Abstraction software Library*. Open Source Geospatial Foundation. <https://doi.org/10.5281/zenodo.5884351>
- PROJ contributors. *PROJ coordinate transformation software library*. Open Source Geospatial Foundation. <https://doi.org/10.5281/zenodo.5884394>
- ONNX Runtime developers. *ONNX Runtime*. <https://onnxruntime.ai>

Cite also the data you processed, following the terms of their providers, and any other library your analysis relies on, such as NumPy, SciPy, scikit-image or PyTorch. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the dependencies of Unbihexium.

## 7. Machine-readable metadata

### 7.1 Files

- [CITATION.cff](CITATION.cff): Citation File Format 1.2.0 [1]. It is read by GitHub, Zenodo and reference managers, and is the authoritative source for the citation.
- [codemeta.json](codemeta.json): CodeMeta 3.0 [2] software metadata in JSON-LD, including requirements, platforms and keywords.
- [.zenodo.json](.zenodo.json): metadata of the Zenodo deposit that the GitHub integration of Zenodo creates for each GitHub release: title, description, creators, licence, keywords and related identifiers. When this file is present, Zenodo reads it instead of `CITATION.cff`; the version and publication date of a deposit are taken from the release. The integration is not yet enabled, so no release has a DOI yet ([ROADMAP.md](ROADMAP.md), Section 5.2).

### 7.2 Maintenance

The maintainer MUST update `version`, `date-released` and the version-specific `identifiers` in `CITATION.cff`, and `version`, `softwareVersion` and `dateModified` in `codemeta.json`, in every release, as listed in [VERSIONING.md](VERSIONING.md). `.zenodo.json` carries no version and changes only when the title, the description, the creators or the keywords change; it MUST then agree with `CITATION.cff`. This document MUST be updated in the same change. `CITATION.cff` is validated against the Citation File Format schema by the Repository Config workflow (`.github/workflows/repo-config.yml`); it can be checked locally with `cffconvert --validate`.

## 8. Corrections

If you find that the citation metadata are wrong or incomplete, open an issue at <https://github.com/unbihexium-oss/unbihexium/issues> or write to <yunus.z.imanov@helsinki.fi>.

## References

[1] Stephan Druskat, Jurriaan H. Spaaks, Neil Chue Hong, Robert Haines, James Baker, Spencer Bliven, Egon Willighagen, David Perez-Suarez and Olexandr Konovalov. Citation File Format, version 1.2.0. 2021. <https://citation-file-format.github.io/>

[2] CodeMeta Project. CodeMeta, version 3.0. 2023. <https://codemeta.github.io/>

[3] Arfon M. Smith, Daniel S. Katz, Kyle E. Niemeyer and FORCE11 Software Citation Working Group. Software citation principles. PeerJ Computer Science 2:e86. 2016. <https://doi.org/10.7717/peerj-cs.86>

[4] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[5] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[6] American Psychological Association. APA Style: Software references. 2020. <https://apastyle.apa.org/style-grammar-guidelines/references/examples/software-references>

[7] Citation File Format project. cffconvert: command line program to validate and convert CITATION.cff files. <https://github.com/citation-file-format/cffconvert>

<!--
=============================================================================
End of file CITATION.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
