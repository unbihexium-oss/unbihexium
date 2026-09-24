<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : NOTICE.md
Title       : Notices
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Notices

| Field | Value |
| --- | --- |
| Document | UBX-DOC-NOTICE |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch: source tree, wheel, source distribution and container image |

## Abstract

This document states the copyright, licence, warranty, model zoo, third-party, container, data and trademark notices of Unbihexium in Markdown. It is the readable companion of the plain text [NOTICE](NOTICE) file, follows the same section order and is distributed with it in every wheel and source distribution. It is written for users, redistributors, packagers and auditors who need to know under which terms Unbihexium and its components may be used and passed on. The detailed inventory of third-party components is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This document is informational and is not legal advice; it does not modify the licence, and where it and [LICENSE.txt](LICENSE.txt) differ, LICENSE.txt prevails.

## Contents

- [1. Copyright and licence](#1-copyright-and-licence)
- [2. Disclaimer of warranty and limitation of liability](#2-disclaimer-of-warranty-and-limitation-of-liability)
- [3. Model zoo](#3-model-zoo)
- [4. Third-party software and material](#4-third-party-software-and-material)
- [5. Container image](#5-container-image)
- [6. Data](#6-data)
- [7. Trademarks](#7-trademarks)
- [8. Related documents](#8-related-documents)
- [References](#references)

## 1. Copyright and licence

### 1.1 Copyright

Unbihexium. Copyright 2025-2026 Unbihexium OSS Foundation and contributors.

This product includes software developed by the Unbihexium OSS Foundation and the contributors listed in [AUTHORS.md](AUTHORS.md) (<https://github.com/unbihexium-oss/unbihexium>).

### 1.2 Licence

Unbihexium is licensed under the Mozilla Public License, version 2.0 [1] (SPDX licence identifier `MPL-2.0` [2]). The full licence text is in [LICENSE.txt](LICENSE.txt) at the root of the source tree and of every source distribution, and at <https://mozilla.org/MPL/2.0/>. Versions up to and including 1.0.1 were released under the Apache License 2.0; the project was relicensed to MPL-2.0 on the main branch after 1.0.1 (see [CHANGELOG.md](CHANGELOG.md)).

Every source file carries the MPL-2.0 Exhibit A notice:

```text
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

Files that cannot carry a comment, such as JSON files and NumPy arrays, are covered by the annotation in `REUSE.toml`, which applies the same copyright and licence to every file in the repository in the manner defined by the REUSE specification [3].

### 1.3 What the licence means in practice

The MPL-2.0 is a file-level copyleft licence. In summary, and without replacing the licence text:

- Modified versions of files covered by the MPL-2.0 must remain under the MPL-2.0, and their source code must be made available when they are distributed (MPL-2.0 sections 3.1 and 3.2).
- Unbihexium may be combined with code under other licences, including proprietary code, to form a "Larger Work" (MPL-2.0 section 3.3). Files that do not contain MPL-2.0 covered code may be licensed as their authors choose.
- The MPL-2.0 is compatible with the GNU GPL 2.0 or later, the GNU LGPL 2.1 or later and the GNU AGPL 3.0 or later (MPL-2.0 section 3.3 and Exhibit B). No file in Unbihexium carries the "Incompatible With Secondary Licenses" notice.
- The licence grants no rights to the trademarks of any contributor (MPL-2.0 section 2.3).

## 2. Disclaimer of warranty and limitation of liability

Unbihexium is provided on an "as is" basis, without warranty of any kind, as set out in sections 6 (Disclaimer of Warranty) and 7 (Limitation of Liability) of the MPL-2.0. Results produced with Unbihexium, including the output of models from the model zoo, must be validated before they are used for decisions that affect people, property or the environment. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) and [COMPLIANCE.md](COMPLIANCE.md).

## 3. Model zoo

The model catalogue (`src/unbihexium/zoo/catalog.yaml`), the generated manifests, model cards, inventory and checksums under `model_zoo/`, and the architectures that implement the models were produced by the Unbihexium project and are licensed under the MPL-2.0, like the rest of the repository.

The catalogue defines 520 models: 130 families in four size variants (tiny, base, large and mega). They are untrained starter models. Each has a complete architecture and starter weights that are initialised deterministically from the model identifier when the model is built into the local model store; no weight files are stored in the repository or the distributions. The models have not been trained on Earth observation data and produce meaningless predictions until they are trained. The exception is the 28 models of the 7 spectral index families, which compute published index formulas and need no training.

The models contain no third-party weights and no third-party training data. Each model card under `model_zoo/cards/` records the licence and status of its model; `model_zoo/checksums.txt` and `src/unbihexium/zoo/digests.json` list the SHA-256 digests of the starter weights, which are verified when a model is loaded. Weights that users obtain by training a model on their own data are the users' responsibility, including the licences of the training data.

## 4. Third-party software and material

### 4.1 Package contents

Unbihexium is distributed as a pure Python package. The wheel and the source distribution contain no third-party code. The dependencies are installed separately by the package installer, each under its own licence, and are not relicensed by Unbihexium. The module `unbihexium.visualization.colormaps` reproduces colour specifications from ColorBrewer (Apache-2.0), samples of the viridis colour map (CC0-1.0) and the ESA WorldCover class legend; the attribution required for ColorBrewer is given in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md):

> This product includes color specifications and designs developed by Cynthia Brewer (<http://colorbrewer.org/>).

### 4.2 Runtime dependencies

The runtime dependencies are numpy, scipy, rasterio, shapely, geopandas, pyproj, click, rich, pydantic, pyyaml, requests, pillow and scikit-image. All are under permissive licences (BSD-3-Clause, MIT, MIT-CMU and Apache-2.0). Binary wheels of some of them bundle native libraries such as GDAL, PROJ, GEOS (LGPL-2.1-or-later) and the GCC runtime libraries (GPL-3.0-or-later with the GCC Runtime Library Exception).

### 4.3 Optional dependencies

The extras `onnx`, `torch`, `serving`, `zarr` and `parquet` install further packages, including onnxruntime and onnx, torch, fastapi, starlette and uvicorn, zarr and numcodecs, and pyarrow. CUDA builds of PyTorch, which users install themselves to use a GPU, require NVIDIA libraries under NVIDIA's own licence terms.

### 4.4 Development and test tools

The tools used to build and check Unbihexium are not imported by the library and are not included in any distribution, so their licences place no obligations on users.

### 4.5 Details and compatibility

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists every package, its minimum and locked versions and its declared licence, the native libraries bundled in wheels, and the licence policy that `.github/dependency-review-config.yml` and `.github/scripts/check_dependency_licenses.py` enforce for new dependencies. For an installed environment, `pip-licenses --from=mixed` lists every package with its version and licence.

## 5. Container image

The image built from the [Dockerfile](Dockerfile) and published at `ghcr.io/unbihexium-oss/unbihexium` contains, in addition to Unbihexium and the packages from `requirements.txt`:

- CPython, under the Python Software Foundation License Version 2 (PSF-2.0), from the official `python:3.14-slim-trixie` image;
- a minimal Debian 13 (trixie) system, whose packages are under their own free software licences, recorded in `/usr/share/doc/*/copyright` inside the image.

Redistributors of the image must comply with the licences of all of these components, including the source availability obligations of the GPL and LGPL licensed Debian packages. Debian publishes the corresponding source code at <https://sources.debian.org/>. The Docker workflow records an SPDX software bill of materials of every pushed image.

## 6. Data

Unbihexium does not ship Earth observation imagery or other third-party data. Users are responsible for complying with the licences and terms of use of the data they process, for example the Copernicus Sentinel data licence or the USGS Landsat terms. The tests generate synthetic arrays at run time; the repository contains no imagery.

## 7. Trademarks

"Unbihexium" is the name of this project. All other product names, logos and brands mentioned in the source code or the documentation are the property of their respective owners and are used for identification only. Their use does not imply endorsement.

## 8. Related documents

| Topic | Document |
| --- | --- |
| Licence text | [LICENSE.txt](LICENSE.txt) |
| Notice in plain text | [NOTICE](NOTICE) |
| Third-party software and material | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| Copyright and licence map | [REUSE.toml](REUSE.toml) |
| Citation | [CITATION.cff](CITATION.cff), [CITATION.md](CITATION.md) |
| Security policy | [SECURITY.md](SECURITY.md) |
| Contact | <https://github.com/unbihexium-oss/unbihexium/issues>, <yunus.z.imanov@helsinki.fi> |

## References

[1] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[2] The Linux Foundation. SPDX License List. 2026. <https://spdx.org/licenses/>

[3] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

<!--
=============================================================================
End of file NOTICE.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
