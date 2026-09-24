<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : THIRD_PARTY_NOTICES.md
Title       : Third-Party Notices
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Third-Party Notices

| Field | Value |
| --- | --- |
| Document | UBX-DOC-THIRD-PARTY |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch: source tree, wheel, source distribution and container image |

## Abstract

This document identifies the third-party material that Unbihexium contains or depends on and the licence under which each item is available. It is intended for users, redistributors, packagers and auditors who need to meet the licence obligations of an Unbihexium installation or of the container image. It covers the third-party material reproduced in the source code, the direct runtime and optional dependencies declared in `pyproject.toml`, the complete locked runtime environment in `requirements.txt`, the native libraries that binary wheels of those dependencies contain, and the container image. The licences were read from the package metadata of the locked versions in September 2026. This document is informational and is not legal advice; the licence texts distributed with each component are authoritative.

## Contents

- [1. Scope and conventions](#1-scope-and-conventions)
- [2. Summary](#2-summary)
- [3. Third-party material in the source code](#3-third-party-material-in-the-source-code)
- [4. Direct runtime dependencies](#4-direct-runtime-dependencies)
- [5. Optional dependencies](#5-optional-dependencies)
- [6. Indirect runtime dependencies](#6-indirect-runtime-dependencies)
- [7. Native libraries in binary wheels](#7-native-libraries-in-binary-wheels)
- [8. Container image](#8-container-image)
- [9. Development and test tools](#9-development-and-test-tools)
- [10. Licence policy for dependencies](#10-licence-policy-for-dependencies)
- [11. Reproducing this inventory](#11-reproducing-this-inventory)
- [References](#references)

## 1. Scope and conventions

### 1.1 Scope

Unbihexium itself is licensed under the Mozilla Public License 2.0 [1] (see [LICENSE.txt](LICENSE.txt), [NOTICE](NOTICE) and [NOTICE.md](NOTICE.md)). This document lists material that is not the work of the Unbihexium project. Licences are given as SPDX identifiers [2] where the package declares one; where a package declares only a Trove classifier or free text, that declaration is reproduced.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [3] and RFC 8174 [4] when, and only when, they appear in capitals. They summarise obligations stated in the licences concerned; they do not add obligations of their own.

### 1.3 Not legal advice

This document is provided for information. It is not legal advice, and it does not modify any licence. Before redistributing Unbihexium or an environment built with it, verify the licences of the exact versions you distribute.

## 2. Summary

| Artefact | Contains third-party code | Contains other third-party material | Notes |
| --- | --- | --- | --- |
| Source tree and source distribution | No | Yes, colour specifications (Section 3) | All files are covered by `REUSE.toml` as MPL-2.0 |
| Wheel (`unbihexium-*.whl`) | No | Yes, colour specifications (Section 3) | Pure Python; dependencies are installed separately by the installer |
| Installed environment | Yes, the dependencies | Yes | Each dependency keeps its own licence (Sections 4 to 7) |
| Container image | Yes | Yes | CPython, Debian packages and the locked dependencies (Section 8) |

Unbihexium does not ship Earth observation imagery, third-party model weights or third-party training data. The starter weights of the model zoo are created locally from the model identifier and are untrained (see [NOTICE.md](NOTICE.md)).

## 3. Third-party material in the source code

The module `src/unbihexium/visualization/colormaps.py` reproduces colour values published by third parties. No third-party source code is copied into the package.

### 3.1 ColorBrewer colour schemes

The diverging and sequential colour maps use the ColorBrewer schemes RdYlGn, RdBu, BrBG and Blues as control points [5].

- Copyright: 2002 Cynthia Brewer, Mark Harrower and The Pennsylvania State University.
- Licence: Apache-2.0, as stated in the "Apache-Style Software License for ColorBrewer software and ColorBrewer Color Schemes" [6].
- Attribution: This product includes color specifications and designs developed by Cynthia Brewer (<http://colorbrewer.org/>).

Redistributors of Unbihexium SHOULD keep this attribution together with the licence notice of the colour schemes, as the Apache License 2.0 requires for redistributed works [7].

### 3.2 Viridis colour map

The `viridis` colour map is interpolated from nine samples of the viridis map designed by Stefan van der Walt and Nathaniel Smith for matplotlib. The authors have dedicated the colour maps to the public domain under CC0-1.0 [8]; no obligation follows, and the origin is recorded here for completeness.

### 3.3 ESA WorldCover class legend

`WORLDCOVER_PALETTE` reproduces the class values, class names and legend colours of the ESA WorldCover 10 m product as given in its product user manual (Zanaga et al., 2022) [9]. The WorldCover product is distributed under CC BY 4.0. Unbihexium reproduces only the legend so that users can display WorldCover data consistently; users who display or process WorldCover data MUST follow the attribution terms of that product.

### 3.4 Published formulas and sensor constants

Spectral index formulas, sensor band tables, calibration constants and statistical methods are implemented from their published descriptions, which are cited in the header of each module. Implementing a published method does not reproduce third-party code; the publications SHOULD be cited as described in [CITATION.md](CITATION.md).

## 4. Direct runtime dependencies

These packages are always installed with Unbihexium. "Minimum" is the lower bound in `pyproject.toml`; "Locked" is the version in `requirements.txt`, which the container image installs. Where a line gives several versions, they apply to Python 3.10, 3.11 and 3.12 or later in that order, or to Python 3.10 and 3.11 or later when two are given.

| Package | Minimum | Locked | Declared licence |
| --- | --- | --- | --- |
| numpy | 2.2.6 / 2.3.3 | 2.2.6 / 2.4.6 / 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.15.3 / 1.16.2 | 1.15.3 / 1.17.1 / 1.18.1 | BSD licence (classifier; BSD-3-Clause text) |
| rasterio | 1.4.4 | 1.4.4 (Python below 3.12) / 1.5.1 | BSD-3-Clause |
| shapely | 2.1.2 | 2.1.2 | BSD licence (classifier; BSD-3-Clause text) |
| geopandas | 1.1.2 | 1.1.4 | BSD-3-Clause |
| pyproj | 3.7.1 / 3.7.2 | 3.7.1 / 3.7.2 / 3.8.0 | MIT |
| click | 8.3.3 | 8.5.0 | BSD-3-Clause |
| rich | 14.1.0 | 15.0.0 | MIT |
| pydantic | 2.12.0 | 2.13.5 | MIT |
| pyyaml | 6.0.3 | 6.0.3 | MIT |
| requests | 2.33.0 | 2.34.2 | Apache-2.0 |
| pillow | 12.3.0 | 12.3.0 | MIT-CMU |
| scikit-image | 0.25.2 / 0.26.0 | 0.25.2 / 0.26.0 | BSD licence (classifier; BSD-3-Clause text) |

## 5. Optional dependencies

These packages are installed only when the corresponding extra is requested, for example `pip install "unbihexium[onnx]"`. The `onnx` and `serving` extras are included in `requirements.txt`; the others are locked in `requirements-dev.txt`.

| Extra | Package | Minimum | Locked | Declared licence |
| --- | --- | --- | --- | --- |
| onnx | onnxruntime | 1.23.2 / 1.24.1 | 1.23.2 / 1.30.0 | MIT |
| onnx, torch | onnx | 1.21.0 | 1.23.0 | Apache-2.0 |
| torch | torch | 2.13.0 | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| serving | fastapi | 0.133.0 | 0.141.1 | MIT |
| serving | starlette | 1.3.1 | 1.7.0 | BSD-3-Clause |
| serving | uvicorn | 0.37.0 | 0.53.0 | BSD-3-Clause |
| zarr | zarr | 2.18.3 / 3.1.4 | 2.18.3 / 3.1.6 / 3.4.0 | MIT |
| zarr | numcodecs | 0.13.1 / 0.16.4 | 0.13.1 / 0.16.5 / 0.17.0 | MIT |
| parquet | pyarrow | 23.0.1 | 25.0.1 | Apache-2.0 |

CUDA builds of PyTorch, which users install themselves to use a GPU, also install or require NVIDIA libraries and the NVIDIA CUDA runtime, which are distributed under NVIDIA's own licence terms, not under an open source licence. Users of those builds MUST accept and follow NVIDIA's terms.

## 6. Indirect runtime dependencies

The following packages are installed as dependencies of the packages in Sections 4 and 5 (`onnx` and `serving` extras). Together with those sections they form the complete environment of `requirements.txt`. Some entries apply only to certain Python versions or platforms (for example `pyreadline3` on Windows, `exceptiongroup` on Python 3.10).

| Package | Declared licence |
| --- | --- |
| affine | BSD-3-Clause |
| annotated-doc, annotated-types, anyio, attrs | MIT |
| certifi | MPL-2.0 |
| charset-normalizer | MIT |
| click-plugins, cligj | BSD (free text) |
| pyreadline3 | BSD licence (classifier) |
| coloredlogs, humanfriendly | MIT |
| exceptiongroup | MIT |
| flatbuffers | Apache-2.0 |
| h11, httptools | MIT |
| idna | BSD-3-Clause |
| imageio | BSD-2-Clause |
| lazy-loader, networkx, tifffile | BSD-3-Clause |
| markdown-it-py, mdurl | MIT |
| ml-dtypes | Apache-2.0 |
| mpmath, sympy | BSD (free text) |
| packaging | Apache-2.0 OR BSD-2-Clause |
| pandas | BSD licence (classifier; BSD-3-Clause text) |
| protobuf | BSD-3-Clause |
| pydantic-core, typing-inspection | MIT |
| pygments | BSD-2-Clause |
| pyogrio | MIT |
| pyparsing | MIT |
| python-dateutil | Apache-2.0 and BSD-3-Clause (its licence file applies both) |
| python-dotenv | BSD-3-Clause |
| pytz | MIT |
| six | MIT |
| typing-extensions | PSF-2.0 |
| tzdata | Apache-2.0 |
| urllib3 | MIT |
| uvloop | MIT OR Apache-2.0 |
| watchfiles | MIT |
| websockets | BSD-3-Clause |

## 7. Native libraries in binary wheels

Several dependencies are distributed as binary wheels that contain compiled third-party libraries. The list below was taken from the Linux x86_64 wheels of the locked versions for CPython 3.12; wheels for other platforms contain equivalent libraries. The numpy, scipy, shapely, pyproj and pillow wheels ship the licence texts of their bundled libraries in their `.dist-info` directories; the rasterio wheel ships only the rasterio licence, so the licences given for its libraries are those published by the upstream projects.

| Wheel | Bundled libraries | Licences |
| --- | --- | --- |
| rasterio | GDAL, PROJ, GEOS, HDF5, netCDF, libtiff, libjpeg, libpng, libwebp, OpenJPEG, libcurl, OpenSSL, SQLite, zstd, liblzma and others | GDAL and PROJ: MIT; GEOS: LGPL-2.1-or-later; OpenSSL: Apache-2.0; the others under permissive licences |
| pyogrio | GDAL | MIT |
| shapely | GEOS | LGPL-2.1-or-later |
| pyproj | PROJ, libcurl, libtiff, SQLite, nghttp2 | PROJ: MIT; the others under permissive licences |
| numpy, scipy | OpenBLAS, LAPACK, libgfortran, libquadmath | OpenBLAS: BSD-3-Clause; LAPACK: BSD-3-Clause-Open-MPI; libgfortran: GPL-3.0-or-later WITH GCC-exception-3.1; libquadmath: LGPL-2.1-or-later |
| pillow | libjpeg, libpng, libtiff, libwebp, FreeType, HarfBuzz, Little CMS, OpenJPEG, libavif, Brotli, zstd and others | Permissive licences listed in the pillow wheel |

The GCC Runtime Library Exception permits the distribution of programs linked with these runtime libraries under terms of the distributor's choice. LGPL-2.1-or-later libraries such as GEOS and libquadmath are shipped as separate shared libraries; a redistributor of the wheels MUST keep their licence texts and make the corresponding source available as the LGPL requires [10].

## 8. Container image

The image built from the [Dockerfile](Dockerfile) and published at `ghcr.io/unbihexium-oss/unbihexium` is based on the official `python:3.14.7-slim-trixie` image, pinned by digest, and contains:

- CPython 3.14, licensed under the Python Software Foundation License Version 2 (PSF-2.0) [11];
- a minimal Debian 13 (trixie) system, whose packages are under their own free software licences, recorded in `/usr/share/doc/*/copyright` inside the image;
- the packages of `requirements.txt` (Sections 4 to 7, `onnx` and `serving` extras included), installed from wheels with verified hashes, and pip;
- Unbihexium itself, with `LICENSE.txt`, `NOTICE` and `NOTICE.md`.

Redistributors of the image MUST comply with the licences of all of these components, including the source availability obligations of the GPL and LGPL licensed Debian packages; Debian publishes the corresponding source code at <https://sources.debian.org/>. The Docker workflow records an SPDX software bill of materials of every pushed image.

## 9. Development and test tools

The tools in the `test` and `dev` extras and the locked CI tools under `.github/requirements/` (for example pytest, ruff, pyright, bandit, pip-audit, build, twine, tox and hatchling) are used to build and check Unbihexium. They are not imported by the library and are not included in the wheel, the source distribution or the container image, so their licences place no obligations on users of Unbihexium. The GPL-3.0-or-later tools `reuse` and `yamllint` run only in isolated pre-commit and tox environments. [NOTICE](NOTICE), section 4.3, lists their licences.

## 10. Licence policy for dependencies

New dependencies are checked against a licence policy in two places: the dependency review of pull requests (`.github/dependency-review-config.yml`) and the licence compliance workflow (`.github/scripts/check_dependency_licenses.py`). The policy denies the AGPL and GPL families, SSPL-1.0, BUSL-1.1 and the non-commercial Creative Commons licences for Python packages; LGPL licences are allowed. The policy applies to the licence declared by each package. It does not apply to native libraries inside wheels (Section 7), which are listed here so that redistributors can meet their obligations.

## 11. Reproducing this inventory

The inventory can be reproduced for any environment. With the locked runtime environment:

```bash
python -m venv .venv-licences
.venv-licences/bin/python -m pip install --require-hashes -r requirements.txt
.venv-licences/bin/python -m pip install pip-licenses
.venv-licences/bin/pip-licenses --from=mixed --format=markdown --with-urls
```

The bundled native libraries of a wheel are listed in its `<package>.libs` directory, and their licence texts in `<package>-<version>.dist-info/licenses/`. Discrepancies between this document and the metadata of an installed package SHOULD be reported through an issue at <https://github.com/unbihexium-oss/unbihexium/issues>.

## References

[1] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[2] The Linux Foundation. SPDX License List. 2026. <https://spdx.org/licenses/>

[3] S. Bradner. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF, 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[4] B. Leiba. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF, 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[5] Mark Harrower and Cynthia A. Brewer. ColorBrewer.org: An Online Tool for Selecting Colour Schemes for Maps. The Cartographic Journal 40(1), 27-37. 2003. <https://doi.org/10.1179/000870403235002042>

[6] Cynthia Brewer, Mark Harrower and The Pennsylvania State University. Apache-Style Software License for ColorBrewer software and ColorBrewer Color Schemes. 2002. <https://github.com/axismaps/colorbrewer/blob/master/LICENCE.txt>

[7] The Apache Software Foundation. Apache License, Version 2.0. 2004. <https://www.apache.org/licenses/LICENSE-2.0>

[8] Nathaniel Smith and Stefan van der Walt. mpl-colormaps, licence (CC0-1.0). 2015. <https://github.com/BIDS/colormap/blob/master/LICENSE.txt>

[9] D. Zanaga et al. ESA WorldCover 10 m 2021 v200. Zenodo, 2022. <https://doi.org/10.5281/zenodo.7254221>

[10] Free Software Foundation. GNU Lesser General Public License, version 2.1. 1999. <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>

[11] Python Software Foundation. History and License, Python documentation. 2026. <https://docs.python.org/3/license.html>

<!--
=============================================================================
End of file THIRD_PARTY_NOTICES.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
