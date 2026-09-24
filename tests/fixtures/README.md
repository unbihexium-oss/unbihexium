<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : tests/fixtures/README.md
Title       : Test Fixture Files
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Test Fixture Files

| Field | Value |
| --- | --- |
| Document | UBX-DOC-TEST-FIXTURES |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | The files under `tests/fixtures/` on the main branch of Unbihexium |

## Abstract

This document describes the four NumPy array files in `tests/fixtures/`: their shapes, data types, value ranges, provenance and checksums, how they were generated as far as the repository shows, and how they relate to the test suite. It is written for contributors who write tests, for packagers who run the test suite from the source distribution, and for reviewers who need to know what data the repository contains. The description was obtained by loading every file and by reading the Git history; the key finding is that the files contain uniformly distributed random values without spatial or spectral structure and that no test currently uses them.

## Contents

1. [Scope and conventions](#1-scope-and-conventions)
2. [Files](#2-files)
3. [Provenance](#3-provenance)
4. [Use in the test suite](#4-use-in-the-test-suite)
5. [Loading the files](#5-loading-the-files)
6. [Adding fixtures](#6-adding-fixtures)
7. [Licensing](#7-licensing)
8. [References](#references)

## 1. Scope and conventions

### 1.1 Scope

The directory contains this document and four files in the NumPy `.npy` format version 1.0 [1], each holding one C-ordered array. The directory is part of the source distribution, because `tests/` is included in it (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`), but not of the wheel.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [2] and RFC 8174 [3] when, and only when, they appear in capitals. Array shapes are given as `(bands, rows, columns)` for rasters and `(rows, columns)` for single-layer arrays, the layout used throughout Unbihexium.

## 2. Files

### 2.1 Summary

| File | Shape | Type | Minimum | Maximum | Mean | Size |
| --- | --- | --- | --- | --- | --- | --- |
| `sample_data.npy` | (3, 64, 64) | float32 | 0.0000 | 1.0000 | 0.5008 | 49,280 bytes |
| `sentinel2_sample.npy` | (10, 64, 64) | float32 | 0.0000 | 0.9999 | 0.4994 | 163,968 bytes |
| `dem_sample.npy` | (1, 128, 128) | float32 | 0.0084 | 999.9722 | 499.9496 | 65,664 bytes |
| `mask_sample.npy` | (64, 64) | uint8 | 0 | 4 | 2.0039 | 4,224 bytes |

Minimum, maximum and mean are rounded to four decimals.

### 2.2 Content

- **`sample_data.npy`**: three layers of values in [0, 1), suitable as a generic three-band reflectance-like raster.
- **`sentinel2_sample.npy`**: ten layers of values in [0, 1). When the file was added, its bands were described as the ten Sentinel-2 bands B2, B3, B4, B5, B6, B7, B8, B8A, B11 and B12, in this order. The values do not reproduce any Sentinel-2 spectral signature: every layer has the same distribution, so a spectral index computed from it is noise centred on zero.
- **`dem_sample.npy`**: one layer of values between 0 and 1000, described as elevations in metres. The values of neighbouring cells are independent, so the array is not a plausible terrain: its median slope at a nominal 10 m resolution is 86.3 degrees (Section 5.2).
- **`mask_sample.npy`**: integer class labels 0 to 4 with the class counts 805, 845, 806, 809 and 831 (4,096 cells), described as a segmentation mask.

All four arrays behave like independent, uniformly distributed random samples: a histogram of each array in five equal bins is flat within sampling error, and the correlation between horizontally adjacent cells of the first layer lies between -0.03 and 0.01. The files carry no georeferencing, no band names and no no-data value.

### 2.3 Checksums

SHA-256 digests of the files as committed:

```text
56f158abd1275244880dc3da1f48f97ea2f8d043aba94e85682aa6555df66a79  sample_data.npy
e49a05fbc533bdf9b568b5c59f14f62251a288cbcfebe38d28e8ae42397126b0  sentinel2_sample.npy
c10382eaeb5958e3736982927771d6148b3fa7d83e8ffb68dc311868b77d2965  dem_sample.npy
5699bfe5623999389014d4e31044bad2c89d9218af171d959b0f32620cc9b1ee  mask_sample.npy
```

Run `sha256sum -c` on a file with these lines inside `tests/fixtures/` to check a copy.

## 3. Provenance

The files were added on 21 December 2025, before the tag `v1.0.0`: `sample_data.npy` in commit `85c6c1c` and the other three in commit `e5bb9d3`. They have not changed since. The repository does not contain the code that generated them, and the random seed is unknown, so they cannot be regenerated bit for bit; the statistics in Section 2 are consistent with draws from uniform distributions. The files are synthetic: they contain no imagery, elevation data or labels from any real place, sensor or data provider, and no personal data.

## 4. Use in the test suite

No test module, fixture or workflow in the repository reads these files. The shared fixtures in [tests/conftest.py](../conftest.py) generate their data at run time from seeded generators instead:

| Fixture | Content |
| --- | --- |
| `sample_raster_data` | (3, 256, 256) float32 values in [0, 1) from `numpy.random.default_rng(42)` |
| `tmp_geotiff` | `sample_raster_data` written as a GeoTIFF over the unit square in a temporary directory |
| `sample_bands` | the bands RED, NIR, GREEN, BLUE, SWIR1 and SWIR2 as (256, 256) arrays scaled to band-specific maxima, from `default_rng(42)` |
| `isolated_cache` | a temporary `UNBIHEXIUM_CACHE` directory, so that tests never write to the user's model store |

The files remain in the repository as small sample arrays for experiments and external scripts. Because they carry no structure, tests that need realistic spatial or spectral behaviour MUST NOT rely on them.

## 5. Loading the files

### 5.1 Inspecting the arrays

Run from the repository root:

```python
from pathlib import Path

import numpy as np

fixtures = Path("tests/fixtures")
for name in ["sample_data", "sentinel2_sample", "dem_sample", "mask_sample"]:
    array = np.load(fixtures / f"{name}.npy")
    print(f"{name:17s} {str(array.shape):14s} {array.dtype}  min {array.min():.4f}  max {array.max():.4f}")

mask = np.load(fixtures / "mask_sample.npy")
print(np.bincount(mask.ravel()))
```

```text
sample_data       (3, 64, 64)    float32  min 0.0000  max 1.0000
sentinel2_sample  (10, 64, 64)   float32  min 0.0000  max 0.9999
dem_sample        (1, 128, 128)  float32  min 0.0084  max 999.9722
mask_sample       (64, 64)       uint8  min 0.0000  max 4.0000
[805 845 806 809 831]
```

The files contain only numeric data, so `numpy.load` with its default `allow_pickle=False` reads them.

### 5.2 Writing a fixture as a GeoTIFF

Functions that expect a file need georeferencing, which the arrays lack. The following example attaches an arbitrary CRS and 10 m pixels to the elevation array, writes it to a temporary GeoTIFF and computes the slope:

```python
import tempfile
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import read_geotiff, write_geotiff
from unbihexium.terrain import slope

dem = np.load("tests/fixtures/dem_sample.npy")
with tempfile.TemporaryDirectory() as tmp:
    path = write_geotiff(dem, Path(tmp) / "dem.tif", crs="EPSG:3067",
                         transform=from_origin(385000, 6672000, 10, 10))
    data, meta = read_geotiff(path)
    print(data.shape, meta["crs"], np.array_equal(data, dem))
    print(round(float(np.nanmedian(slope(data[0], resolution=10.0))), 1))
```

```text
(1, 128, 128) EPSG:3067 True
86.3
```

The median slope of 86.3 degrees shows that the array is noise rather than terrain.

## 6. Adding fixtures

New test data SHOULD be generated inside the test, or in a fixture of `tests/conftest.py`, from a seeded `numpy.random.Generator`, as the existing fixtures do; this keeps the repository small and makes the data reproducible. A file MAY be added to this directory when generating the data would be impractical, provided that:

1. it is small (the existing files are below 200 KB each) and synthetic, or its licence permits redistribution under the terms in Section 7;
2. it contains no personal data and no imagery or labels whose redistribution is restricted;
3. the generating code, including the seed, is committed next to it or in the test that uses it;
4. this document is updated with its shape, type, value range, provenance and SHA-256 digest;
5. at least one test uses it.

Contributions follow [CONTRIBUTING.md](../../CONTRIBUTING.md).

## 7. Licensing

The files are covered by the repository-wide annotation of [REUSE.toml](../../REUSE.toml) [4]: copyright 2025 Unbihexium OSS Foundation and contributors, licensed under the Mozilla Public License 2.0 [5] ([LICENSE.txt](../../LICENSE.txt)). This note is not legal advice.

## References

[1] NumPy developers. A simple file format for NumPy arrays (NEP 1). 2007. <https://numpy.org/neps/nep-0001-npy-format.html>

[2] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[3] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[4] Free Software Foundation Europe. REUSE Specification, version 3.3. 2024. <https://reuse.software/spec-3.3/>

[5] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

<!--
=============================================================================
End of file tests/fixtures/README.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
