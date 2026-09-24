<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : examples/README.md
Title       : Examples
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Examples

| Field | Value |
| --- | --- |
| Document | UBX-DOC-307 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../MAINTAINERS.md)) |
| Applies to | The files under `examples/` on the main branch of Unbihexium |

## Abstract

This document describes the three example programs in the `examples/` directory of the repository: a command line script that computes the Normalized Difference Vegetation Index (NDVI) of a GeoTIFF, a command line script that runs the ship detector and writes GeoJSON, and a small FastAPI application that exposes both over HTTP. It is written for users who want a minimal, readable starting point for their own scripts, and for reviewers who need to know what the examples actually do and where they fall short. For each example it gives the purpose, the requirements, the options, a tested invocation with its real output, and the known limitations. The examples were run against the main branch on 24 September 2026; they are not part of the package, are not run in CI, and are not the recommended interface for production work, which is the `unbihexium` command and the `unbihexium.serving` service.

## Contents

1. [Overview and conventions](#1-overview-and-conventions)
2. [Preparing test inputs](#2-preparing-test-inputs)
3. [scripts/calculate_ndvi.py](#3-scriptscalculate_ndvipy)
4. [scripts/detect_ships.py](#4-scriptsdetect_shipspy)
5. [serving/api.py](#5-servingapipy)
6. [Test status and known issues](#6-test-status-and-known-issues)
7. [Licence](#7-licence)
8. [References](#references)

## 1. Overview and conventions

### 1.1 Files

```text
examples/
  README.md                this document
  scripts/
    calculate_ndvi.py      NDVI of a multi-band GeoTIFF, written as a GeoTIFF
    detect_ships.py        ship detection on a GeoTIFF, written as GeoJSON
  serving/
    __init__.py            makes examples.serving importable for uvicorn
    api.py                 FastAPI application with health, info, detection and NDVI routes
```

| Example | Library functions used | Extras needed |
| --- | --- | --- |
| `scripts/calculate_ndvi.py` | `core.raster.Raster`, `core.index.compute_index`, `io.geotiff.write_geotiff` | none |
| `scripts/detect_ships.py` | `core.raster.Raster`, `ai.detection.ShipDetector` | `torch` |
| `serving/api.py` | the above and `ai.detection.BuildingDetector` | `torch`, `serving` |

Every Python file carries the MPL-2.0 header of the project, a description of its method and usage, and a comment on each line. The models used by the detection examples are untrained starter models: the 520 models of the model zoo (130 families in the variants tiny, base, large and mega) have deterministic, untrained weights, except the 28 models of the 7 spectral index families, which compute exact formulas. The detection results of these examples are therefore not meaningful ([RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md)).

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [5] and RFC 8174 [6] when, and only when, they appear in capitals.

## 2. Preparing test inputs

The commands in Sections 3 to 5 use two synthetic GeoTIFFs, written by the following script (`make_inputs.py`) in an empty working directory. `input.tif` has four reflectance bands in the order blue, green, red and near infrared (NIR); `rgb.tif` has three bands in the order red, green and blue.

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.io import write_geotiff

rng = np.random.default_rng(0)
transform = from_origin(500000, 6700000, 10, 10)
# Four bands in the order blue, green, red, near infrared (reflectance).
write_geotiff(rng.uniform(0.02, 0.5, (4, 64, 64)).astype("float32"), "input.tif",
              crs="EPSG:32635", transform=transform)
# Three bands red, green, blue for the ship detector.
write_geotiff(rng.uniform(0.0, 0.3, (3, 128, 128)).astype("float32"), "rgb.tif",
              crs="EPSG:32635", transform=transform)
```

In the commands below, `$REPO` stands for the path of the repository checkout, and `UNBIHEXIUM_CACHE` MAY be set to a temporary directory to keep the model store out of the home directory.

## 3. scripts/calculate_ndvi.py

### 3.1 Purpose

Reads a multi-band GeoTIFF, computes NDVI [1],

$$\mathrm{NDVI} = \frac{\mathrm{NIR} - \mathrm{RED}}{\mathrm{NIR} + \mathrm{RED}},$$

prints its range and mean, and writes it as a single-band GeoTIFF with the coordinate reference system and geotransform of the input.

### 3.2 Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input`, `-i` | required | Input GeoTIFF |
| `--output`, `-o` | required | Output GeoTIFF |
| `--nir-band` | 4 | 1-based band number of the NIR band |
| `--red-band` | 3 | 1-based band number of the red band |

The defaults match a four-band image in the order blue, green, red, NIR. For a Sentinel-2 stack in the order B01, B02, ..., B08, use `--nir-band 8 --red-band 4`.

### 3.3 Run

```bash
python "$REPO/examples/scripts/calculate_ndvi.py" --input input.tif --output ndvi.tif
```

```text
Loading: input.tif
Image shape: (4, 64, 64)
CRS: EPSG:32635
Calculating NDVI...
NDVI range: [-0.920, 0.919]
NDVI mean: 0.000
Saving: ndvi.tif
Done!
```

The mean is close to zero because the synthetic red and NIR bands are drawn from the same distribution.

### 3.4 Behaviour and limitations

- The index is computed with `unbihexium.core.index.compute_index("NDVI", {"NIR": nir, "RED": red})`; pixels where NIR + RED = 0 become NaN, and the printed statistics ignore them.
- The output is a `float64` GeoTIFF with the library defaults of `write_geotiff` (DEFLATE compression, tiled layout). No no-data value is set in the file, so NaN pixels are not flagged as no-data for other software.
- The input must be reflectance; digital numbers of Level-1 or Level-2 products give meaningful NDVI only after scaling.
- The same result, as `float32`, is produced by the maintained command `unbihexium index NDVI -i input.tif -o ndvi.tif --red 3 --nir 4`.

## 4. scripts/detect_ships.py

### 4.1 Purpose

Runs `unbihexium.ai.ShipDetector` with the model given by `--model-id` on a GeoTIFF and writes the detections as a GeoJSON FeatureCollection [2], one rectangular polygon per detection with the properties `class_id`, `class_name` and `confidence`. The polygons are in the map coordinates of a georeferenced input, and the CRS is recorded in the file.

### 4.2 Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input`, `-i` | required | Input GeoTIFF with the bands red, green and blue |
| `--output`, `-o` | required | Output GeoJSON file |
| `--threshold`, `-t` | 0.5 | Minimum detection score |
| `--model-id` | `ship_detector_tiny` | Model zoo family or model id of a detection model |

### 4.3 Run

```bash
python "$REPO/examples/scripts/detect_ships.py" --input rgb.tif --output ships.geojson --threshold 0.3
```

```text
Loading: rgb.tif
Using model: ship_detector_tiny
Threshold: 0.3
Found 0 ships
Saving: ships.geojson (EPSG:32635)
Done!
```

`ships.geojson` then contains a FeatureCollection without features, the member `"crs": {"type": "name", "properties": {"name": "EPSG:32635"}}` and `"model_id": "ship_detector_tiny"`.

### 4.4 Behaviour and limitations

- The model given by `--model-id` is built and verified in memory on first use; a model of another task is rejected.
- `ship_detector_tiny` is an untrained starter model. Finding no ships, or finding boxes in arbitrary places, is the expected behaviour until the model is trained ([docs/model_zoo/training.md](../docs/model_zoo/training.md)). A trained checkpoint can be used by changing the constructor call to `ShipDetector(weights="runs/ship_detector_tiny/best.pt", threshold=...)`.
- For a georeferenced input the polygons are built from `Detection.geo_bbox` in the CRS of the raster, which is recorded in the named `crs` member of the GeoJSON 2008 specification; `unbihexium.io.reproject_geojson` converts such a file to longitude and latitude. Inputs without a CRS or with the identity transform are written in pixel coordinates (column, row) with `crs` set to `pixel`. `unbihexium predict ship_detector_tiny rgb.tif ships.geojson` writes the same kind of file.
- Building the detector requires PyTorch (extra `torch`).

## 5. serving/api.py

### 5.1 Purpose

A minimal FastAPI [3] application that accepts uploaded GeoTIFFs and returns ship or building detections or NDVI statistics. It shows how the library can be wrapped in a web service; it is not the project's REST service.

| Method and path | Input | Response |
| --- | --- | --- |
| `GET /health` | none | `status` and library `version` |
| `GET /info` | none | package name, version and a one-line description of the library |
| `POST /detect/ships` | multipart field `file`; query `threshold` (default 0.5) | `count`, `model_id`, `detections` with `bbox`, `confidence`, `class_id`, `class_name` |
| `POST /detect/buildings` | as above; the image must have 3 bands | as above |
| `POST /index/ndvi` | multipart field `file`; query `nir_band` (default 4), `red_band` (default 3) | `index_name`, `min_value`, `max_value`, `mean_value`, `shape` |

### 5.2 Run

The application needs the `serving` and `torch` extras and `python-multipart`, which FastAPI uses to read file uploads (`python -m pip install python-multipart`). It is started from the repository root, so that `examples.serving` can be imported:

```bash
cd "$REPO"
uvicorn examples.serving.api:app --host 127.0.0.1 --port 8000
```

From the working directory of Section 2, in a second terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -F "file=@input.tif" "http://127.0.0.1:8000/index/ndvi?nir_band=4&red_band=3"
curl -s -F "file=@rgb.tif" "http://127.0.0.1:8000/detect/ships?threshold=0.5"
curl -s -F "file=@input.tif" "http://127.0.0.1:8000/detect/buildings"
```

```text
{"status":"healthy","version":"2.0.1"}
{"index_name":"NDVI","min_value":-0.9200797496617876,"max_value":0.9190091032253411,"mean_value":0.00028512461033844655,"shape":[64,64]}
{"count":0,"model_id":"ship_detector_base","detections":[]}
{"detail":"building_detector_base expects 3 bands (red, green, blue), got shape (4, 64, 64)"}
```

curl prints no line break after a response; the responses are shown on separate lines. The last request fails with status 422 because the building detector expects three bands and `input.tif` has four. The interactive OpenAPI documentation is served at `/docs`.

### 5.3 Behaviour and limitations

The application is deliberately simple and MUST NOT be exposed to untrusted clients:

- There is no authentication, no rate limit and no limit on the upload size; every upload is read completely into memory.
- Invalid input (an upload that is not a raster, band numbers or band counts that do not fit) is answered with status 422; other failures are status 500 without internal details.
- Uploads are written to a temporary file that is deleted after every request, whether it succeeds or fails.
- The detectors use their default models `ship_detector_base` and `building_detector_base`, which are untrained starter models, and the boxes are returned in pixel coordinates.

The maintained service is `unbihexium.serving` (`unbihexium serve`), which serves every zoo model through `POST /predict/{model_id}` with request size, pixel and value limits, optional API keys, rate limiting and CORS settings; see [README.md, Section 8](../README.md#8-rest-service) and the REST tutorial in [docs/tutorials/index.md](../docs/tutorials/index.md#5-tutorial-4-serve-models-over-http).

## 6. Test status and known issues

| Example | Runs on the main branch | Known issues |
| --- | --- | --- |
| `scripts/calculate_ndvi.py` | Yes | No no-data value in the output file |
| `scripts/detect_ships.py` | Yes | Untrained starter model |
| `serving/api.py` | Yes (all five routes answered as shown in Section 5.2) | No authentication or limits; needs python-multipart; untrained models |

The runs were made on 24 September 2026 in a Linux container with CPython 3.13 and the CPU build of PyTorch, with the commands shown above. None of the examples fails to run; because of the limitations listed in the table they SHOULD be read as illustrations. Improvements are welcome through the process in [CONTRIBUTING.md](../CONTRIBUTING.md).

## 7. Licence

The examples are part of Unbihexium and are licensed under the Mozilla Public License 2.0 [4], like the rest of the repository ([LICENSE.txt](../LICENSE.txt)). Code copied from them into another project keeps that licence for the copied files; this note is not legal advice.

## References

[1] Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third Earth Resources Technology Satellite-1 Symposium, NASA SP-351, 309-317. 1974. <https://ntrs.nasa.gov/citations/19740022614>

[2] Butler, H., Daly, M., Doyle, A., Gillies, S., Hagen, S. and Schaub, T. RFC 7946: The GeoJSON Format. IETF. 2016. <https://www.rfc-editor.org/rfc/rfc7946>

[3] Ramirez, S. FastAPI. 2026. <https://github.com/fastapi/fastapi>

[4] Mozilla Foundation. Mozilla Public License, version 2.0. 2012. <https://mozilla.org/MPL/2.0/>

[5] Bradner, S. RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[6] Leiba, B. RFC 8174: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file examples/README.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
