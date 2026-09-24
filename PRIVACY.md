<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : PRIVACY.md
Title       : Privacy Statement
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Privacy Statement

| Field | Value |
| --- | --- |
| Document | UBX-DOC-202 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch |

## Abstract

This statement describes, for users, operators and data protection reviewers, what data the Unbihexium software processes, what it stores on disk, when it opens network connections and what the project itself receives. It is based on the source code of the library, the command line interface and the REST service at the date of review, and each statement names the module that implements the behaviour so that it can be checked. In short: Unbihexium contains no telemetry, no analytics and no automatic update checks, it processes data locally, and it connects to the network only when the user asks it to query a STAC API or to download a checkpoint from a URL the user registered. The statement also sets out the responsibilities of those who deploy the software to process personal data. It is not legal advice.

## Contents

- [1. Introduction](#1-introduction)
- [2. Summary](#2-summary)
- [3. Data Processed by the Library and the CLI](#3-data-processed-by-the-library-and-the-cli)
- [4. Network Access](#4-network-access)
- [5. Local Storage](#5-local-storage)
- [6. The REST Service](#6-the-rest-service)
- [7. Data Received by the Project](#7-data-received-by-the-project)
- [8. Responsibilities of Deployers](#8-responsibilities-of-deployers)
- [9. Changes and Contact](#9-changes-and-contact)
- [References](#references)

## 1. Introduction

### 1.1 Scope

This statement covers the `unbihexium` Python package, its command line interface (`unbihexium`), its REST service (`unbihexium.serving`), the container image built from the [Dockerfile](Dockerfile) and the project's own channels (the GitHub repository and the maintainer's e-mail address). It does not cover third-party libraries that Unbihexium calls (for example GDAL through rasterio, PyTorch or ONNX Runtime), the services of GitHub and PyPI, or deployments operated by others; their own privacy terms apply.

### 1.2 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. They are used only in [Section 8](#8-responsibilities-of-deployers), where they describe what the maintainers expect of deployers; they do not create legal obligations.

### 1.3 Roles

Unbihexium is software, not a service. When you run it, you (or your organisation) decide which data it processes and for what purpose, so under the General Data Protection Regulation (GDPR) [3] you are the controller of that processing. The project and its maintainer do not receive, host or process the data you analyse with the software.

## 2. Summary

| Question | Answer | Where to check |
| --- | --- | --- |
| Does the software send telemetry, usage statistics or crash reports? | No. There is no telemetry code and no opt-in either. | [src/unbihexium/](src/unbihexium/) |
| Does it check for updates or contact a project server? | No. | [src/unbihexium/](src/unbihexium/) |
| Does it connect to the network by itself? | No. It connects only to URLs the user supplies (Section 4). | [src/unbihexium/io/stac.py](src/unbihexium/io/stac.py), [src/unbihexium/zoo/store.py](src/unbihexium/zoo/store.py) |
| Are model zoo models downloaded? | No. Catalogue models are built locally from their identifier. | [src/unbihexium/zoo/store.py](src/unbihexium/zoo/store.py) |
| What does it write to disk? | Outputs the user requests, and the model cache (Section 5). | [src/unbihexium/zoo/store.py](src/unbihexium/zoo/store.py) |
| Does the REST service store uploaded images? | No. Images are processed in memory and discarded after the response. | [src/unbihexium/serving/](src/unbihexium/serving/) |
| Does the REST service log requests? | The service code does not; the ASGI server (uvicorn) writes its standard access log to the console. | Section 6 |

## 3. Data Processed by the Library and the CLI

### 3.1 Input Data

The library and the CLI read the files, arrays and documents that the calling program or the user passes to them: raster images (GeoTIFF, Cloud Optimized GeoTIFF, Zarr), vector data (GeoJSON, GeoParquet), STAC items and catalogues, configuration files and model checkpoints. Processing happens in the memory of the local process. Unbihexium does not copy these inputs elsewhere and does not keep them after the call returns, except where the user explicitly writes an output.

Earth observation data can be personal data. Very high resolution imagery can show individuals, vehicles, number plates or private property, and location data linked to an identifiable person is personal data under the GDPR. The software does not detect or classify personal data; this assessment belongs to the user (Section 8).

### 3.2 Outputs

Outputs (index rasters, predictions, vector layers, reports) are written only to the paths the user chooses, for example with `unbihexium index` or `unbihexium predict`.

### 3.3 Provenance Records

The audit trail in [src/unbihexium/core/evidence.py](src/unbihexium/core/evidence.py) can record, when the user creates such records, the SHA-256 digests, sizes, file names and locations of the inputs and outputs of a run, the configuration and a description of the environment (Python version, implementation, the operating system string reported by `platform.platform()` and the versions of NumPy, Unbihexium and loaded numerical libraries). The record contains no user name or host name. It stays in memory unless the user saves it, and it is never transmitted by the library. File paths can contain user names; review records before sharing them.

### 3.4 Logging

The library logs through the Python `logging` module to a stream on the console ([src/unbihexium/utils/log.py](src/unbihexium/utils/log.py)); the level is set with `UNBIHEXIUM_LOG_LEVEL` and defaults to `WARNING`. It does not create log files.

## 4. Network Access

Unbihexium opens a network connection in the following cases only, each initiated by the user.

1. **STAC API search.** `unbihexium.io.stac.STACClient` sends HTTP requests to the STAC API URL given by the user, and to the pagination (`next`) links that the API returns, with the headers the user sets (for example an authorisation token), and receives the search results. Local STAC catalogues are read from disk without network access, and remote child links are not followed.
2. **Registered checkpoint downloads.** A model that the user registers with a download URL (`source="url"`) is fetched once into the model cache ([src/unbihexium/zoo/store.py](src/unbihexium/zoo/store.py)). The 520 catalogue models of the zoo (130 families in 4 variants) are never downloaded: their starter weights are generated locally and deterministically, and they are untrained, except the 7 spectral index families (28 models) that compute exact formulas.
3. **Remote paths passed to third-party readers.** If the user passes a URL instead of a local path to a function that reads rasters, the underlying library (for example GDAL through rasterio) may fetch the data over the network under its own configuration.

Requests made by Unbihexium itself use the `requests` library with HTTPS certificate verification enabled by default. The remote server receives the usual metadata of an HTTP request, such as the client IP address and a `User-Agent` header naming the `requests` library. Installing the package with pip contacts PyPI, and building or pulling the container image contacts the registries involved; these actions are outside the software.

## 5. Local Storage

### 5.1 Model Cache

Models built or downloaded with `unbihexium zoo build`, `ensure_model` or `download_model` are stored under `$UNBIHEXIUM_CACHE/models/<model_id>/`, with `UNBIHEXIUM_CACHE` defaulting to `~/.cache/unbihexium`. Each model directory holds `model.pt`, optionally `model.onnx`, `config.json` and `model.sha256`. The cache contains model weights and metadata only, never user imagery. In the container image the cache is `/home/unbihexium/.cache/unbihexium`, declared as a volume.

The cache can be filled, inspected and removed with the CLI:

```bash
unbihexium zoo build ship_detector_tiny
unbihexium zoo where ship_detector_tiny
unbihexium zoo clear ship_detector_tiny --yes
unbihexium zoo clear --yes
```

`load_model` builds catalogue models in memory and writes nothing to disk; only models registered with a download URL are fetched into the cache first.

### 5.2 Configuration

Configuration is read from a YAML file named by `UNBIHEXIUM_CONFIG` and from `UNBIHEXIUM_<SECTION>__<KEY>` environment variables ([src/unbihexium/config/settings.py](src/unbihexium/config/settings.py)). Unbihexium does not write configuration files. Secrets such as the REST service API key SHOULD be supplied through environment variables or a secret store rather than committed files.

## 6. The REST Service

The REST service ([src/unbihexium/serving/app.py](src/unbihexium/serving/app.py)) receives images in the body of `POST` requests, decodes them into memory, runs the requested model and returns a JSON summary. It does not write uploaded images, results or request metadata to disk or to a database. Opened models are kept in a small in-memory cache (four models by default) that contains no request data.

The service keeps the following transient data in memory:

- when the per-client rate limit is enabled (`UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE` greater than `0`), a token bucket keyed by the client IP address, or by the `X-Forwarded-For` address when that is explicitly trusted; the buckets are held only in the process memory and are lost on restart;
- when an API key is configured, the key itself, which is compared with the `X-API-Key` header of each request and not logged.

The service code does not log requests. The ASGI server that runs it, typically uvicorn, prints an access log line per request (client address and port, method, path and status code) to the console by default; its retention is decided by the operator's logging setup. Operators who need to avoid this can start uvicorn with `--no-access-log`.

## 7. Data Received by the Project

The project receives personal data only when people contact it:

- **GitHub issues, pull requests, discussions and security advisories** are processed by GitHub under its own privacy statement [5]. Content posted publicly stays public; do not post personal data, credentials or non-public imagery in public issues.
- **E-mail** to <yunus.z.imanov@helsinki.fi> is received by the maintainer through the University of Helsinki mail service and is used only to answer the message and, for security reports, to handle the vulnerability as described in [SECURITY.md](SECURITY.md).
- **Contributions** record the author name and e-mail address in the Git history, which is public and permanent; the project also lists contributors in [AUTHORS.md](AUTHORS.md) and [CITATION.cff](CITATION.cff).
- **Package downloads** from PyPI and the GitHub Container Registry are logged by those services; the project sees at most aggregate download statistics that they publish.

The project does not operate analytics, a mailing list or any other service that collects data from users of the software.

## 8. Responsibilities of Deployers

Organisations that use Unbihexium to process imagery or location data that relates to identifiable people act as controllers under the GDPR [3] or equivalent law. The maintainers expect that they:

- MUST establish a lawful basis for the processing (GDPR Article 6) and apply data minimisation, for example by using the coarsest resolution that serves the purpose;
- MUST carry out a data protection impact assessment where GDPR Article 35 requires it, which is likely for systematic monitoring of publicly accessible areas;
- SHOULD enable the API key, restrict CORS origins, terminate TLS in front of the REST service and configure access log retention, as described in [SECURITY.md](SECURITY.md);
- SHOULD review provenance records and outputs for file paths or locations that identify people before sharing them;
- MAY consult the European Data Protection Board guidelines on video devices [4] for principles that also apply to imagery of people.

[RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) lists uses of the software that the project does not support, including surveillance of individuals without a lawful basis.

## 9. Changes and Contact

This statement is reviewed when code that affects data handling changes, and at least once a year. Changes are recorded in the Git history of this file and summarised in [CHANGELOG.md](CHANGELOG.md). Questions about this statement can be sent to <yunus.z.imanov@helsinki.fi> or raised with the "Compliance, licensing and ethics" issue form. Report a privacy defect that is also a security vulnerability privately, as described in [SECURITY.md](SECURITY.md).

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] European Parliament and Council. Regulation (EU) 2016/679 (General Data Protection Regulation). 2016. <https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng>

[4] European Data Protection Board. Guidelines 3/2019 on processing of personal data through video devices, version 2.0. 2020. <https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-32019-processing-personal-data-through-video_en>

[5] GitHub. GitHub General Privacy Statement. 2026. <https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement>

<!--
=============================================================================
End of file PRIVACY.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
