<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/security/responsible_use.md
Title       : Responsible Use: Technical Guidance
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Responsible Use: Technical Guidance

| Field | Value |
| --- | --- |
| Document | UBX-DOC-SEC-RESPONSIBLE-USE |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch, including the model zoo |

## Abstract

The normative policy on intended, unsupported and dual-use applications of Unbihexium is [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) in the repository root. This companion document is for developers and operators who build systems with the library: it shows how to put that policy into practice with the features the code actually provides, namely the `requires_training` flag of the model catalogue, provenance records, the security settings of the REST service and the labelling of generated imagery. It adds no obligations of its own; where the two documents differ, [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) prevails. This document is not legal advice.

## Contents

- [1. Relationship to the Policy](#1-relationship-to-the-policy)
- [2. Detecting Starter Models](#2-detecting-starter-models)
- [3. Recording Provenance](#3-recording-provenance)
- [4. Validation Before Use](#4-validation-before-use)
- [5. Sensitive Components](#5-sensitive-components)
- [6. Deploying the REST Service](#6-deploying-the-rest-service)
- [7. Generated Imagery](#7-generated-imagery)
- [8. Reporting](#8-reporting)
- [References](#references)

## 1. Relationship to the Policy

### 1.1 Conventions

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119 [1] and RFC 8174 [2]. In this document they only restate expectations that [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) already sets, and point to the technical means of meeting them.

### 1.2 What the Software Does and Does Not Enforce

The library does not restrict how it is used: it has no usage telemetry, no licence keys and no allow-lists of purposes, and the Mozilla Public License 2.0 grants the right to run it for any purpose. The technical measures described below make the limits of the software visible and help deployers keep records; responsibility for lawful and appropriate use remains with the deployer, as stated in Section 6 of the policy.

## 2. Detecting Starter Models

All 520 models of the model zoo (130 families in 4 variants) are untrained starter models with deterministic weights, except the 28 models of the 7 spectral index families, which compute exact formulas. The catalogue marks this for every model with `requires_training`, which is available:

- in Python as `unbihexium.zoo.get_model(model_id).requires_training`;
- in `unbihexium zoo info <model_id>` and `unbihexium zoo list --json`;
- in the REST service, in the model descriptions of `GET /models` and `GET /models/{model_id}` and in every response of `POST /predict/{model_id}`;
- at the top of each model card under [model_zoo/cards/](../../model_zoo/cards/).

The command line `unbihexium predict` does not warn when it runs a starter model, and detection output (GeoJSON) records the `model_id` but not the flag. Applications that present results to other people SHOULD therefore check the flag themselves and SHOULD refuse or clearly mark outputs of models with `requires_training` set to true, as required by item 6 of Section 4 of the policy.

## 3. Recording Provenance

A result can only be judged if it can be traced to the exact software and model that produced it. The following record identifies both; it runs without PyTorch because `get_model` reads catalogue metadata only.

```python
import json

import unbihexium
from unbihexium.zoo import catalog_version, get_model

entry = get_model("ship_detector_tiny")
if entry.requires_training:
    print(f"{entry.model_id} has untrained starter weights; its output is not meaningful")

record = {
    "library_version": unbihexium.__version__,
    "catalog_version": catalog_version(),
    "model_id": entry.model_id,
    "weights_digest": entry.weights_digest,
    "requires_training": entry.requires_training,
}
print(json.dumps(record, indent=2))
```

Output:

```text
ship_detector_tiny has untrained starter weights; its output is not meaningful
{
  "library_version": "1.0.1",
  "catalog_version": "2.0.0",
  "model_id": "ship_detector_tiny",
  "weights_digest": "e278457bcbb75b891f2b0730889fab1a33a7f346fa695e33cc5545b5952e2899",
  "requires_training": true
}
```

For a model trained by the user, the record SHOULD contain the digest of the trained checkpoint (returned by `save_checkpoint` and by `load_checkpoint(path).digest()`) instead of the starter digest, together with a reference to the training data and the validation report. How digests are checked is described in [model_integrity.md](model_integrity.md).

## 4. Validation Before Use

Section 2.3 of the policy requires validation of trained models on independent reference data from the area, sensor and period of intended use. The library provides the tools for this: `unbihexium evaluate` evaluates a model on a dataset split, and the `unbihexium.metrics` package implements accuracy assessment, segmentation, change detection, regression and image quality metrics (for example `confusion_matrix`, `cohen_kappa`, `iou`, `rmse` and `ssim`). The project publishes no accuracy figures for any model; results obtained with these tools describe only the model and data that were evaluated. Training and evaluation are described in [docs/model_zoo/training.md](../model_zoo/training.md).

## 5. Sensitive Components

The families listed in Section 5.1 of the policy (for example `military_objects_detector`, `target_detector`, `border_monitor`, `security_monitor`, `maritime_awareness`, the vessel, aircraft and vehicle detectors, change detection, viewshed and route planning) are ordinary catalogue entries and are listed with `unbihexium zoo list --domain defense` and the corresponding domains. They ship as untrained architectures. Technical measures that deployers SHOULD apply when they train or operate them:

- keep training data, trained checkpoints and outputs under access control, and log who ran which model on which data;
- keep a qualified person in the decision loop; integrate outputs as leads for human review, never as triggers for automated action;
- document the operating conditions under which each trained model was validated, and restrict its use to them.

## 6. Deploying the REST Service

A service that accepts imagery from others processes data that may be personal (see [PRIVACY.md](../../PRIVACY.md)). The service defaults are suitable for local use only: no API key, CORS open to all origins, no rate limit and plain HTTP. Before exposing it, operators SHOULD at least set `UNBIHEXIUM_SERVING__API_KEY`, restrict `UNBIHEXIUM_SERVING__CORS_ORIGINS`, set `UNBIHEXIUM_SERVING__RATE_LIMIT_PER_MINUTE` and terminate TLS in a reverse proxy, as described in [SECURITY.md](../../SECURITY.md), Section 8. The container configuration is described in [docs/operations/docker.md](../operations/docker.md).

## 7. Generated Imagery

Outputs of the `synthetic_imagery` models (optical images synthesised from SAR) and of super-resolution models are generated, not observed. The library does not write a marker into such outputs. Deployers MUST label them as generated wherever they are shown or shared (item 7 of Section 4 of the policy, and the transparency obligations described in [COMPLIANCE.md](../../COMPLIANCE.md)), for example in the file name, in the GeoTIFF metadata tags and in the captions of any figure.

## 8. Reporting

Concerns about misuse, bias or harmful behaviour are raised as described in Section 8 of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md). Security vulnerabilities MUST be reported privately through <https://github.com/unbihexium-oss/unbihexium/security/advisories/new>, as described in [SECURITY.md](../../SECURITY.md).

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

<!--
=============================================================================
End of file docs/security/responsible_use.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
