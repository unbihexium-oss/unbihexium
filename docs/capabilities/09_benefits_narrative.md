<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/capabilities/09_benefits_narrative.md
Title       : Capability Domain 09: Benefits Narrative and Reportable Outputs
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Capability Domain 09: Benefits Narrative and Reportable Outputs

| Field | Value |
| --- | --- |
| Document | UBX-DOC-609 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch |

## Abstract

This document explains what Unbihexium can actually report and summarise, so that anyone who describes the benefits of a workflow built with the library can base the description on measurable outputs. It lists the capabilities the library registers, the quantities its functions and result records produce (counts, areas, fractions, zonal statistics, change components, accuracies with confidence intervals, validation statistics), and the records that make a result reproducible and verifiable (pipeline runs, SHA-256 evidence, provenance, product metadata). It is written for project leads, analysts and reviewers who prepare reports, proposals or evaluations, and for contributors. It replaces an earlier version that contained return-on-investment tables, sector case studies and accuracy figures that were not derived from the software; the project publishes no economic figures, performance claims or accuracy figures, and the learned models of the model zoo are untrained starter models. All examples were executed against the current code.

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [What the library provides](#2-what-the-library-provides)
3. [Quantities the library reports](#3-quantities-the-library-reports)
4. [Uncertainty and accuracy](#4-uncertainty-and-accuracy)
5. [Reproducibility and evidence](#5-reproducibility-and-evidence)
6. [Model families and benefit statements](#6-model-families-and-benefit-statements)
7. [Writing a benefits statement](#7-writing-a-benefits-statement)
8. [Limitations](#8-limitations)
9. [Related documents](#9-related-documents)
10. [References](#references)

## 1. Purpose and scope

### 1.1 Purpose

Benefits of Earth observation are realised by the organisations that use it, in their own context: a map is useful when it answers a decision question with known accuracy at an acceptable cost. A software library cannot measure that benefit. What it can do is produce well-defined quantities, state their uncertainty, and record how they were produced. This document describes those quantities and records, and gives rules for turning them into a benefits narrative that a reviewer can check.

### 1.2 Scope

The document covers the reporting functions of the packages `unbihexium.registry`, `unbihexium.ai` (result records), `unbihexium.analysis`, `unbihexium.metrics` and `unbihexium.core` (pipelines, evidence, products). It does not describe algorithms in detail; those are documented in the other capability documents listed in [index.md](index.md). No model family has this document as its domain: the capability registry has no "benefits" domain, and none of the 130 families of `src/unbihexium/zoo/catalog.yaml` belongs to it.

### 1.3 Conventions

The key words MUST, MUST NOT, SHOULD and MAY in sections 6 and 7 are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in capitals. They state the expectations of this project for statements about the software; they do not restrict the licence (Mozilla Public License 2.0, [LICENSE.txt](../../LICENSE.txt)). This document is not legal advice.

## 2. What the library provides

The capability registry lists what the library can do. Every capability has a domain, a maturity level and the entry points that implement it. There are two kinds:

- 17 library capabilities: algorithms and formats implemented directly in the library (input and output, spectral indices, SAR processing, terrain, geostatistics, spatial analysis, preprocessing, postprocessing, metrics, visualisation, the model zoo, training and serving). They have maturity `stable`.
- 130 model capabilities, one per model family. The 7 spectral index families compute published formulas and have maturity `stable`; the other 123 families are untrained starter models with maturity `beta` and the tag `requires_training: true`.

The registry can be queried at run time, so a report can state exactly which capabilities a given installation offers:

```python
from unbihexium.registry import CapabilityRegistry

counts = CapabilityRegistry.domain_counts()
print(sum(counts.values()), dict(sorted(counts.items())))
print({m: len(CapabilityRegistry.by_maturity(m)) for m in ("stable", "beta")})
flood = CapabilityRegistry.require("sar_flood_detector")
print(flood.maturity.value, flood.tags, flood.models[0], flood.cli_command)
```

Output:

```text
147 {'agriculture': 16, 'ai': 10, 'analysis': 12, 'assets': 7, 'defense': 5, 'energy': 5, 'environment': 11, 'forestry': 4, 'imaging': 23, 'indices': 8, 'io': 5, 'risk': 10, 'sar': 9, 'tourism': 4, 'urban': 10, 'water': 8}
{'stable': 24, 'beta': 123}
beta {'requires_training': 'true'} sar_flood_detector_tiny unbihexium predict sar_flood_detector_base INPUT OUTPUT
```

The same totals are printed by `unbihexium info` (`Registered capabilities: 147`, `Model zoo models: 520 (catalogue 2.0.0)`, `Registered pipelines: 5`).

## 3. Quantities the library reports

### 3.1 Overview

The table lists the quantities that can be reported directly from library outputs. Area quantities are in square units of the coordinate reference system unless stated otherwise; converting to hectares or square kilometres is the caller's responsibility and requires a projected CRS with metric units.

| Question | Function or record | Reported quantity |
| --- | --- | --- |
| How many objects of each class were detected? | `DetectionResult.count`, `counts_by_class()`, `to_geojson()` | Counts per class and the boxes with class, confidence and map coordinates |
| What share and area of an image does each class cover? | `SegmentationResult.class_fractions()`, `class_areas()` | Fraction of labelled pixels, pixel count times the pixel area of the affine transform |
| What are the statistics of an estimated variable? | `RegressionResult.summary()`, `Raster.statistics()` | Mean, minimum, maximum and standard deviation; percentiles for rasters |
| What are the statistics of a layer per zone (field, district, catchment)? | `analysis.zonal_statistics`, `zonal_table`, `rasterize_zones` | Count, sum, mean, std, min, max, range, median, majority, minority, variety and percentiles per zone [3] |
| What changed between two dates? | `metrics.transition_matrix`, `transition_summary`, `change_map` | From-to counts and the persistence, gross gain, gross loss, net change, swap and total change of every class [4] |
| How well does a change map agree with reference data? | `metrics.change_detection_metrics` | Detection rate, false alarm rate, missed detection rate, precision, F1, IoU, overall accuracy and kappa |
| How accurate is a class map, and what are the class areas? | `metrics.accuracy_assessment`, `stratified_area_estimate` | Overall, user's and producer's accuracy, kappa, quantity and allocation disagreement; unbiased areas with confidence intervals (section 4) |
| How close are estimates to reference measurements? | `metrics.regression_report` | Bias, MAE, RMSE, unbiased RMSE, relative RMSE, $R^2$, Pearson's $r$, slope and intercept |
| How close is an enhanced image to a reference? | `metrics.psnr`, `ssim`, `sam`, `ergas`, `q_index` | Image quality measures (see [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md)) |

### 3.2 Example: change, zonal statistics and agreement

```python
import numpy as np

from unbihexium.analysis import zonal_table
from unbihexium.metrics import change_detection_metrics, transition_matrix, transition_summary

rng = np.random.default_rng(4)

# Two class maps of the same 10 m grid (0 water, 1 forest, 2 built-up) and the change between them.
before = rng.choice(3, size=(100, 100), p=[0.1, 0.7, 0.2])
after = before.copy()
cleared = (before == 1) & (rng.random(before.shape) < 0.05)
after[cleared] = 2
matrix = transition_matrix(before, after, labels=[0, 1, 2])
summary = transition_summary(matrix, classes=["water", "forest", "built_up"])
pixel_ha = 10 * 10 / 10_000
for name, row in summary.items():
    print(f"{name:>8}: gain {row['gain'] * pixel_ha:5.2f} ha, loss {row['loss'] * pixel_ha:5.2f} ha, net {row['net_change'] * pixel_ha:+6.2f} ha")

# Per-zone statistics of a continuous layer (for example NDVI) in two administrative zones.
ndvi = np.where(after == 1, 0.7, 0.2) + rng.normal(0, 0.05, size=after.shape)
zones = np.zeros(after.shape, dtype=int)
zones[:, 50:] = 1
table = zonal_table(ndvi, zones, stats=["count", "mean", "std"])
print({stat: {int(z): round(v, 3) for z, v in values.items()} for stat, values in table.items()})

# Agreement of a detected change mask with a reference change mask.
detected = cleared & (rng.random(before.shape) < 0.9)
metrics = change_detection_metrics(cleared, detected)
print({k: round(metrics[k], 3) for k in ("detection_rate", "false_alarm_rate", "precision", "f1")})
```

Output:

```text
   water: gain  0.00 ha, loss  0.00 ha, net  +0.00 ha
  forest: gain  0.00 ha, loss  3.31 ha, net  -3.31 ha
built_up: gain  3.31 ha, loss  0.00 ha, net  +3.31 ha
{'count': {0: 5000.0, 1: 5000.0}, 'mean': {0: 0.528, 1: 0.537}, 'std': {0: 0.243, 1: 0.241}}
{'detection_rate': 0.906, 'false_alarm_rate': 0.0, 'precision': 1.0, 'f1': 0.951}
```

The maps are synthetic; the example shows the form of the outputs, not a result about any real area.

### 3.3 Example: an exact index model

The spectral index families are the only models of the zoo that give meaningful outputs without training. The example runs the NDVI model [5] on a two-band raster and checks it against the formula of `unbihexium.indices`.

```python
import numpy as np
from rasterio.transform import from_origin

from unbihexium.ai import predict
from unbihexium.indices import ndvi
from unbihexium.io import write_geotiff

rng = np.random.default_rng(2)
red = rng.uniform(0.02, 0.1, size=(64, 64))
nir = rng.uniform(0.2, 0.5, size=(64, 64))
write_geotiff(np.stack([red, nir]).astype("float32"), "red_nir.tif", crs="EPSG:32635",
              transform=from_origin(500000, 6700000, 10, 10))

result = predict("ndvi_calculator_tiny", "red_nir.tif")  # Exact formula, no training needed.
print(result.names, result.units, {k: round(v, 4) for k, v in result.summary()["ndvi"].items()})
print(np.allclose(result.output("ndvi"), ndvi(nir=nir, red=red), atol=1e-6))
```

Output:

```text
['ndvi'] [] {'mean': 0.6993, 'min': 0.3463, 'max': 0.9202, 'std': 0.1216}
True
```

`predict` builds a catalogue model in memory with its deterministic starter weights and checks them against the published digest; `unbihexium zoo build` stores a model in the directory named by `UNBIHEXIUM_CACHE` (default `~/.cache/unbihexium`).

## 4. Uncertainty and accuracy

A quantity without an uncertainty cannot support a claim of benefit. Two functions of `unbihexium.metrics` address the most common case, a class map used to report areas:

- `accuracy_assessment(matrix)` reports the overall accuracy, user's and producer's accuracies, F1, IoU, Cohen's kappa [6] and the quantity and allocation disagreement of Pontius and Millones [7] from an error matrix with reference classes in rows and map classes in columns.
- `stratified_area_estimate(matrix, mapped_area, confidence=0.95)` computes unbiased class areas, accuracies and their confidence intervals from a stratified random sample, following the good practice recommendations of Olofsson et al. [8]. Area obtained by counting map pixels is biased by the classification errors; the estimator corrects it with the reference labels of the sample. `sample_allocation` computes the stratum sample sizes for a target standard error.

The formulas and a worked example are in section 8 of [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md). For continuous variables, `regression_report` gives the validation statistics against independent reference measurements.

## 5. Reproducibility and evidence

### 5.1 Records

A benefit that cannot be reproduced cannot be verified. The library provides the following records:

| Record | Module | Content |
| --- | --- | --- |
| `PipelineRun` | `unbihexium.core.pipeline` | Status, per-step name, status, times and duration, inputs and outputs as text, the seed, and a provenance record; `to_json()` writes it |
| `ProvenanceRecord` | `unbihexium.core.evidence` | Inputs, outputs and models of one run with SHA-256 digests [9], configuration, software environment (interpreter, platform, library versions) and links to parent runs, following the entity-activity-agent structure of W3C PROV [10]; the record digest detects later changes; `verify_outputs()` lists outputs whose files no longer match |
| `Evidence` | `unbihexium.core.evidence` | One artefact identified by its SHA-256 digest, with `verify()` |
| `Product`, `ProductMetadata` | `unbihexium.core.product` | Derived product with CRS, bounds, resolution, source scenes, processing chain, optional quality score, licence, SHA-256 digest of the data and a STAC item |
| Model digests | `src/unbihexium/zoo/digests.json` | Published SHA-256 digests of the deterministic starter weights, checked when a model is built or verified (`unbihexium zoo verify`) |

When a pipeline configuration has a seed, the Python, NumPy and, when loaded, PyTorch random number generators are seeded before the first step, so repeated runs with the same inputs give the same outputs. Releases of the package are built by `.github/workflows/release.yml`, signed with Sigstore and published with SLSA provenance and GitHub artifact attestations (releases 1.0.0 and 1.0.1 predate signing); see [SECURITY.md](../../SECURITY.md) and [docs/security/supply_chain_security.md](../security/supply_chain_security.md).

### 5.2 Example: a recorded run

```python
import json

import numpy as np
from rasterio.transform import from_origin

from unbihexium.core import Pipeline, PipelineConfig
from unbihexium.indices import ndvi
from unbihexium.io import read_geotiff, write_geotiff

rng = np.random.default_rng(0)
write_geotiff(rng.uniform(0.02, 0.5, size=(4, 32, 32)).astype("float32"), "scene.tif",
              crs="EPSG:32635", transform=from_origin(500000, 6700000, 10, 10))

pipeline = Pipeline(PipelineConfig("ndvi_report", "NDVI report", seed=0))


@pipeline.step()
def compute(values):
    data, meta = read_geotiff(values["input"])
    index = ndvi(nir=data[3], red=data[2]).astype("float32")
    path = write_geotiff(index, "ndvi.tif", crs=meta["crs"], transform=meta["transform"])
    return {"output": str(path), "mean_ndvi": float(np.nanmean(index))}


run = pipeline.run({"input": "scene.tif"})
record = json.loads(run.to_json("run.json"))
print(run.status.value, [s["name"] for s in record["steps"]], round(run.results["mean_ndvi"], 3))
provenance = run.provenance
print(len(provenance.inputs), len(provenance.outputs), provenance.verify_outputs())
print(sorted(provenance.environment)[:4])
```

Output:

```text
completed ['compute'] -0.006
1 2 []
['implementation', 'numpy', 'platform', 'python']
```

The input file and the output file are recorded as `Evidence` with their digests; the value `mean_ndvi` is recorded as text. An empty list from `verify_outputs()` means that every output file still matches its recorded digest.

## 6. Model families and benefit statements

Of the 520 models of the zoo (130 families in the variants `tiny`, `base`, `large` and `mega`), only the 28 models of the 7 spectral index families compute exact formulas. The other 492 models are untrained starter models: complete, trainable networks with deterministically initialised weights whose outputs carry no information until they are trained on labelled data for the user's area and sensor (see [docs/model_zoo/training.md](../model_zoo/training.md) and section 2 of [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md)). The domain of a family describes its intended application, not a validated product. This also applies to families whose names suggest economic outputs, such as `business_valuation`, `economic_spatial_assessor` and `resource_allocation` (domain `analysis`): they are scene regression networks that must be trained on reference values before their estimates mean anything.

Consequently:

- A statement about the benefit of a model output MUST refer to a model trained or fine-tuned by the author of the statement, and MUST report its accuracy on an independent reference sample.
- A statement MUST NOT present the outputs of an untrained starter model as results, and MUST NOT attribute accuracy figures to the project, which publishes none.

## 7. Writing a benefits statement

A benefits statement about a workflow built with Unbihexium SHOULD contain the following elements, each of which the library can supply or record:

| Element | Content | Source in the library |
| --- | --- | --- |
| Question | The decision the output supports, for example the area of forest cleared in a district in one year | (author) |
| Data | Sensors, acquisition dates, processing levels and catalogue identifiers | `SceneMetadata`, STAC items, `Product.metadata.source_scenes` |
| Method | Functions and models with the library version, model identifiers and weights digests | `PipelineRun`, `ProvenanceRecord`, `unbihexium info`, `unbihexium zoo info` |
| Result | The reported quantity with units | Section 3 |
| Uncertainty | Accuracy and confidence intervals from an independent sample | Section 4 |
| Evidence | Digests of inputs, outputs and models, and where the records are kept | Section 5 |
| Limitations | Known error sources, area of validity, and the fact that starter models were trained by the author | (author), [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) |

In addition:

- Economic figures (costs, savings, return on investment, payback periods) MAY be included when they come from the author's own accounting; they SHOULD be stated separately from the quantities of the library and with their own sources, because the library neither computes nor validates them.
- Comparisons with other methods SHOULD state the data, the metric and the direction of the difference, and SHOULD be reproducible from the recorded runs.
- Statements SHOULD cite the software as described in [CITATION.cff](../../CITATION.cff) and the methods by their primary sources, which the capability documents list.

## 8. Limitations

- The library does not compute economic value, cost, return on investment or any other financial measure, and the project publishes no such figures, case studies or customer results.
- There is no report generator; the quantities of section 3 are Python values, JSON records and raster or vector files that the author assembles.
- Area quantities are exact only in equal-area or local projected coordinate systems; `class_areas` and zonal counts use the pixel area of the affine transform.
- The quality of any reported quantity depends on the input data and, for learned models, on the training data supplied by the user.

## 9. Related documents

- [README.md](../../README.md): project overview, installation and status of the model zoo.
- [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md): intended uses, limits of the starter models and responsibilities of deployers.
- [11_resolution_metadata_qa.md](11_resolution_metadata_qa.md): metadata, quality checks and the formulas of the accuracy metrics.
- [docs/model_zoo/model_catalog.md](../model_zoo/model_catalog.md) and [docs/model_zoo/training.md](../model_zoo/training.md): the model catalogue and how to train a family.
- [index.md](index.md): overview of all capability documents.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. IETF RFC 2119. 1997. <https://doi.org/10.17487/RFC2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. IETF RFC 8174. 2017. <https://doi.org/10.17487/RFC8174>

[3] Tomlin, C. D. Geographic Information Systems and Cartographic Modeling. Prentice Hall, Englewood Cliffs NJ. 1990.

[4] Pontius, R. G., Shusas, E., McEachern, M. Detecting important categorical land changes while accounting for persistence. Agriculture, Ecosystems and Environment 101(2-3), 251-268. 2004. <https://doi.org/10.1016/j.agee.2003.09.008>

[5] Rouse, J. W., Haas, R. H., Schell, J. A., Deering, D. W. Monitoring vegetation systems in the Great Plains with ERTS. Third ERTS Symposium, NASA SP-351, 309-317. 1974.

[6] Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1), 37-46. 1960. <https://doi.org/10.1177/001316446002000104>

[7] Pontius, R. G., Millones, M. Death to kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. International Journal of Remote Sensing 32(15), 4407-4429. 2011. <https://doi.org/10.1080/01431161.2011.552923>

[8] Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., Wulder, M. A. Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148, 42-57. 2014. <https://doi.org/10.1016/j.rse.2014.02.015>

[9] National Institute of Standards and Technology. Secure Hash Standard (SHS), FIPS PUB 180-4. 2015. <https://doi.org/10.6028/NIST.FIPS.180-4>

[10] Moreau, L., Missier, P. (eds.). PROV-DM: The PROV Data Model. W3C Recommendation. 2013. <https://www.w3.org/TR/prov-dm/>

<!--
=============================================================================
End of file docs/capabilities/09_benefits_narrative.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
