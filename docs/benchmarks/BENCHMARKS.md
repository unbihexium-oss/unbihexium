<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/benchmarks/BENCHMARKS.md
Title       : Benchmarks
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Benchmarks

| Field | Value |
| --- | --- |
| Document | UBX-DOC-904 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 2.0.0 and the main branch |

## Abstract

This document describes the performance benchmark tests of Unbihexium, which live in [tests/benchmarks/test_performance.py](../../tests/benchmarks/test_performance.py), explains exactly what each test measures and asserts, shows how to run them, and reports the figures obtained by running them for this revision, together with the environment and the command used. It is written for contributors who want to detect performance regressions and for users who want an order-of-magnitude impression of CPU inference cost. The figures describe one small, shared CPU container and are not a performance guarantee for any other system. The benchmarks measure speed and memory only: the models involved are untrained starter models, so no accuracy is measured or reported.

## Contents

- [1. Introduction](#1-introduction)
- [2. The benchmark tests](#2-the-benchmark-tests)
- [3. Running the benchmarks](#3-running-the-benchmarks)
- [4. Measured results](#4-measured-results)
- [5. Interpretation and limits](#5-interpretation-and-limits)
- [6. Adding or reporting benchmarks](#6-adding-or-reporting-benchmarks)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

The benchmark tests serve as regression guards. Each prints a measurement and checks it against a loose bound that every supported CPU meets, so that gross regressions (for example an accidentally quadratic tiling loop or a collapse of batched throughput) fail the test suite, while normal variation between machines does not.

### 1.2 Status of the models

The model zoo has 520 models (130 families in four variants). Apart from the 28 models of the 7 spectral index families, which compute exact formulas, they are untrained starter models with deterministic weights. The benchmarks run starter models on random input. Their outputs are not meaningful, and nothing in this document is, or may be read as, a statement about the accuracy of any model. Earlier versions of this document contained accuracy, latency and scaling figures that were not produced by the project's code; they have been removed.

### 1.3 Conventions

The key words MUST, SHOULD and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in all capitals. They are used in [Section 6](#6-adding-or-reporting-benchmarks).

## 2. The benchmark tests

### 2.1 Location and selection

All benchmarks are in one module, [tests/benchmarks/test_performance.py](../../tests/benchmarks/test_performance.py). They carry no pytest marker; the markers registered in [pyproject.toml](../../pyproject.toml) are `slow`, `gpu` and `integration`, and none of them is applied to the benchmarks. The benchmarks are selected by their directory. The module calls `pytest.importorskip("torch")`, so all its tests are skipped when PyTorch (the `torch` extra) is not installed. Models are built in memory with `unbihexium.ai.models.build_model`; nothing is written to the model store and no network access is needed.

### 2.2 What each test measures

| Test | Model | Input | Measurement | Assertion |
| --- | --- | --- | --- | --- |
| `TestInferenceBenchmarks::test_detection_throughput` | `ship_detector_tiny` (3 bands, 730,581 parameters) | Random batch of 4 images, 128 x 128 pixels | Images per second of the forward pass | More than 1 image/s |
| `TestInferenceBenchmarks::test_segmentation_throughput` | `lulc_classifier_tiny` (10 bands, 734,107 parameters) | Random batch of 4 images, 128 x 128 pixels | Images per second of the forward pass | More than 1 image/s |
| `TestInferenceBenchmarks::test_batch_size_scaling` | `water_surface_detector_tiny` (4 bands, 733,090 parameters) | Batches of 1 and of 8 images, 128 x 128 pixels | Images per second for both batch sizes | Throughput at batch 8 is more than half of the throughput at batch 1 |
| `TestMemoryBenchmarks::test_model_memory_footprint` | `lulc_classifier_base` (7,060,779 parameters) | none | Bytes of all parameters and buffers | At least 4 bytes per parameter (float32) |
| `TestMemoryBenchmarks::test_inference_memory_usage` | `tree_height_estimator_tiny` (12 bands) through the tiled `Predictor` with 256-pixel tiles | Random 12-band image of 768 x 768 pixels (27.0 MiB as float32) | Peak memory traced by `tracemalloc` during `Predictor.dense` | Output shape (1, 768, 768) and peak below 10 times the input size |

### 2.3 Method

- **Throughput.** The helper `throughput(model_id, batch, size=128, repeats=3)` builds the model, creates one random batch with `torch.rand`, runs one untimed warm-up forward pass under `torch.inference_mode()`, then times three forward passes with `time.perf_counter()` and returns $\text{batch} \times 3 / t$. Only the network forward pass is timed: no file input and output, no normalisation, no tiling and no decoding of detections.
- **Model memory.** The sum of `numel() * element_size()` over parameters and buffers, printed in MiB.
- **Tiled inference memory.** `tracemalloc` [3] records the peak of memory allocations made through Python's allocators, which include NumPy arrays, while `Predictor.dense` tiles the image, runs the network and blends the tiles. Memory allocated by PyTorch's own C++ allocator for tensors is not traced, so the figure describes the NumPy side of tiled inference (input copy, padding, tile batches, accumulators and output), not the resident memory of the process.

### 2.4 Where the benchmarks run

The test job of [.github/workflows/ci.yml](../../.github/workflows/ci.yml) installs the hashed CPU build of PyTorch and runs `pytest tests/` on CPython 3.10, 3.11, 3.12, 3.13 and 3.14, so the benchmarks run, with their loose bounds, on every push and pull request that triggers CI. `make test` and `make test-cov` also include them; `make test-fast` runs only `tests/unit` and excludes them. CI logs do not show the printed figures because the workflow does not pass `-s`.

## 3. Running the benchmarks

Install the package with the `torch` and `test` extras (or the development lock file, see [CONTRIBUTING.md](../../CONTRIBUTING.md)) and run, from the repository root:

```bash
python -m pytest tests/benchmarks -s
```

`-s` disables output capturing so that the measurements are printed; without it the tests only pass or fail. A single test is selected with its node identifier, for example:

```bash
python -m pytest "tests/benchmarks/test_performance.py::TestInferenceBenchmarks::test_detection_throughput" -s
```

PyTorch uses as many intra-op threads as it detects cores by default. To compare runs, fix the thread count, for example with `OMP_NUM_THREADS=1`, and close other workloads; report the setting with the results.

## 4. Measured results

### 4.1 Environment

| Item | Value |
| --- | --- |
| Date | 2026-09-24 |
| Code | Branch `dev`, commit `8493397` |
| Machine | Shared cloud container, 4 vCPUs reported by `nproc`, CPU model string `Intel(R) Xeon(R) Processor @ 2.80GHz`, 16 GiB RAM, no GPU |
| Operating system | Ubuntu 24.04.4 LTS, Linux 6.18.44, x86_64 |
| Python | CPython 3.13.12 |
| Libraries | NumPy 2.5.3, PyTorch 2.14.0+cu130 (CUDA build; `torch.cuda.is_available()` is false, so all computation ran on the CPU), pytest 9.1.1 |
| PyTorch threads | 4 (`torch.get_num_threads()`, default, no thread variables set) |
| Command | `python -m pytest tests/benchmarks -s -q -p no:cacheprovider`, run six times in succession |

### 4.2 Results

All five tests passed in every run; one run reported `5 passed in 11.01s`. The printed figures of the six runs were:

| Measurement | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ship_detector_tiny`, batch 4, images/s | 96.4 | 100.6 | 96.6 | 4.6 | 111.6 | 89.1 | 96.5 |
| `lulc_classifier_tiny`, batch 4, images/s | 49.3 | 52.0 | 54.0 | 54.1 | 56.6 | 62.9 | 54.05 |
| `water_surface_detector_tiny`, batch 1, images/s | 38.1 | 50.9 | 46.6 | 53.6 | 23.2 | 56.8 | 48.75 |
| `water_surface_detector_tiny`, batch 8, images/s | 22.2 | 78.1 | 69.4 | 45.6 | 69.7 | 67.6 | 68.5 |
| `lulc_classifier_base`, weights, MiB | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 | 26.9 |
| Tiled inference peak (`tracemalloc`), MiB | 87.9 | 87.9 | 87.9 | 87.9 | 87.9 | 87.9 | 87.9 |

All images are 128 x 128 pixels. The test rounds the printed values to one decimal; medians are computed from the printed values.

### 4.3 Single-thread run

One further run in the same environment with one intra-op thread, `OMP_NUM_THREADS=1 python -m pytest tests/benchmarks -s -q -p no:cacheprovider`, printed 42.9 images/s for `ship_detector_tiny`, 26.7 images/s for `lulc_classifier_tiny`, 24.9 and 26.6 images/s for `water_surface_detector_tiny` at batch 1 and 8, and the same memory figures (26.9 MiB and 87.9 MiB). All five tests passed.

## 5. Interpretation and limits

- **Variance is large on shared machines.** The throughput of the same test varied by more than a factor of two between consecutive runs, and one run of the detection test measured 4.6 images/s against a median of 96.5, most likely because of competing load on the shared host during the three timed passes. Three repetitions after one warm-up pass are enough for a regression bound, not for a precise measurement.
- **Batching is not guaranteed to help on a CPU.** In five of six runs, batch 8 gave more images per second than batch 1; in run 1 it gave fewer. The test only requires that batching does not reduce throughput by more than half.
- **The memory figures are deterministic.** The weight size follows from the parameter count: $7{,}060{,}779 \times 4$ bytes $= 26.9$ MiB. The traced peak of tiled inference (87.9 MiB for a 27.0 MiB input) is well below the bound of 10 times the input size (270 MiB).
- **What the figures do not cover.** They do not include reading or writing files, normalisation, tile blending or detection decoding in the throughput tests, the ONNX Runtime backend, GPUs, the REST service, variants other than those listed, or any accuracy. They are not comparisons with other software.

## 6. Adding or reporting benchmarks

1. A benchmark test MUST assert a bound that every supported CPU meets, so that it cannot fail because of an ordinary slow runner; it SHOULD print its measurement so that `-s` shows it.
2. Benchmarks that need PyTorch MUST skip cleanly without it, as the existing module does with `pytest.importorskip`.
3. A benchmark SHOULD use tiny variants and small inputs, because it runs in CI on five Python versions.
4. Figures published in this document MUST come from running the tests in this repository, and MUST be accompanied by the environment and the command, as in [Section 4](#4-measured-results). Figures that were not measured MUST NOT be added, and accuracy figures of starter models MUST NOT be reported.
5. Contributors MAY add results from other machines as additional subsections of Section 4, each with its own environment table.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] Python Software Foundation. tracemalloc: Trace memory allocations. 2026. <https://docs.python.org/3/library/tracemalloc.html>

<!--
=============================================================================
End of file docs/benchmarks/BENCHMARKS.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
