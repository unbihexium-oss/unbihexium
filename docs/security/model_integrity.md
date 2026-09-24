<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : docs/security/model_integrity.md
Title       : Model Integrity
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Model Integrity

| Field | Value |
| --- | --- |
| Document | UBX-DOC-SEC-MODEL-INTEGRITY |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-24 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](../../MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.1 and the main branch, model catalogue 2.0.0 |

## Abstract

This document describes how Unbihexium protects the integrity of the models of its model zoo and of checkpoints and ONNX files handled by the library: how starter weights are derived and compared with published digests, how checkpoints are loaded without executing pickled code, what the ONNX export checks, and what the local checksum files do and do not prove. It is written for users who load models, operators who run the REST service, and auditors who need to know exactly which threats are covered. It describes the behaviour of the code in [src/unbihexium/zoo/](../../src/unbihexium/zoo/) and states the known gaps explicitly. Reporting a weakness in these mechanisms is handled as described in [SECURITY.md](../../SECURITY.md).

## Contents

- [1. Introduction](#1-introduction)
- [2. Starter Weights and Published Digests](#2-starter-weights-and-published-digests)
- [3. The Local Model Store](#3-the-local-model-store)
- [4. Checkpoint Loading](#4-checkpoint-loading)
- [5. ONNX Export and Loading](#5-onnx-export-and-loading)
- [6. Verification Procedures](#6-verification-procedures)
- [7. What Is and Is Not Protected](#7-what-is-and-is-not-protected)
- [8. Recommendations](#8-recommendations)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

A model file is code-adjacent data: a malicious checkpoint can execute code when it is deserialised carelessly, and a silently modified model produces wrong results without any visible error. This document explains the controls that Unbihexium applies to model files and where their limits are.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when, and only when, they appear in all capitals.

### 1.3 The Models Are Untrained

The model zoo contains 520 models (130 families in 4 variants: tiny, base, large and mega). They are untrained starter models with deterministic weights, except the 28 models of the 7 spectral index families, which compute exact formulas and have no learnable parameters. Integrity verification proves that a model is the one the project published; it says nothing about the quality of its predictions, which is not meaningful for a starter model until it is trained. See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md), Section 2.

### 1.4 Terminology

| Term | Meaning |
| --- | --- |
| Weights digest | SHA-256 over the model's state dictionary as defined in Section 2.2. It identifies the numerical weights independently of the file format. |
| File checksum | SHA-256 of the bytes of a file, as written by `sha256sum`. |
| Published digest | The weights digest of a catalogue model recorded in [src/unbihexium/zoo/digests.json](../../src/unbihexium/zoo/digests.json), shipped inside the package, and mirrored in [model_zoo/checksums.txt](../../model_zoo/checksums.txt). |
| Checkpoint | A file `model.pt` written by `unbihexium.zoo.checkpoint.save_checkpoint`. |

## 2. Starter Weights and Published Digests

### 2.1 Deterministic Generation

The starter weights of every catalogue model are generated locally from the model identifier; there is no download. The generator is seeded with the first four bytes of `sha256("unbihexium:" + model_id)` and uses `numpy.random.RandomState`, as recorded in the `initialisation` field of `digests.json`. Every user therefore obtains bit-identical starter weights on every platform where the generation is reproducible, and the [.github/workflows/model-zoo.yml](../../.github/workflows/model-zoo.yml) workflow checks this: pull requests rebuild the tiny variants and compare their digests with the published ones, and a weekly run rebuilds all 520 models.

### 2.2 The Weights Digest

The weights digest is computed by `unbihexium.ai.models.init.weights_digest`. For a state dictionary with keys $k_1 < k_2 < \dots < k_n$ in lexicographic order, each tensor $W_i$ is converted to 32-bit little-endian floating point and the digest is

$$
D = \mathrm{SHA256}\left( h_1 \Vert b_1 \Vert h_2 \Vert b_2 \Vert \cdots \Vert h_n \Vert b_n \right),
$$

where $h_i$ is the UTF-8 encoding of the line `<key>:<shape>` followed by a newline (for `ship_detector_tiny` the first line is `network.decoder.stages.0.block.conv1.conv.weight:(64, 64, 3, 3)`), $b_i$ is the raw float32 little-endian byte representation of $W_i$, $\Vert$ denotes concatenation and SHA-256 is the hash function of FIPS 180-4 [3]. Because the digest covers names, shapes and values but not the serialisation format, the same weights give the same digest whether they are held in memory, in a checkpoint or in the metadata of an ONNX export. The spectral index models have no parameters, so their digest is the SHA-256 of the empty string, `e3b0c442...b855`.

### 2.3 Where the Digests Are Published

- [src/unbihexium/zoo/digests.json](../../src/unbihexium/zoo/digests.json) holds the weights digest and parameter count of all 520 models, the catalogue version (2.0.0) and a description of the algorithm. It is part of the installed package and is the reference used at run time.
- [model_zoo/checksums.txt](../../model_zoo/checksums.txt) lists the same digests in `sha256sum` layout (digest, model identifier) for human review.
- The manifests under [model_zoo/manifests/](../../model_zoo/manifests/) and the output of `unbihexium zoo info <model_id>` show the digest of each model.

The digests travel with the source code and the released distributions, so their authenticity rests on the integrity of the repository and of the release (see [supply_chain_security.md](supply_chain_security.md)).

### 2.4 Verification When a Model Is Built

`unbihexium.zoo.load_model(model_id)` builds a catalogue model in memory and, unless it is called with `verify=False`, compares its weights digest with the published digest. A mismatch raises `unbihexium.zoo.VerificationError`. The check is skipped only for models whose configuration was customised (for example a different band count), because their weights legitimately differ. `unbihexium zoo build` and `unbihexium.zoo.ensure_model` go through the same function when they create a checkpoint in the store.

## 3. The Local Model Store

### 3.1 Layout

Models written to disk are kept under `$UNBIHEXIUM_CACHE/models/<model_id>/`; `UNBIHEXIUM_CACHE` defaults to `~/.cache/unbihexium`. Each directory contains:

| File | Content |
| --- | --- |
| `model.pt` | Checkpoint with the configuration, the state dictionary and the weights digest recorded at save time |
| `model.onnx` | ONNX export, only when requested (`--onnx` or `onnx=True`) |
| `config.json` | Catalogue entry of the model and the SHA-256 checksum of `model.pt` |
| `model.sha256` | `sha256sum`-compatible checksums of the three files above |

### 3.2 The Role of `model.sha256`

`model.sha256` is rewritten from the current files every time `ensure_model` (and therefore `unbihexium zoo build`) runs on the directory, even when the model was already cached. It is a record of the state of the directory at that moment, not a reference value from an independent source: it detects accidental corruption between two points in time if it is checked with `sha256sum --check` or `unbihexium.zoo.verify_directory`, but it cannot detect a modification made before it was last written, and it is not consulted by `verify_model` or `unbihexium zoo verify` (Section 7.2).

### 3.3 User-Registered Models

Trained models can be added to the registry with `unbihexium.zoo.register_model`, using a `ModelZooEntry` whose `source` is `local` (with `local_path`) or `url` (with `download_url`). Downloads use `requests` with HTTPS certificate verification by default, a 60 second timeout and a hard limit of 4 GiB (`MAX_DOWNLOAD_BYTES`); a larger response raises `VerificationError`. The file is written to a temporary name and renamed only when the download is complete.

## 4. Checkpoint Loading

### 4.1 No Arbitrary Unpickling

Checkpoints are PyTorch files, which use the Python pickle format internally. `unbihexium.zoo.checkpoint.read_checkpoint` loads them with `torch.load(path, map_location="cpu", weights_only=True)`. In this mode PyTorch's unpickler accepts only tensors, primitive types and containers, and refuses to construct arbitrary Python objects, so a crafted checkpoint cannot run code through `__reduce__` [4]. A test with a checkpoint that embeds such an object ends with `_pickle.UnpicklingError` before any object is created.

### 4.2 Format and Digest Checks

After deserialisation the loader requires the marker `format == "unbihexium-checkpoint"` and a `format_version` not newer than the installed library supports (currently 1), and raises `CheckpointError` otherwise. `load_checkpoint` then rebuilds the network from the stored configuration, loads the state dictionary with `strict=True` (unexpected or missing tensors are errors), and, unless `verify=False`, recomputes the weights digest and compares it with the digest stored in the file. This detects accidental corruption and naive edits of the tensors. Because the stored digest is inside the same file, it does not detect a deliberate modification whose author also rewrote the digest; that requires comparison with the published digest (Section 6).

### 4.3 Other Model Formats

`unbihexium.core.model.ModelWrapper` can also open TorchScript archives (`torch.jit.load`) and programs saved with `torch.export` (`.pt2`). These formats carry executable graphs and are not restricted by `weights_only`; they MUST be opened only from trusted sources. Estimators of other frameworks (for example scikit-learn) are never loaded from files by the library, because unpickling them can execute code; they must be passed in as fitted objects.

## 5. ONNX Export and Loading

### 5.1 Export Checks

`unbihexium.zoo.export.export_onnx` (used by `unbihexium zoo export` and by `ensure_model(..., onnx=True)`) exports with the TorchScript-based exporter at ONNX opset 18 [5] and then:

1. writes two metadata properties into the ONNX model: `unbihexium_config` (the JSON build configuration) and `unbihexium_weights_digest` (the weights digest of the exported PyTorch model);
2. unless verification is disabled (`--no-verify`), runs the exported file with ONNX Runtime on the CPU and the PyTorch model on the same seeded random batch of two images, and raises `ExportError` when the output shapes differ or when the largest absolute difference exceeds $10^{-3} \cdot \max(1, \max|y_{\mathrm{torch}}|)$.

The comparison guards against export defects (unsupported operators, wrong dynamic axes); it is a functional check, not a security control.

### 5.2 Loading ONNX Files

The ONNX backend (`unbihexium.ai.inference.OnnxBackend`) opens a file with ONNX Runtime and refuses files that lack the `unbihexium_config` metadata. This prevents accidental use of foreign models with the wrong band layout; it does not authenticate the file, because anyone can write that metadata. ONNX Runtime does not execute Python code from a model file, but a malicious model can still produce misleading outputs or consume excessive memory and time. The library does not compare the ONNX file with any digest when it is loaded.

## 6. Verification Procedures

### 6.1 Command Line

Build a model into the store and verify it:

```bash
unbihexium zoo build ship_detector_tiny --onnx
unbihexium zoo verify ship_detector_tiny
```

`zoo build` prints `Cached:` followed by the model directory. `zoo verify` prints `Verified: ship_detector_tiny` and exits with status 0 when the cached checkpoint loads, its internal digest matches its weights and its weights match the published digest; otherwise it prints `Error: ship_detector_tiny is not cached or does not verify` and exits with status 1. `unbihexium zoo export <model_id or checkpoint> <file.onnx>` exports and checks an ONNX file as described in Section 5.1.

To check the file checksums of a cached model independently, run `sha256sum --check model.sha256` in its directory (the directory is printed by `zoo build`).

### 6.2 Python

The following example uses a temporary store; it needs the `torch` and `onnx` extras.

```python
import os
import tempfile

os.environ["UNBIHEXIUM_CACHE"] = tempfile.mkdtemp()  # Throw-away model store.

from unbihexium.zoo import ensure_model, get_model, load_model, verify_directory, verify_model

entry = get_model("ship_detector_tiny")
print(entry.weights_digest[:16], entry.requires_training)

model = load_model("ship_detector_tiny")  # Builds the model and checks digests.json.
print(model.digest() == entry.weights_digest)

directory = ensure_model("ship_detector_tiny", onnx=True)  # model.pt, model.onnx, config.json.
print(verify_model("ship_detector_tiny"))  # Checkpoint digest against digests.json.
print(verify_directory(directory))  # File checksums listed in model.sha256.

import onnx

props = {p.key: p.value for p in onnx.load(str(directory / "model.onnx")).metadata_props}
print(sorted(props))
print(props["unbihexium_weights_digest"] == entry.weights_digest)
```

Output:

```text
e278457bcbb75b89 True
True
True
{'config.json': True, 'model.onnx': True, 'model.pt': True}
['unbihexium_config', 'unbihexium_weights_digest']
True
```

The last check compares the digest recorded in the ONNX metadata with the published one. It shows which weights were exported, but since the metadata is part of the file it does not prove that the graph in the file was not altered afterwards.

### 6.3 Checkpoints of Trained Models

For models that users train themselves there is no published digest. The digest returned by `save_checkpoint` (and stored in the checkpoint) SHOULD be recorded outside the file, for example in the model card or next to the results, and compared with `load_checkpoint(path).digest()` before the model is used for results that matter.

## 7. What Is and Is Not Protected

### 7.1 Protected

| Threat | Control |
| --- | --- |
| Code execution through a crafted `model.pt` | `torch.load(weights_only=True)` in `read_checkpoint` |
| Corrupted or naively edited checkpoint tensors | Internal digest check in `load_checkpoint` |
| Catalogue checkpoint replaced by other weights, including a rewritten internal digest | `verify_model` and `unbihexium zoo verify` compare with `digests.json` |
| Starter weights that differ between platforms or after a code change | Digest check in `load_model`; rebuild in the Model Zoo workflow |
| Unbounded downloads of registered models | 4 GiB limit, HTTPS verification, atomic rename |
| Silent export defects | ONNX Runtime comparison in `export_onnx` |

### 7.2 Not Protected

The following gaps exist in the current code. They are listed so that operators can compensate for them (Section 8).

- **Authenticity of registered models.** A `ModelZooEntry` has no field for an expected file checksum, and `load_model` does not compare a model loaded from a `local` or `url` source with the entry's `weights_digest`; only `verify_model` does. Whoever controls the URL or the file controls the model.
- **ONNX files in the store.** `model.onnx` and `config.json` are not checked by `verify_model` or `unbihexium zoo verify`, and the ONNX backend uses a cached `model.onnx` without verifying it. A modified ONNX file in the store is therefore used silently.
- **`model.sha256` as a reference.** The checksum file is regenerated from the current files by every `ensure_model` call (Section 3.2) and is written by the same process that could be compromised.
- **Signatures.** Models and checkpoints are not signed. Trust in the published digests derives from the integrity of the repository and of the release artefacts.
- **Model quality and poisoning.** Verification cannot tell whether trained weights were learned from poisoned or biased data, and it does not make starter models meaningful.
- **Executable formats.** TorchScript and `torch.export` archives opened with `ModelWrapper` are not sandboxed.

## 8. Recommendations

- Users SHOULD run `unbihexium zoo verify <model_id>` after building or copying a model store, and SHOULD rebuild with `unbihexium zoo build <model_id> --force` when verification fails.
- Operators of the REST service with the ONNX backend SHOULD keep a copy of `model.sha256` outside the store after the models have been built and checked, and check the files against that copy with `sha256sum --check` before start-up. The store cannot currently be mounted read-only, because `ensure_model`, which the ONNX backend calls, rewrites `config.json` and `model.sha256` on every call; access to it SHOULD therefore be limited to the service account.
- Users who register trained models SHOULD distribute the checkpoint digest through a channel independent of the download location, and SHOULD call `verify_model` (not only `load_model`) after registering.
- Users MUST NOT open TorchScript, `torch.export` or ONNX files from untrusted sources, and SHOULD apply the resource limits of the REST service described in [SECURITY.md](../../SECURITY.md), Section 8.
- Suspected weaknesses in these controls MUST be reported privately as described in [SECURITY.md](../../SECURITY.md).

Further reading: [supply_chain_security.md](supply_chain_security.md) for the integrity of the package itself, and [docs/model_zoo/download_and_verify.md](../model_zoo/download_and_verify.md) for day-to-day use of the model store.

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] National Institute of Standards and Technology. FIPS PUB 180-4: Secure Hash Standard (SHS). 2015. <https://doi.org/10.6028/NIST.FIPS.180-4>

[4] PyTorch contributors. torch.load, PyTorch documentation. 2026. <https://docs.pytorch.org/docs/stable/generated/torch.load.html>

[5] ONNX contributors. ONNX Operator Schemas and Versioning. 2026. <https://onnx.ai/onnx/repo-docs/Versioning.html>

<!--
=============================================================================
End of file docs/security/model_integrity.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
