<!--
Complete every question. For closed questions tick the options that apply. If none fits,
tick "Other:" and write your answer on the same line after the colon. Write N/A in an
answer only when the question genuinely does not apply. Every confirmation must be ticked
before review.
-->

## Summary of the change

Explain what the pull request changes and why.

### Q1. Summary

Describe the change in a few sentences, written for a reviewer who has not seen the related issue.

Answer:

### Q2. Motivation and context

Explain why the change is needed and which problem it solves.

Answer:

### Q3. Issues closed by this pull request

List the issues this pull request resolves, using `Closes #number` so they close on merge. Write `N/A` if none.

Answer:

### Q4. Type of change

Select every type that applies.

_Select all that apply._

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (existing functionality changes behaviour)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Test addition or modification
- [ ] Performance improvement
- [ ] Build, packaging or CI
- [ ] Dependency update
- [ ] Model zoo change
- [ ] Security fix
- [ ] Compliance or licensing change
- [ ] Other:

### Q5. Modules changed

Select every module the pull request changes.

_Select all that apply._

- [ ] unbihexium.ai (detection, segmentation, classification)
- [ ] unbihexium.analysis (spatial and network analysis)
- [ ] unbihexium.cli (command line interface)
- [ ] unbihexium.config (configuration handling)
- [ ] unbihexium.core (rasters, vectors, pipelines, tiling)
- [ ] unbihexium.geostat (kriging, variograms, spatial statistics)
- [ ] unbihexium.indices (NDVI, NDWI, EVI, NBR, MSI and others)
- [ ] unbihexium.io (raster, vector and STAC input and output)
- [ ] unbihexium.metrics (accuracy and quality metrics)
- [ ] unbihexium.postprocessing
- [ ] unbihexium.preprocessing (radiometric and geometric preparation)
- [ ] unbihexium.registry (capability and model registry)
- [ ] unbihexium.sar (SAR amplitude, phase and interferometry)
- [ ] unbihexium.serving (FastAPI REST service)
- [ ] unbihexium.terrain (DEM, DSM, DTM, slope, aspect)
- [ ] unbihexium.utils
- [ ] unbihexium.visualization
- [ ] unbihexium.zoo (model download, verification and loading)
- [ ] Model zoo assets (model_zoo/)
- [ ] Documentation or example notebooks
- [ ] Packaging, Docker image or CI configuration
- [ ] Not sure
- [ ] Other:

### Q6. Capability domains affected

Select every capability domain affected.

_Select all that apply._

- [ ] 01 AI products (detection, segmentation, super-resolution)
- [ ] 02 Tourism and data processing
- [ ] 03 Indices, flood and water
- [ ] 04 Environment, forestry and image processing
- [ ] 05 Asset management and energy
- [ ] 06 Urban planning and agriculture
- [ ] 07 Risk and defense (neutral analytics)
- [ ] 08 Value-added imagery (DEM, orthorectification, mosaicking)
- [ ] 09 Benefits narrative and reporting
- [ ] 10 Satellite imagery features
- [ ] 11 Resolution, metadata and quality assurance
- [ ] 12 Radar and SAR
- [ ] Cross-cutting or not domain specific
- [ ] Other:

### Q7. Design and implementation notes

Describe the approach, key decisions and trade-offs.

Answer:

### Q8. Alternatives considered

Describe alternatives you rejected and why. Write `N/A` if none.

Answer:

## Compatibility

Describe the effect on users.

### Q9. Backward compatibility

Declare the compatibility impact under the [versioning policy](https://github.com/unbihexium-oss/unbihexium/blob/main/VERSIONING.md).

_Select one._

- [ ] Fully backward compatible
- [ ] Deprecates existing behaviour with warnings
- [ ] Breaking change
- [ ] Other:

### Q10. Migration notes

Describe what users must change. Write `N/A` if the change is backward compatible.

Answer:

### Q11. Public API changes

Select every kind of public API change.

_Select all that apply._

- [ ] No public API change
- [ ] API added
- [ ] API changed
- [ ] API removed
- [ ] CLI changed
- [ ] REST API changed
- [ ] Other:

### Q12. Version increment

Select the semantic version increment the change requires.

_Select one._

- [ ] Patch
- [ ] Minor
- [ ] Major
- [ ] Other:

## Testing

Show that the change works.

### Q13. Tests

Select every kind of test added or changed.

_Select all that apply._

- [ ] Unit tests added
- [ ] Integration tests added
- [ ] End-to-end tests added
- [ ] Existing tests updated
- [ ] No tests needed, explained in the testing notes
- [ ] Other:

### Q14. Test commands and results

Paste the commands you ran and a summary of their output.

```shell

```

### Q15. Python versions tested locally

Select every Python version you tested on. CI tests all supported versions.

_Select all that apply._

- [ ] 3.14
- [ ] 3.13
- [ ] 3.12
- [ ] 3.11
- [ ] 3.10
- [ ] Other:

### Q16. Local checks run

Select every check you ran locally.

_Select all that apply._

- [ ] ruff check
- [ ] ruff format --check
- [ ] pyright
- [ ] pytest
- [ ] pre-commit run --all-files
- [ ] Other:

### Q17. Test coverage

Select how coverage changed.

_Select one._

- [ ] Coverage increased
- [ ] Coverage unchanged
- [ ] Coverage decreased
- [ ] Not measured
- [ ] Other:

### Q18. Manual testing

Describe manual testing, including data and commands. Write `N/A` if none.

Answer:

## Impact

Describe side effects of the change.

### Q19. Performance impact

Declare the performance impact.

_Select one._

- [ ] No expected impact
- [ ] Improved, measured
- [ ] Regressed, measured and justified
- [ ] Not measured
- [ ] Other:

### Q20. Benchmark results

Give before and after measurements. Write `N/A` if not measured.

Answer:

### Q21. Security impact

Declare the security impact. Undisclosed vulnerabilities must go through a [private advisory](https://github.com/unbihexium-oss/unbihexium/security/advisories/new).

_Select one._

- [ ] No security impact
- [ ] Improves security
- [ ] Changes a security-relevant component and needs security review
- [ ] Other:

### Q22. Documentation updated

Select every documentation artefact updated.

_Select all that apply._

- [ ] README
- [ ] Documentation in docs/
- [ ] Docstrings
- [ ] Example notebooks
- [ ] Model cards
- [ ] Not needed
- [ ] Other:

### Q23. Changelog

Declare whether [CHANGELOG.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CHANGELOG.md) was updated under `[Unreleased]`.

_Select one._

- [ ] Updated
- [ ] Not needed (internal change)
- [ ] Other:

### Q24. Licence headers

Declare whether every new source file carries the MPL-2.0 Exhibit A notice.

_Select one._

- [ ] All new source files carry the MPL-2.0 notice
- [ ] No new source files
- [ ] Other:

### Q25. Dependency changes

Select every dependency change.

_Select all that apply._

- [ ] No dependency changes
- [ ] Dependencies added
- [ ] Dependencies updated
- [ ] Dependencies removed
- [ ] Other:

### Q26. Changed dependencies

List each changed dependency with version and SPDX licence identifier. Write `N/A` if none.

Answer:

### Q27. Model zoo changes

Select every model zoo change.

_Select all that apply._

- [ ] No model changes
- [ ] Model weights added
- [ ] Model weights updated
- [ ] Checksums updated
- [ ] Model cards updated
- [ ] Manifests or inventory updated
- [ ] Other:

### Q28. Data files added

Declare data files added to the repository.

_Select one._

- [ ] No data files
- [ ] Synthetic test fixtures
- [ ] Samples of openly licensed data with attribution
- [ ] Other:

### Q29. CI and container changes

Select every change to CI or containers.

_Select all that apply._

- [ ] No CI or container changes
- [ ] GitHub workflows changed
- [ ] Dockerfile or compose files changed
- [ ] Release or signing process changed
- [ ] Other:

### Q30. Commit messages

Declare whether commits follow the `type(scope): description` format from [CONTRIBUTING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CONTRIBUTING.md).

_Select one._

- [ ] All commits follow the format
- [ ] Commits will be squashed with a conforming message
- [ ] Other:

### Q31. Notes for reviewers

Point reviewers to the parts that need the most attention. Write `N/A` if none.

Answer:

### Q32. Contribution terms

Confirm each statement.

_Tick every statement._

- [ ] I have the right to submit this contribution as set out in the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
- [ ] I agree that my contribution is licensed under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/), as stated in [CONTRIBUTING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CONTRIBUTING.md).
- [ ] The change is written in English and contains no emojis and no em dashes.

## Submitter and context

These questions help maintainers understand who is affected and how to prioritise the pull request. The answers are used for triage only.

### Q33. Your role

Select the role that best describes how you interact with Unbihexium in the context of this pull request.

_Select one._

- [ ] End user of the library or CLI
- [ ] Application developer integrating the library
- [ ] Researcher or academic
- [ ] Data scientist or machine learning engineer
- [ ] GIS or remote sensing analyst
- [ ] Maintainer or regular contributor
- [ ] Representative of an organisation (company, public body or NGO)
- [ ] Student
- [ ] Other:

### Q34. Organisation type

Select the type of organisation on whose behalf you are submitting. Choose the individual option if you act in a private capacity.

_Select one._

- [ ] Individual, no organisation
- [ ] Academic or research institution
- [ ] Public sector body in the EU or EEA
- [ ] Public sector body outside the EU or EEA
- [ ] Private company, small or medium-sized enterprise
- [ ] Private company, large enterprise
- [ ] Non-profit organisation or NGO
- [ ] Other:

### Q35. Primary jurisdiction of use

Select the jurisdiction in which you mainly use or distribute Unbihexium. This determines which of the regulatory questions below apply to you.

_Select one._

- [ ] EU or EEA member state
- [ ] United Kingdom
- [ ] Switzerland
- [ ] United States
- [ ] Canada
- [ ] Other European country
- [ ] Asia-Pacific
- [ ] Middle East and North Africa
- [ ] Sub-Saharan Africa
- [ ] Latin America and the Caribbean
- [ ] Other:

### Q36. Search for existing reports

Confirm that you searched open and closed issues, pull requests and the documentation before creating this pull request.

_Select one._

- [ ] I searched and found no existing issue or pull request
- [ ] I found related issues or pull requests and link them in the next question
- [ ] I found a closed issue or pull request, but the subject is still open
- [ ] Other:

### Q37. Related issues, pull requests and discussions

List related issues, pull requests, discussions or external tickets as links or `#number` references, separated by commas. Write `N/A` if there are none.

Answer:

### Q38. Impact on your work

Select how strongly the subject of this pull request affects your work today.

_Select one._

- [ ] Blocking: I cannot continue without a resolution
- [ ] Major: significant workaround or delay
- [ ] Moderate: inconvenient but manageable
- [ ] Minor: small inconvenience
- [ ] Cosmetic or informational only
- [ ] Other:

## Environment

Describe the exact environment. Precise versions allow maintainers to reproduce the situation without additional round trips.

### Q39. Unbihexium version

Select the Unbihexium version you are using. Run `python -c "import unbihexium; print(unbihexium.__version__)"` if you are unsure.

_Select one._

- [ ] 1.0.1 (latest release)
- [ ] 1.0.0
- [ ] main branch (unreleased)
- [ ] 0.9.x
- [ ] 0.8.x or earlier
- [ ] Other:

### Q40. Exact version string or commit SHA

Paste the exact version string, and for source installations also the output of `git rev-parse HEAD`.

Answer:

### Q41. Installation method

Select how Unbihexium was installed in the environment described here.

_Select one._

- [ ] PyPI with pip
- [ ] PyPI with uv, pipx, poetry or pdm
- [ ] conda or mamba (conda-forge)
- [ ] Source checkout, editable install
- [ ] Official Docker image
- [ ] Custom container image
- [ ] Other:

### Q42. Python version

Select the Python version. Unbihexium supports CPython 3.10 to 3.14 as documented in [VERSIONING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/VERSIONING.md).

_Select one._

- [ ] 3.14
- [ ] 3.13
- [ ] 3.12
- [ ] 3.11
- [ ] 3.10
- [ ] Other:

### Q43. Python implementation

Select the Python implementation and build.

_Select one._

- [ ] CPython, standard build
- [ ] CPython, free-threaded build
- [ ] PyPy
- [ ] Other:

### Q44. Operating system family

Select the operating system family on which you observed the behaviour or tested the change.

_Select one._

- [ ] Linux
- [ ] Windows
- [ ] macOS
- [ ] Windows Subsystem for Linux (WSL2)
- [ ] Other:

### Q45. Operating system distribution and version

Give the distribution and version, for example `Ubuntu 24.04`, `Windows 11 24H2` or `macOS 15.5`.

Answer:

### Q46. CPU architecture

Select the processor architecture of the machine.

_Select one._

- [ ] x86_64 or AMD64
- [ ] arm64 or aarch64 (including Apple silicon)
- [ ] Other:

### Q47. Hardware accelerator

Select the accelerator used for inference or processing.

_Select one._

- [ ] No accelerator, CPU only
- [ ] NVIDIA GPU (CUDA)
- [ ] AMD GPU (ROCm)
- [ ] Apple GPU (Metal or MPS)
- [ ] Intel GPU or NPU
- [ ] Other:

### Q48. Accelerator details

Give the accelerator model, driver version, CUDA or ROCm version and available memory. Write `N/A` if you use the CPU only.

Answer:

### Q49. Inference backend

Select the backend used to execute models, if any.

_Select one._

- [ ] ONNX Runtime, CPU execution provider
- [ ] ONNX Runtime, CUDA or TensorRT execution provider
- [ ] PyTorch
- [ ] Both ONNX Runtime and PyTorch
- [ ] No model inference involved
- [ ] Other:

### Q50. Deployment context

Select where the code runs.

_Select one._

- [ ] Local workstation or laptop
- [ ] On-premises server
- [ ] Public cloud virtual machine
- [ ] Kubernetes or container orchestration
- [ ] HPC cluster or batch scheduler
- [ ] Edge device
- [ ] CI pipeline
- [ ] Notebook service (JupyterHub, Colab-like environment)
- [ ] Other:

### Q51. Relevant package versions

Paste the output of `pip freeze` or `conda list`, or at least the versions of numpy, rasterio, GDAL, shapely, pyproj, onnxruntime and torch.

```text

```

## Declarations: repository, libraries, models and data

Declare the provenance of everything involved: the Unbihexium source, third-party libraries, models and datasets. These declarations are required for licence compliance under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) and the notices in [THIRD_PARTY_NOTICES.md](https://github.com/unbihexium-oss/unbihexium/blob/main/THIRD_PARTY_NOTICES.md).

### Q52. Source of the Unbihexium code

Select where the Unbihexium code you use comes from.

_Select one._

- [ ] Official repository unbihexium-oss/unbihexium
- [ ] Fork of the official repository without modifications
- [ ] Fork of the official repository with modifications
- [ ] Official PyPI distribution
- [ ] Mirror or vendored copy
- [ ] Other:

### Q53. Local modifications

Declare whether your copy differs from the official release or branch.

_Select one._

- [ ] No modifications
- [ ] Configuration changes only
- [ ] Source code modifications
- [ ] Model weights replaced, retrained or fine-tuned
- [ ] Source code and model modifications
- [ ] Other:

### Q54. Description of local modifications

Describe every modification, including files changed and why. Modified MPL-2.0 files remain under the MPL-2.0 ([Section 3.1](https://www.mozilla.org/en-US/MPL/2.0/)). Write `N/A` if there are none.

Answer:

### Q55. Source of third-party libraries

Select every source from which third-party Python or system libraries were obtained.

_Select all that apply._

- [ ] PyPI wheels
- [ ] conda-forge
- [ ] Operating system packages
- [ ] Built from source
- [ ] Private or internal package index mirror
- [ ] Other:

### Q56. Licences of additional libraries

Declare the licence categories of any libraries beyond those declared in `pyproject.toml` that are involved.

_Select all that apply._

- [ ] Only the dependencies declared in pyproject.toml
- [ ] Additional permissive dependencies (MIT, BSD, Apache-2.0, MPL-2.0)
- [ ] Additional weak copyleft dependencies (LGPL, EPL)
- [ ] Additional strong copyleft dependencies (GPL, AGPL)
- [ ] Additional proprietary or unknown-licence dependencies
- [ ] Other:

### Q57. Known vulnerabilities in dependencies

Run `pip-audit` or an equivalent scanner and declare the result.

_Select one._

- [ ] Scanned, no known vulnerabilities
- [ ] Scanned, known vulnerabilities unrelated to this submission
- [ ] Scanned, known vulnerabilities possibly related to this submission
- [ ] Not scanned
- [ ] Other:

### Q58. Models involved

Declare which models are involved. Select every option that applies.

_Select all that apply._

- [ ] No models involved
- [ ] Unbihexium model zoo models (Git LFS)
- [ ] Unbihexium models downloaded from a release asset
- [ ] Custom models trained by me or my organisation
- [ ] Unbihexium models fine-tuned by me or my organisation
- [ ] Third-party models
- [ ] Other:

### Q59. Model identifiers

List every model identifier and variant, for example `ship_detector_base` or `ubx-det-multiclass-1.0.0`. For third-party models give the name, version and source. Write `N/A` if no models are involved.

Answer:

### Q60. Model variants

Select every model variant involved.

_Select all that apply._

- [ ] tiny
- [ ] base
- [ ] large
- [ ] mega
- [ ] Not applicable
- [ ] Other:

### Q61. Model integrity verification

Declare whether the model files were verified against the published SHA256 checksums.

_Select one._

- [ ] Verified against model_zoo/checksums.txt
- [ ] Verified against the sha256 field of the model manifest
- [ ] Not verified
- [ ] Not applicable, no models involved
- [ ] Other:

### Q62. Model card and licence review

Declare whether you read the model card, including intended use, limitations and licence.

_Select one._

- [ ] Read in full, including limitations and licence
- [ ] Read partially
- [ ] Not read
- [ ] Not applicable, no models involved
- [ ] Other:

### Q63. Training data of any model you trained or fine-tuned

Declare the provenance of the training data for any model you trained or fine-tuned.

_Select all that apply._

- [ ] No training or fine-tuning performed
- [ ] Synthetic data only
- [ ] Openly licensed data (for example Copernicus, Landsat, CC-BY datasets)
- [ ] Commercially licensed data with a licence that permits training
- [ ] Data that contains personal data
- [ ] Data of unknown licence or provenance
- [ ] Other:

### Q64. Data involved

Select every type of data involved.

_Select all that apply._

- [ ] No data involved
- [ ] Sentinel-2 optical
- [ ] Sentinel-1 SAR
- [ ] Landsat
- [ ] Other Copernicus services data
- [ ] Commercial very high resolution optical imagery
- [ ] Commercial SAR imagery
- [ ] Aerial or drone imagery
- [ ] Elevation models (DEM, DSM, DTM)
- [ ] Vector data (for example OpenStreetMap)
- [ ] Street-level imagery
- [ ] Synthetic data
- [ ] Other:

### Q65. Data licences

Declare the licences under which the data involved is available to you.

_Select all that apply._

- [ ] Copernicus full, free and open data policy
- [ ] Open licences (CC0, CC-BY, ODbL or equivalent)
- [ ] Commercial licence that permits this use
- [ ] Internal or proprietary data of my organisation
- [ ] Unknown licence
- [ ] Not applicable, no data involved
- [ ] Other:

### Q66. Personal data

Declare whether any data involved in or attached to this pull request is personal data as defined in [GDPR Article 4(1)](https://gdpr-info.eu/art-4-gdpr/). High resolution imagery and location data can be personal data.

_Select all that apply._

- [ ] No personal data
- [ ] Imagery that could identify people, vehicles or private property
- [ ] Location data linked to individuals
- [ ] Direct identifiers such as names or e-mail addresses
- [ ] Special categories of data under GDPR Article 9
- [ ] Not sure
- [ ] Other:

### Q67. Geographic sensitivity

Declare whether the area of interest is sensitive.

_Select all that apply._

- [ ] Not sensitive
- [ ] Critical infrastructure
- [ ] Military or defence sites
- [ ] Locations of protected species or habitats
- [ ] Areas of armed conflict or humanitarian crisis
- [ ] Not applicable, no geographic data involved
- [ ] Other:

### Q68. Declarations

Confirm each declaration. They cover the repository, libraries, models and data.

_Tick every statement._

- [ ] I declare that the repository and version information above is complete and accurate.
- [ ] I declare that every third-party library involved is listed above or in `pyproject.toml`, and that its licence is compatible with the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/).
- [ ] I declare that every model involved is listed above together with its provenance, licence and [model card](https://github.com/unbihexium-oss/unbihexium/blob/main/model_zoo/MODEL_CARDS.md) status.
- [ ] I declare that I have the right to use every dataset involved and that any data I attach may be published in this public repository.

## AI usage declaration

Declare whether AI systems, as defined in [EU AI Act Article 3(1)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3), were used to prepare this pull request. AI assistance is allowed, but it must be disclosed so reviewers can calibrate their review and so the project can meet the transparency expectations of [EU AI Act Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50).

### Q69. Use of AI

Select how much of this pull request, including code, text, logs and analysis, was produced with the help of an AI system.

_Select one._

- [ ] No AI system was used
- [ ] Minor: spelling, grammar or translation only
- [ ] Partial: some sections or code were drafted with AI
- [ ] Substantial: most of the content was drafted with AI
- [ ] Other:

### Q70. Types of AI tools used

Select every type of AI tool used.

_Select all that apply._

- [ ] Not applicable, no AI used
- [ ] Code completion in an editor
- [ ] Chat-based general-purpose language model assistant
- [ ] Agentic coding assistant that edits files or runs commands
- [ ] Machine translation
- [ ] Image or diagram generation
- [ ] Log, trace or data analysis tool
- [ ] Other:

### Q71. AI tool names, versions and providers

Name each AI tool, its model or version and its provider. Write `N/A` if no AI was used.

Answer:

### Q72. Parts produced with AI assistance

Select every part that was produced or substantially changed with AI assistance.

_Select all that apply._

- [ ] Not applicable, no AI used
- [ ] Problem description or summary
- [ ] Reproduction steps or example code
- [ ] Analysis of logs or errors
- [ ] Source code
- [ ] Tests
- [ ] Documentation text
- [ ] Translation
- [ ] Diagrams or images
- [ ] Other:

### Q73. Human review of AI output

Describe the human review applied to AI generated content before submission.

_Select one._

- [ ] Not applicable, no AI used
- [ ] Every line reviewed, understood and verified by me
- [ ] Reviewed and verified by running tests or reproductions
- [ ] Partially reviewed
- [ ] Not reviewed
- [ ] Other:

### Q74. Data provided to AI tools

Declare what data you entered into AI tools. Entering personal data into a third-party AI service is processing under [GDPR Article 4(2)](https://gdpr-info.eu/art-4-gdpr/) and may be a transfer under [GDPR Article 44](https://gdpr-info.eu/art-44-gdpr/).

_Select all that apply._

- [ ] Not applicable, no AI used
- [ ] Only public information
- [ ] Proprietary code or data, with permission of the owner
- [ ] Personal data
- [ ] Confidential data without explicit permission
- [ ] Other:

### Q75. Rights to AI generated output

Declare whether the terms of the AI tools allow you to submit their output under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) and whether the output may reproduce third-party code.

_Select one._

- [ ] Not applicable, no AI used
- [ ] Tool terms grant me the rights needed and I checked the output for copied third-party code
- [ ] Tool terms grant me the rights needed, but I did not check for copied code
- [ ] Not sure
- [ ] Other:

### Q76. Verification of AI generated claims

Declare how factual and technical claims produced with AI (for example API behaviour, benchmark numbers, legal statements) were verified.

_Select one._

- [ ] Not applicable, no AI used
- [ ] Verified against source code, documentation or primary sources
- [ ] Verified by running the code
- [ ] Not verified
- [ ] Other:

### Q77. Additional AI disclosure details

Describe anything reviewers should know about AI assistance, for example prompts that shaped the design or content that was AI generated and not modified. Write `N/A` if no AI was used.

Answer:

### Q78. AI usage attestation

Confirm each statement.

_Tick every statement._

- [ ] The AI usage answers above are complete and accurate.
- [ ] I did not use AI to fabricate logs, stack traces, benchmark results, test results or citations.
- [ ] I take full responsibility for the content, whether or not AI assisted in producing it.

## Regulatory classification

Unbihexium is distributed as free and open-source software. Obligations under the regulations below mostly fall on those who deploy it or place products built with it on the market. Answer for your own use. The full texts are the [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng) and the [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng); every answer links to the relevant article.

### Q79. Role under the GDPR

Select your role for any personal data processing involved, as defined in [GDPR Article 4(7) and 4(8)](https://gdpr-info.eu/art-4-gdpr/) and [Article 26](https://gdpr-info.eu/art-26-gdpr/).

_Select one._

- [ ] No personal data processed
- [ ] Controller
- [ ] Processor
- [ ] Joint controller
- [ ] Data subject
- [ ] Other:

### Q80. Lawful basis for processing

Select the lawful basis under [GDPR Article 6(1)](https://gdpr-info.eu/art-6-gdpr/) for any processing of personal data in your use of Unbihexium.

_Select one._

- [ ] Not applicable, no personal data processed
- [ ] Article 6(1)(a) consent
- [ ] Article 6(1)(b) performance of a contract
- [ ] Article 6(1)(c) legal obligation
- [ ] Article 6(1)(d) vital interests
- [ ] Article 6(1)(e) public interest or official authority
- [ ] Article 6(1)(f) legitimate interests
- [ ] Other:

### Q81. Data protection impact assessment

Declare the status of a DPIA under [GDPR Article 35](https://gdpr-info.eu/art-35-gdpr/). A DPIA is typically required for systematic monitoring of publicly accessible areas on a large scale.

_Select one._

- [ ] Not required, no high-risk processing
- [ ] Completed
- [ ] Planned
- [ ] Not assessed
- [ ] Other:

### Q82. Role under the EU AI Act

Select your role as defined in [EU AI Act Article 3](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3) and [Article 25](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-25).

_Select one._

- [ ] Not placing on the market or putting into service any AI system
- [ ] Provider (Article 3(3))
- [ ] Deployer (Article 3(4))
- [ ] Importer (Article 3(6))
- [ ] Distributor (Article 3(7))
- [ ] Product manufacturer integrating an AI system (Article 25(3))
- [ ] Authorised representative (Article 3(5))
- [ ] Other:

### Q83. EU AI Act classification of your use

Classify the AI system in which you use Unbihexium. See [Article 2](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-2) for the exclusions, [Article 5](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5) for prohibited practices, [Article 6](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-6) and [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3) for high-risk systems and [Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) for transparency obligations.

_Select one._

- [ ] No AI system involved
- [ ] Excluded: exclusively military, defence or national security (Article 2(3))
- [ ] Excluded: scientific research and development only (Article 2(6))
- [ ] Excluded: research, testing or development before market placement (Article 2(8))
- [ ] Minimal risk
- [ ] Transparency obligations apply (Article 50)
- [ ] High-risk under Article 6(2) and Annex III
- [ ] High-risk under Article 6(1) as a safety component
- [ ] Not yet assessed
- [ ] Other:

### Q84. Annex III area

If your system is high-risk, select the area listed in [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3).

_Select one._

- [ ] Not high-risk
- [ ] Point 1: biometrics
- [ ] Point 2: critical infrastructure
- [ ] Point 3: education and vocational training
- [ ] Point 4: employment and workers management
- [ ] Point 5: access to essential private and public services
- [ ] Point 6: law enforcement
- [ ] Point 7: migration, asylum and border control management
- [ ] Point 8: administration of justice and democratic processes
- [ ] Other:

### Q85. Intended end use

Select every intended end use of your work with Unbihexium.

_Select all that apply._

- [ ] Civil research
- [ ] Commercial civil use
- [ ] Public sector civil use
- [ ] Environmental monitoring
- [ ] Humanitarian or disaster response
- [ ] Defence or military
- [ ] Law enforcement
- [ ] Border surveillance
- [ ] Other:

### Q86. Export control assessment

Declare the export control status of your use. Most published open-source code is not subject to the EAR under [15 CFR 734.7](https://www.ecfr.gov/current/title-15/section-734.7), but end-use controls in [Regulation (EU) 2021/821 Article 4](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng#art_4) and [EAR Part 744](https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C/part-744) still apply.

_Select one._

- [ ] No export involved
- [ ] Assessed, no licence required
- [ ] Assessed, licence obtained
- [ ] Not assessed
- [ ] Other:

### Q87. Frameworks and standards applied

Select every AI governance or security framework applied in your use.

_Select all that apply._

- [ ] No framework applied
- [ ] NIST AI Risk Management Framework 1.0
- [ ] ISO/IEC 42001 AI management system
- [ ] ISO/IEC 23894 AI risk management
- [ ] ISO/IEC 27001 information security
- [ ] OECD AI Principles
- [ ] Other:

## Regulatory confirmations

Confirm every statement. Each link opens the exact article or section it refers to.

### Q88. GDPR confirmations (Regulation (EU) 2016/679)

Applies to any processing of personal data. Full text: [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng).

_Tick every statement._

- [ ] My processing follows the principles in [Article 5](https://gdpr-info.eu/art-5-gdpr/), including data minimisation and purpose limitation.
- [ ] Any processing has a lawful basis under [Article 6](https://gdpr-info.eu/art-6-gdpr/) and, for special categories, a condition under [Article 9](https://gdpr-info.eu/art-9-gdpr/).
- [ ] Data subjects are informed as required by [Articles 13 and 14](https://gdpr-info.eu/art-13-gdpr/) and automated decisions follow [Article 22](https://gdpr-info.eu/art-22-gdpr/).
- [ ] I apply data protection by design and by default under [Article 25](https://gdpr-info.eu/art-25-gdpr/) and security measures under [Article 32](https://gdpr-info.eu/art-32-gdpr/).
- [ ] I carry out a DPIA where required by [Article 35](https://gdpr-info.eu/art-35-gdpr/).
- [ ] International transfers of personal data comply with [Chapter V, Articles 44 to 49](https://gdpr-info.eu/art-44-gdpr/).
- [ ] This pull request contains no personal data of third parties.

### Q89. EU AI Act confirmations (Regulation (EU) 2024/1689)

Applies to AI systems placed on the market, put into service or used in the EU. Full text: [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng).

_Tick every statement._

- [ ] I do not use Unbihexium for any prohibited practice listed in [Article 5](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5), including untargeted scraping for facial recognition databases and real-time remote biometric identification.
- [ ] If my system is high-risk under [Article 6](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-6) and [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3), I meet the requirements of [Articles 9 to 15](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-9).
- [ ] As a deployer I follow [Article 26](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-26), including human oversight, and carry out a fundamental rights impact assessment under [Article 27](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-27) where required.
- [ ] I meet the transparency obligations of [Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) where they apply.
- [ ] Staff dealing with the AI system have sufficient AI literacy as required by [Article 4](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-4).
- [ ] I understand that the free and open-source exemptions in [Article 2(12)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-2) and [Article 53(2)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-53) do not cover high-risk or prohibited uses.

### Q90. Cyber security and product liability confirmations

Covers the [Cyber Resilience Act](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng), the [NIS2 Directive](https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng) and the [Product Liability Directive](https://eur-lex.europa.eu/eli/dir/2024/2853/oj/eng).

_Tick every statement._

- [ ] I report vulnerabilities privately through a [security advisory](https://github.com/unbihexium-oss/unbihexium/security/advisories/new) following [SECURITY.md](https://github.com/unbihexium-oss/unbihexium/blob/main/SECURITY.md), consistent with the coordinated vulnerability handling of [CRA Article 13](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng#art_13) and [Article 14](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng#art_14).
- [ ] If I place a product with digital elements that includes Unbihexium on the EU market, I meet the manufacturer obligations of [CRA Article 13](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng#art_13) and the essential requirements of CRA Annex I.
- [ ] I understand the role of open-source software stewards under [CRA Article 24](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng#art_24).
- [ ] If my organisation is an essential or important entity, I apply the risk management measures of [NIS2 Article 21](https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng#art_21) and the reporting obligations of [NIS2 Article 23](https://eur-lex.europa.eu/eli/dir/2022/2555/oj/eng#art_23).
- [ ] I understand that free and open-source software supplied outside a commercial activity is excluded by [Product Liability Directive Article 2(2)](https://eur-lex.europa.eu/eli/dir/2024/2853/oj/eng#art_2) and that the software is provided without warranty under [MPL-2.0 Sections 6 and 7](https://www.mozilla.org/en-US/MPL/2.0/).

### Q91. Export control and sanctions confirmations

Geospatial analytics and object detection can be dual-use.

_Tick every statement._

- [ ] I comply with [Regulation (EU) 2021/821](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng), including authorisation requirements in [Article 3](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng#art_3), end-use controls in [Article 4](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng#art_4) and cyber-surveillance controls in [Article 5](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng#art_5).
- [ ] I comply with EU restrictive measures, including [Regulation (EU) No 833/2014](https://eur-lex.europa.eu/eli/reg/2014/833/oj/eng), [Article 2](https://eur-lex.europa.eu/eli/reg/2014/833/oj/eng#art_2) on dual-use goods and technology.
- [ ] I comply with the US Export Administration Regulations, including [15 CFR 734.7](https://www.ecfr.gov/current/title-15/section-734.7) on published information and the end-use and end-user controls in [15 CFR Part 744](https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C/part-744).
- [ ] I comply with US sanctions administered under [31 CFR Chapter V](https://www.ecfr.gov/current/title-31/subtitle-B/chapter-V).

### Q92. Geospatial data, open data and copyright confirmations

Covers the terms under which Earth observation and geospatial data may be used.

_Tick every statement._

- [ ] I use Copernicus data in line with [Regulation (EU) 2021/696 Article 55](https://eur-lex.europa.eu/eli/reg/2021/696/oj/eng#art_55) and [Delegated Regulation (EU) No 1159/2013](https://eur-lex.europa.eu/eli/reg_del/2013/1159/oj/eng), including attribution.
- [ ] I respect access limitations on spatial data sets under [INSPIRE Directive Article 13](https://eur-lex.europa.eu/eli/dir/2007/2/oj/eng#art_13).
- [ ] I respect the conditions for re-use of public sector data under [Open Data Directive Article 14](https://eur-lex.europa.eu/eli/dir/2019/1024/oj/eng#art_14) and [Implementing Regulation (EU) 2023/138](https://eur-lex.europa.eu/eli/reg_impl/2023/138/oj/eng) on high-value datasets.
- [ ] Any text and data mining for model training respects [DSM Directive Articles 3 and 4](https://eur-lex.europa.eu/eli/dir/2019/790/oj/eng#art_4), including rights reservations under Article 4(3).

### Q93. Other privacy law confirmations

Covers telemetry and privacy regimes outside the GDPR.

_Tick every statement._

- [ ] Any telemetry or storage of information on user devices complies with [ePrivacy Directive Article 5(3)](https://eur-lex.europa.eu/eli/dir/2002/58/oj/eng#art_5); Unbihexium telemetry is opt-in as described in [PRIVACY.md](https://github.com/unbihexium-oss/unbihexium/blob/main/PRIVACY.md).
- [ ] For processing subject to UK law I comply with the [UK GDPR](https://www.legislation.gov.uk/eur/2016/679/article/5) and the [Data Protection Act 2018](https://www.legislation.gov.uk/ukpga/2018/12/contents).
- [ ] For consumers in California I comply with the CCPA as amended by the CPRA, including [Civil Code 1798.100](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=1798.100), [1798.105](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=1798.105) and [1798.120](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=1798.120).

### Q94. Licence confirmations (MPL-2.0)

Unbihexium is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/). The project copy is [LICENSE.txt](https://github.com/unbihexium-oss/unbihexium/blob/main/LICENSE.txt).

_Tick every statement._

- [ ] I have read the licence grants in [MPL-2.0 Section 2](https://www.mozilla.org/en-US/MPL/2.0/).
- [ ] When I distribute Covered Software in source or executable form I meet [MPL-2.0 Sections 3.1 and 3.2](https://www.mozilla.org/en-US/MPL/2.0/), and for Larger Works [Section 3.3](https://www.mozilla.org/en-US/MPL/2.0/).
- [ ] I keep the licence notices and the third-party notices in [NOTICE](https://github.com/unbihexium-oss/unbihexium/blob/main/NOTICE) and [THIRD_PARTY_NOTICES.md](https://github.com/unbihexium-oss/unbihexium/blob/main/THIRD_PARTY_NOTICES.md).
- [ ] I understand the disclaimer of warranty and limitation of liability in [MPL-2.0 Sections 6 and 7](https://www.mozilla.org/en-US/MPL/2.0/).

### Q95. Project policy confirmations

Covers the policies of this repository and of GitHub.

_Tick every statement._

- [ ] I follow the [Code of Conduct](https://github.com/unbihexium-oss/unbihexium/blob/main/CODE_OF_CONDUCT.md).
- [ ] I have read [CONTRIBUTING.md](https://github.com/unbihexium-oss/unbihexium/blob/main/CONTRIBUTING.md), [COMPLIANCE.md](https://github.com/unbihexium-oss/unbihexium/blob/main/COMPLIANCE.md) and [PRIVACY.md](https://github.com/unbihexium-oss/unbihexium/blob/main/PRIVACY.md).
- [ ] This pull request complies with the [GitHub Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies).

### Q96. Compliance notes

Add anything relevant to the declarations and confirmations above, for example jurisdiction-specific constraints or pending assessments. Write `N/A` if there is nothing to add.

Answer:

## Closing

Final details before you submit.

### Q97. Additional context

Add any other context, screenshots or links that help maintainers handle this pull request. Write `N/A` if there is nothing to add.

Answer:

### Q98. Final confirmation

Confirm each statement before submitting.

_Tick every statement._

- [ ] The information in this pull request is accurate and complete to the best of my knowledge.
- [ ] I removed secrets, access tokens, credentials and internal host names.
- [ ] I removed personal data and confidential information.
- [ ] I wrote this submission in English.
