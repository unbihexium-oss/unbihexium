<!--
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

=============================================================================
Project     : Unbihexium
File        : RESPONSIBLE_USE.md
Title       : Responsible Use Policy
Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
Affiliation : University of Helsinki
Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
Licence     : Mozilla Public License 2.0, see LICENSE.txt
Format      : Markdown (CommonMark with GitHub Flavored Markdown extensions)
=============================================================================
-->

# Responsible Use Policy

| Field | Value |
| --- | --- |
| Document | UBX-DOC-204 |
| Version | 2.0 |
| Status | Active |
| Last reviewed | 2026-09-23 |
| Owner | Unbihexium maintainers (see [MAINTAINERS.md](MAINTAINERS.md)) |
| Applies to | Unbihexium 1.0.x and the main branch, including the model zoo |

## Abstract

Unbihexium provides Earth observation, remote sensing, SAR and geospatial analytics, including a model zoo with detectors for vessels, aircraft, vehicles and military objects. Such capabilities serve environmental science, agriculture and disaster response, and they can also be used for surveillance and military purposes. This policy, written for users, deployers, contributors and reviewers, states the uses the project is designed for, the uses it does not support, the dual-use considerations that apply to its most sensitive components, the technical limits of the starter models, and the responsibilities of those who build systems with the software. It complements the Mozilla Public License 2.0 in [LICENSE.txt](LICENSE.txt): the licence grants the right to use the software and this policy does not restrict it, but the policy defines which uses the maintainers support, document and accept contributions for. It is not legal advice.

## Contents

- [1. Introduction](#1-introduction)
- [2. Limits of the Starter Models](#2-limits-of-the-starter-models)
- [3. Intended Uses](#3-intended-uses)
- [4. Uses the Project Does Not Support](#4-uses-the-project-does-not-support)
- [5. Dual-Use Considerations](#5-dual-use-considerations)
- [6. Responsibilities of Deployers](#6-responsibilities-of-deployers)
- [7. Contributions in Sensitive Areas](#7-contributions-in-sensitive-areas)
- [8. Reporting Misuse or Concerns](#8-reporting-misuse-or-concerns)
- [References](#references)

## 1. Introduction

### 1.1 Purpose

Open source software cannot control who runs it. The purpose of this policy is therefore to state clearly what the project is for, to make the limits of the software visible to those who might rely on it, and to set the boundaries within which the maintainers develop, document and support it.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY in this document are to be interpreted as described in RFC 2119 [1] and RFC 8174 [2] when they appear in all capitals. Statements addressed to deployers describe what the maintainers expect as a condition of support; they do not modify the licence and do not replace the law that applies to the deployer.

### 1.3 Scope

The policy applies to the library, the command line interface, the REST service, the container image and the model zoo. The library registers 147 capabilities in 16 domains, and the model zoo contains 520 models (130 families in 4 variants: tiny, base, large and mega), as reported by `unbihexium info`.

## 2. Limits of the Starter Models

### 2.1 The Models Are Untrained

The 520 models of the model zoo are untrained starter models. Each consists of a complete network architecture with deterministic starter weights generated from its identifier, so that every user obtains bit-identical, verifiable weights without a download. The weights have not been learned from any data. **Until a model is trained on labelled data, its predictions are meaningless**: detections, class maps, scores and regression values produced by a starter model carry no information about the input image.

The exception is the 7 spectral index families (28 models: `evi_calculator`, `msi_calculator`, `nbr_calculator`, `ndvi_calculator`, `ndwi_calculator`, `savi_calculator` and `vegetation_condition`). They compute exact published formulas and need no training. Their results are only as good as the input reflectance data, and their interpretation (for example a vegetation threshold) still depends on sensor, season and region.

Whether a model needs training is recorded in the catalogue and can be checked before use:

```python
from unbihexium.zoo import get_model

print(get_model("ship_detector_tiny").requires_training)    # True
print(get_model("ndvi_calculator_tiny").requires_training)  # False
```

The same flag appears as `requires_training` in the output of `unbihexium zoo info <model_id>`, and every model card in [model_zoo/cards/](model_zoo/cards/) states it at the top.

### 2.2 No Published Accuracy

The project publishes no accuracy, precision or recall figures for any model, because no trained model is distributed. Any claim about the performance of a model built with Unbihexium refers to the training and validation performed by whoever trained it.

### 2.3 Limits After Training

Models trained by users remain subject to the usual limits of remote sensing and machine learning:

- performance changes with the sensor, spatial resolution, viewing geometry, atmosphere, season and region, and a model trained in one setting can fail silently in another;
- the smallest detectable object is limited by the ground sampling distance; at the resolution of common open data, individual people cannot be resolved;
- class labels and training data carry the biases of those who produced them;
- detections are probabilistic and include false positives and false negatives.

Users MUST validate a trained model on independent reference data that represent the area, sensor and period of intended use, and SHOULD report the validation method and results with any output used for decisions. Guidance on training is in [docs/model_zoo/training.md](docs/model_zoo/training.md).

## 3. Intended Uses

Unbihexium is designed for, and the maintainers support its use in:

- environmental monitoring, including forests, land degradation, water bodies, water quality, pollution and protected areas;
- agriculture, including crop mapping, crop condition, yield estimation and land suitability;
- natural hazards and disaster management, including floods, wildfires, landslides, subsidence and damage assessment for humanitarian response;
- urban planning, infrastructure and asset monitoring, and renewable energy siting;
- maritime safety and environmental protection, such as oil spill detection and vessel traffic awareness in support of search and rescue;
- verification and transparency, such as the independent monitoring of commitments under environmental or arms control agreements, carried out in accordance with applicable law;
- scientific research, teaching and the development of new methods, including the training and evaluation of models.

The capability domains are documented in [docs/capabilities/index.md](docs/capabilities/index.md).

## 4. Uses the Project Does Not Support

The maintainers do not support, will not provide help for, and will not accept contributions intended for the following uses. Users MUST NOT use Unbihexium for any practice that is unlawful where it takes place.

1. **Prohibited AI practices.** Any practice prohibited by Article 5 of the EU Artificial Intelligence Act [3], including real-time remote biometric identification in publicly accessible spaces for law enforcement outside the narrow exceptions of that article, the untargeted scraping of facial images, and social scoring.
2. **Surveillance of individuals.** Tracking, monitoring or profiling of identifiable individuals or groups without a lawful basis under Article 6 of the General Data Protection Regulation [4] or equivalent law, including the monitoring of journalists, human rights defenders, political opponents or minorities.
3. **Weapons targeting.** Selection or engagement of targets for weapons, guidance of weapons, or any use that directly supports the application of lethal force.
4. **Violations of international law.** Any use in violation of international humanitarian law or international human rights law.
5. **Breach of export controls or sanctions.** Supply or use in breach of export controls or restrictive measures, including Regulation (EU) 2021/821 [5], see [COMPLIANCE.md](COMPLIANCE.md).
6. **Misrepresentation of outputs.** Presenting outputs of untrained starter models, or of models that have not been validated for the purpose, as evidence or as reliable information for decisions about people, property or safety.
7. **Deceptive synthetic imagery.** Presenting images produced by the `synthetic_imagery` models (optical images synthesised from SAR) or by super-resolution models as real observations, without disclosing that they are generated.

## 5. Dual-Use Considerations

### 5.1 Sensitive Components

Several components have both civilian and military or security applications:

| Component | Civilian use | Concern |
| --- | --- | --- |
| Defence domain families: `military_objects_detector`, `target_detector`, `border_monitor`, `security_monitor`, `maritime_awareness` | verification, humanitarian monitoring, protection of critical sites, maritime safety | military intelligence, targeting, border enforcement against people seeking protection |
| Object detectors: `ship_detector`, `sar_ship_detector`, `aircraft_detector`, `vehicle_detector`, `object_detector` | traffic statistics, fisheries control, port and airport planning | tracking of specific assets and, indirectly, of their owners |
| Change detection and damage assessment | disaster response, urban growth | monitoring of sensitive sites or populations |
| Viewshed, route planning and cost surfaces | infrastructure planning, accessibility | military route and line-of-sight planning |

All of these are shipped as untrained starter architectures or generic algorithms of the kind found in textbooks and other open source libraries. The project provides no trained weights, no labelled data of military objects and no integration with weapon systems. The catalogue contains no models for detecting or identifying people, faces or other biometric features. The `target_detector` family is a single-class detector for a user-defined object of interest; what it detects depends entirely on the data it is trained on.

### 5.2 Expectations for Sensitive Uses

Those who use the components above in a security, defence or law enforcement context:

- MUST ensure that the use complies with international humanitarian law and international human rights law, and SHOULD follow the UN Guiding Principles on Business and Human Rights [6] when they act as a business;
- SHOULD, where they are a State developing or acquiring a means or method of warfare, include the system in the legal review required by Article 36 of Additional Protocol I to the Geneva Conventions [7];
- MUST keep a qualified human responsible for every decision that affects people, and MUST NOT act on a model output alone;
- SHOULD document the validation of each trained model for its specific operating conditions (Section 2.3).

The AI Act does not apply to AI systems used exclusively for military, defence or national security purposes (Article 2(3)) [3]; this exclusion does not remove the obligations of international law.

## 6. Responsibilities of Deployers

Anyone who builds a product or service with Unbihexium is responsible for its compliance and for the consequences of its use. In particular, deployers:

- MUST determine the intended purpose of their system and classify it under the EU AI Act [3]. A system that is high-risk under Article 6 and Annex III (for example as a safety component of critical infrastructure) must meet the requirements of Articles 9 to 15, and deployers of such systems have the obligations of Article 26;
- MUST provide effective human oversight of decisions that affect people (AI Act Article 14) and respect the rules on automated individual decision-making (GDPR Article 22);
- MUST treat high resolution imagery and location data as potentially personal data and carry out a data protection impact assessment where GDPR Article 35 requires it, as described in [PRIVACY.md](PRIVACY.md);
- MUST respect the licences of the imagery and data they process, including the attribution requirements for Copernicus data;
- MUST meet the transparency obligations of AI Act Article 50 where they apply, for example by marking synthetic images as artificially generated;
- SHOULD update the model card of every trained model with its training data, validation results and known limitations before sharing it;
- SHOULD secure deployments of the REST service as described in [SECURITY.md](SECURITY.md).

Further regulatory information is collected in [COMPLIANCE.md](COMPLIANCE.md).

## 7. Contributions in Sensitive Areas

The maintainers review contributions in the defence and security domains, and any contribution that could enable the uses in [Section 4](#4-uses-the-project-does-not-support), with particular care. They will decline contributions that:

- add biometric identification, face recognition or the detection or tracking of individual people;
- add trained weights or labelled datasets whose primary purpose is military targeting;
- integrate the software with weapon systems or with fire control;
- remove or weaken the statements about starter models in the documentation, the model cards or the catalogue.

A contributor who is unsure whether a proposal is acceptable can ask first with the "Compliance, licensing and ethics" issue form.

## 8. Reporting Misuse or Concerns

Concerns about misuse of the software, about bias or harmful behaviour of models, or about this policy can be raised with the "Compliance, licensing and ethics" issue form ([.github/ISSUE_TEMPLATE/06_compliance.yml](.github/ISSUE_TEMPLATE/06_compliance.yml)), or by e-mail to <yunus.z.imanov@helsinki.fi> when the matter should not be public. The maintainers cannot stop the use of open source software by third parties, but they will consider documentation changes, safeguards in the code and the refusal of related contributions. Security vulnerabilities MUST be reported privately as described in [SECURITY.md](SECURITY.md).

## References

[1] Bradner, S. Key words for use in RFCs to Indicate Requirement Levels. RFC 2119. 1997. <https://www.rfc-editor.org/rfc/rfc2119>

[2] Leiba, B. Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words. RFC 8174. 2017. <https://www.rfc-editor.org/rfc/rfc8174>

[3] European Parliament and Council. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). 2024. <https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng>

[4] European Parliament and Council. Regulation (EU) 2016/679 (General Data Protection Regulation). 2016. <https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng>

[5] European Parliament and Council. Regulation (EU) 2021/821 setting up a Union regime for the control of exports, brokering, technical assistance, transit and transfer of dual-use items. 2021. <https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng>

[6] United Nations Office of the High Commissioner for Human Rights. Guiding Principles on Business and Human Rights. 2011. <https://www.ohchr.org/en/publications/reference-publications/guiding-principles-business-and-human-rights>

[7] International Committee of the Red Cross. Protocol Additional to the Geneva Conventions of 12 August 1949 (Protocol I), Article 36: New weapons. 1977. <https://ihl-databases.icrc.org/en/ihl-treaties/api-1977/article-36>

<!--
=============================================================================
End of file RESPONSIBLE_USE.md
Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
Cite the project as described in CITATION.cff.
=============================================================================
-->
