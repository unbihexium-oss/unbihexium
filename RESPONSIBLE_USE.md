# Responsible Use

Unbihexium provides Earth observation, remote sensing and geospatial analytics, including object detection models that can be applied to people, vehicles, infrastructure and military objects. This document states the intended uses, the uses that are not permitted and the obligations of those who deploy the software. It complements the [MPL-2.0 licence](LICENSE.txt), which grants the right to use the software; this document does not restrict the licence but describes the conditions under which the maintainers support its use.

## Intended Use

- Environmental monitoring, agriculture, forestry, water and disaster management.
- Urban planning, infrastructure and asset monitoring.
- Scientific research and education.
- Neutral analytics for risk assessment, as described in [docs/capabilities](docs/capabilities/index.md).

## Model Limitations

- The published models are trained on synthetic data, as documented in [docs/model_zoo/licensing_and_provenance.md](docs/model_zoo/licensing_and_provenance.md). They are not validated for operational decisions.
- Performance differs across sensors, resolutions, regions and seasons. Validate every model on representative data before relying on it.
- Read the model card of every model you use in [model_zoo/cards](model_zoo/cards/).

## Uses That Are Not Supported

The maintainers do not support, and will not accept contributions intended for:

- Any practice prohibited by [EU AI Act Article 5](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5), including real-time remote biometric identification in publicly accessible spaces for law enforcement and untargeted scraping of facial images.
- Surveillance, tracking or profiling of individuals without a lawful basis under [GDPR Article 6](https://gdpr-info.eu/art-6-gdpr/) or equivalent law.
- Targeting of people or selection of targets for weapons.
- Use in breach of export controls or sanctions, including [Regulation (EU) 2021/821](https://eur-lex.europa.eu/eli/reg/2021/821/oj/eng), [Regulation (EU) No 833/2014](https://eur-lex.europa.eu/eli/reg/2014/833/oj/eng) and the [US Export Administration Regulations](https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C).

## Obligations of Deployers

If you build a product or service with Unbihexium, you are responsible for its compliance. In particular:

- Classify your AI system under the [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng). High-risk systems listed in [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3) must meet the requirements of [Articles 9 to 15](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-9), and deployers must follow [Article 26](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-26).
- Keep a human in the loop for decisions that affect people, as required by [EU AI Act Article 14](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14) and [GDPR Article 22](https://gdpr-info.eu/art-22-gdpr/).
- Treat high resolution imagery and location data as potentially personal data and carry out a data protection impact assessment where [GDPR Article 35](https://gdpr-info.eu/art-35-gdpr/) requires it.
- Respect the licences of the imagery and data you process, including the Copernicus attribution requirements.
- Meet the transparency obligations of [EU AI Act Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) where they apply.

Further guidance is collected in [COMPLIANCE.md](COMPLIANCE.md) and [PRIVACY.md](PRIVACY.md).

## Reporting Misuse or Concerns

Report concerns about misuse, bias or harmful behaviour of the models with the compliance issue form. Report security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
