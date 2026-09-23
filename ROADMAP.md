# Roadmap

This roadmap lists planned work. It is a statement of intent, not a commitment to dates or features. Proposals and discussion happen in [issues](https://github.com/unbihexium-oss/unbihexium/issues) opened with the feature request form.

## Near Term

### Python Support

- Drop Python 3.10 in the first minor release after its upstream end of life on 2026-10-31, as defined in the [Python version support policy](VERSIONING.md).

### Quality

- Make the unit and integration test jobs fail on test failures and fix the tests that currently fail.
- Enforce the documented coverage threshold in CI.

### Model Zoo

- Align the model manifests with `model_zoo/manifest.schema.json` and validate them in CI.
- Regenerate `model_zoo/checksums.txt` so that it matches `model.sha256` for every variant, including the mega tier.
- Extend model cards with the information providers must give deployers under [EU AI Act Article 13](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-13).

### Security and Supply Chain

- Publish a security self-assessment and reference it in [security-insights.yml](security-insights.yml).
- Keep vulnerability handling aligned with the [Cyber Resilience Act](https://eur-lex.europa.eu/eli/reg/2024/2847/oj/eng) obligations for open-source software stewards.

## Longer Term

- Train and publish models on openly licensed Earth observation data with documented provenance.
- Publish reproducible benchmarks for every model family.

## Completed

See [CHANGELOG.md](CHANGELOG.md).
