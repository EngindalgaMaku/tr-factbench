# TR-FactBench v0.1.0-preview

TR-FactBench is a multi-domain Turkish benchmark for source-grounded factuality verification. It covers finance, legal, and medical content with four labels: `supported`, `partially_supported`, `contradicted`, and `unverifiable`.

## Preview annotation status

The first independent blind-human pass is complete for both evaluation sets:

- DEV960-GEN: 960/960 reviewed by annotator 1
- DEV600-STR: 600/600 reviewed by annotator 1
- DEV600-STR annotator 2: 80/600

A second annotation pass and adjudication are planned for a future release.

## Files for new evaluations

Use:

- `data/evaluation/preview_v0.1/dev_general.jsonl`
- `data/evaluation/preview_v0.1/dev_stress.jsonl`
- Domain-specific views: `data/evaluation/preview_v0.1/domain/`
- Dev600 ID mapping: `data/manifests/dev600_legacy_to_preview_id_map.csv`

Archived encoder and LLM results remain reproducible through the byte-identical files in `data/evaluation/legacy_experiment_version/`. All archived manifests point to that directory and preserve their former recorded paths in relocation metadata.

## Human-review changes

- DEV960: 2 label revisions
- Dev600: 3 label revisions
- Dev600: 1 documented claim correction

## Included artifacts

- 4,800-example training split
- 960-example general evaluation split
- 600-example stress evaluation split
- 144 archived encoder experiment runs
- Four LLM baselines in zero-shot and fixed 8-shot conditions
- Run-level predictions, metrics, manifests, validation outputs, data card, model card, and label guide

## Validation status

- Repository preflight: PASS
- Run validation: 144/144 PASS
- Canonical metric rows: 432
- Preview evaluation file validation: PASS
- Release lock: `preview_release_ready: true`

## Evaluation-version note

Archived scores use the original pre-review evaluation files. The provisional preview files contain limited human-review revisions, so scores may shift slightly when systems are evaluated again on the reviewed versions. Training and API inference were not repeated for this preview packaging update.

## Citation

See `CITATION.cff`.
