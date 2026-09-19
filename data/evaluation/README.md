# Evaluation datasets

Evaluation files are separated from the training split to prevent accidental mixing.

- `preview_v0.1/`: recommended single-annotator-reviewed files for new preview evaluations.
- `legacy_experiment_version/`: byte-identical pre-review files used by the archived encoder and LLM experiments.
- `gold_v1.0/`: frozen 480-example dual-annotator gold (A/B complete, adjudicated). Canonical labels are in `gold_v1.0/gold/`. Raw annotator files, agreement, adjudication, and provenance sit beside them.

Do not compare newly produced preview scores directly with archived scores without clearly naming the evaluation version.
