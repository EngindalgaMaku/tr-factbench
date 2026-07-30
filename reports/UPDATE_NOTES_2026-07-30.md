# TR-FactBench Update Notes — 2026-07-30

## Preview release status

TR-FactBench is now packaged as `v0.1.0-preview` with one complete independent blind-human pass on both evaluation sets:

- DEV960-GEN annotator 1: 960/960 complete
- DEV600-STR annotator 1: 600/600 complete
- DEV600-STR annotator 2: 80/600 complete
- DEV960-GEN annotator 2 and adjudication: deferred to the stable release

The second annotation pass is not treated as a blocker for the research preview. The release remains explicitly single-annotator provisional and must not be presented as a fully adjudicated gold benchmark.

## Recommended preview files

New evaluations should use:

- `data/evaluation/preview_v0.1/dev_general.jsonl`
- `data/evaluation/preview_v0.1/dev_stress.jsonl`

The original `dev_general.jsonl` and `dev_stress.jsonl` files remain unchanged because the archived encoder and LLM scores were produced on those versions.

## Human-review revisions

- DEV960: 2/960 label revisions
- Dev600: 3/600 label revisions
- Dev600: 1 documented claim correction

The reviewed label distributions remain close to balanced. Macro-F1 remains the primary comparison metric.

## Repository changes

- Added a reproducible preview builder: `scripts/43_build_single_annotator_preview.py`
- Added the DEV960 preview audit file and manifest
- Added a combined preview evaluation manifest
- Updated README, Turkish README, data card, model card, citation metadata, schema, licensing note, lock file, and validation rules
- Updated release validation to require both provisional evaluation files and their audit copies
- Marked the release lock as `preview_release_ready: true`

## Validation

The final repository validation completed successfully:

- 144/144 run directories passed validation
- 432 canonical metric rows were rebuilt
- Both provisional evaluation files passed count, label, metadata, and duplicate-ID checks
- Release validation status: PASS

## Evaluation layout and reproducibility migration

- `data/primary/` now contains only `train.jsonl`.
- Reviewed files for new evaluations are under `data/evaluation/preview_v0.1/`.
- Byte-identical inputs used by archived results are under `data/evaluation/legacy_experiment_version/`.
- The legacy directory also contains exact domain subsets for A1 and A3/LODO runs.
- All 144 encoder run manifests and all archived LLM manifests were updated to repository-relative resolvable paths.
- Each encoder run manifest retains its former runtime paths under `path_relocation`.
- Derived-data writers now preserve the CRLF serialization used by the archived hashes.
- Validation checks train, DEV960, and DEV600 paths and SHA-256 values for every archived run.
