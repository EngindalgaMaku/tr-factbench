# Repository update notes — 2026-07-26

> Historical snapshot: counts and model-selection wording in this file were superseded by `UPDATE_NOTES_2026-07-27.md` after the H2 five-seed series was added.

This snapshot synchronizes the experiment archive, documentation, annotation workflow, and generated reports for the TR-FactBench v0.1.0 research preview.

## Completed in this update

- Synchronized canonical experiment reporting with the current archive: 136 validated run directories, 416 canonical metric rows, 118 successful runs, 17 class-collapse runs, and 1 degenerate-skewed run.
- Separated the historical G1 dropout ablation from the seed-corrected H1 final validation series.
- Updated the selected checkpoint metadata to `H1_DROP10_S45` and linked it to the matching `H1_SENTINEL_DROP10_S45_RERUN` reproducibility run.
- Corrected the reproducibility sentinel command and lock-file descriptions. The sentinel meets the predeclared absolute Macro-F1 tolerance of 0.010 on both development sets; this is performance-level, not prediction-identical, reproducibility.
- Updated README, Turkish README, data card, model card, benchmark report, lock file, result tables, statistics, prediction index, and data checksums.
- Corrected the annotation terminology: LLM-assisted review is recorded as candidate-label refinement, not as independent human annotation.
- Migrated active annotation metadata to record whether candidate labels and LLM suggestions were visible.
- Preserved the completed Dev600 candidate-label review and the ongoing first blind-human pass without altering human decisions.
- Added artifact-derived `annotation_status.json` reporting.
- Replaced the legacy invalid 500-example train IAA setup with a context-grouped, stratified 600-example audit sample: 150 contexts, 50 per domain, all four claims per context, opaque identifiers, and candidate labels stored separately.
- Rewrote the IAA utility for two independent blind-human files, unweighted Cohen's kappa, class/domain diagnostics, disagreement export, and context-level bootstrap confidence intervals.
- Hid internal example identifiers in the review interface and added a train-audit preset.
- Preserved normalized LLM preannotations as explicit artifacts and added postprocessing manifests. Historical raw artifacts were not available in this snapshot; this limitation is documented rather than concealed.
- Updated the shared model-loading configuration so the four-label metadata is explicit in training and evaluation utilities.

## Validation performed

- Python compilation: passed for `scripts`, `src`, and `annotation`.
- Release validation: passed with 136/136 structurally valid run directories, 416 canonical metric rows, and no duplicate `(run_id, eval_set)` rows.
- Dataset checksums: all listed files passed.
- `final_config.lock.yaml`: valid YAML and its canonical metrics hash matches the current metrics file.
- Train audit sample: 600 examples, 150 four-claim context groups, and 50 contexts per domain; no candidate labels or source identifiers are present in the blind input.
- LLM normalized artifacts: legacy aliases and normalized files are byte-identical, with matching postprocessing-manifest hashes.

## Work still requiring people or unavailable artifacts

- Finish Dev600 blind annotator 1: currently 400/600.
- Complete the independent second Dev600 blind annotation.
- Compute pre-adjudication agreement and Cohen's kappa, then adjudicate disagreements.
- Complete two blind annotations for the 600-example train audit sample.
- Decide and apply any label/claim corrections only after the planned human-validation workflow.
- Perform a clean-environment LoRA adapter load and inference test when the final adapter weights are available.

This remains a research-preview repository and is not yet a fully human-adjudicated gold benchmark.
