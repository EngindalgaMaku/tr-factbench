# TR-FactBench update notes — 2026-07-27

This update integrates the completed H2/dropout=0.00 five-seed series and removes ambiguity between historical screening results and the current final-validation pipeline.

## Experiment reporting

- Canonical archive synchronized to **141 validated runs**, **426 canonical metric rows**, **123 successful runs**, **17 historical class-collapse runs**, and **1 severely skewed run**.
- Added the directly comparable seed-fixed H1/H2 dropout analysis:
  - H1/dropout=0.10: DEV960 0.8930 ± 0.0071; DEV600 0.8587 ± 0.0155.
  - H2/dropout=0.00: DEV960 0.9041 ± 0.0081; DEV600 0.8433 ± 0.0280.
- Retained H1/dropout=0.10 as the selected configuration because it has the higher and more stable stress-set result and a slightly higher two-set mean.
- Kept `H1_DROP10_S45` as the selected checkpoint within H1.
- Reframed the old G1 seed-46 collapse as a historical observed failure that was not reproduced in the five seed-fixed H2 runs.
- Replaced missing-value dashes for historical G1/dropout=0.00 with all-seed means and standard deviations that include the failed run.

## Figures and tables

- Regenerated all canonical English and Turkish figures.
- Clarified that the model-family figure belongs to the aligned F1 screening stage, not the final H1/H2 comparison.
- Regenerated confusion matrices from `H1_DROP10_S45` for both DEV960 and DEV600.
- Added final H1/H2 mean±SD and per-seed stability figures.
- Added `table_H_final_dropout_comparison.csv/.md` and `final_dropout_comparison.json`.
- Moved historical collapse/stability figures under `results/figures/historical/`.
- Moved exploratory/error-analysis figures under `results/figures/supplementary/`.

## Documentation and release metadata

- Updated README, README_TR, MODEL_CARD, BENCHMARK_RESULTS, and `final_config.lock.yaml` to use the same H1/H2 interpretation.
- Preserved research-preview status while human validation remains incomplete.
- Updated CI to regenerate the final dropout report and historical/supplementary figures.
- Release validation and checksums were rerun after regeneration.
