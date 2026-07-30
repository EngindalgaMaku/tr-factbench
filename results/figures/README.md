# Figure index

The figure directories are separated by experimental role so that values from different stages are not mistaken for one another.

## `en/` and `tr/` — canonical figures

- `fig_01_model_comparison_bar.png`: aligned `F1_*` model-family screening, three-seed means. This is not the final H1/H2 comparison.
- `fig_02_rank_scaling_line.png`: historical BERT-32k rank screening.
- `fig_03_*` and `fig_04_*`: class-level results from the aligned F1 screening stage.
- `fig_05_stress_vs_general_scatter.png`: general/stress relationship in the F1 screening stage.
- `fig_06_confusion_matrix_selected_dev960.png`: selected checkpoint `H1_DROP10_S45` on DEV960.
- `fig_06_confusion_matrix_selected_dev600.png`: selected checkpoint `H1_DROP10_S45` on DEV600.
- `fig_07_cross_domain_heatmap.png`: three-seed cross-domain analysis.
- `fig_08_final_dropout_comparison.png`: directly comparable seed-fixed H1/dropout=0.10 and H2/dropout=0.00 five-seed means and standard deviations.
- `fig_09_final_seed_stability.png`: per-seed H1/H2 performance.
- `fig_10_llm_baselines_dev960.png`: four LLMs under zero-shot and fixed 8-shot prompting on DEV960, with the selected ELECTRA checkpoint as reference.
- `fig_10_llm_baselines_dev600.png`: corresponding DEV600 comparison; values use the pre-review evaluation file.

`fig_06_confusion_matrix_best.png` is retained only as a compatibility alias of the selected DEV960 checkpoint figure.

## `historical/`

Historical multi-seed and collapse diagnostics. They are retained for auditability but are not used as the current final H1/H2 comparison.

The legacy G1/D1 collapse observations are real archived outcomes. The H2 five-seed rerun shows that the dropout=0.00 seed-46 collapse did not recur under the seed-fixed current pipeline.

## `supplementary/`

Exploratory rank, data-composition, hyperparameter, error-analysis, and annotation-distribution figures.
