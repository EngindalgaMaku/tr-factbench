# TR-FactBench update notes — 2026-07-29

- Canonical encoder archive synchronized to **144 validated runs**, **432 metric rows**, **126 successful runs**, **17 class-collapse runs**, and **1 severely skewed run**.
- ModernBERT-TR added to the aligned three-seed model-family comparison.
- Four LLMs were completed under both zero-shot and fixed 8-shot prompting on DEV960 and DEV600 (16 experiment conditions total).
- Added canonical LLM tables, bilingual reports, exact model-route matrix, and LLM comparison figures.
- Dev600 blind annotator 1 completed 600/600; annotator 2 is 80/600. DEV960 annotator 1 is 288/960.
- Added the single-annotator provisional Dev600 preview and disclosed that archived DEV600 results precede three record-level review revisions (3/600; 0.5%). Runs were not repeated for this limited change.
- Updated README files, DATA_CARD, MODEL_CARD, benchmark report, model inventory, annotation status, release lock, and release validation.
- Removed unsupported causal explanations for collapsed model runs; negative results are now described as outcomes under the tested configurations.
- Sanitized active manifests to use portable repository-relative paths.
