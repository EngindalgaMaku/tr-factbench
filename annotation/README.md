# Annotation workflows

TR-FactBench separates two methodologically different workflows.

## 1. Candidate-label refinement

This is a quality-control stage, not independent annotation. The interface intentionally shows:

- the current candidate label,
- the normalized LLM suggestion,
- the LLM rationale and atomic relation analysis.

A human reviewer accepts, corrects, or flags the candidate. The resulting file is useful for locating construction errors quickly, but it must not be counted as a second human annotator or used as human-human IAA evidence.

## 2. Independent blind-human annotation

Blind mode shows only the context, question, claim, and label definitions. It hides:

- candidate labels,
- LLM suggestions,
- internal example IDs,
- the other annotator's decisions.

True two-human IAA requires two different people to complete blind files independently. Cohen's kappa is computed before adjudication.

## Installation

```bash
pip install -r requirements-annotation.txt
```

## Raw and normalized LLM artifacts

New OpenRouter runs write raw output to:

```text
results/annotation/dev600/llm_preannotations_raw.jsonl
```

Then run:

```bash
python scripts/_llm_postprocess.py
```

This produces:

```text
llm_preannotations_normalized.jsonl
llm_postprocess_manifest.json
```

The legacy `llm_preannotations.jsonl` path is retained as a compatibility alias. In the current snapshot, historical raw files were not preserved; this is recorded in the postprocess manifests.

## OpenRouter pre-annotation

```bash
python scripts/50_openrouter_preannotate_dev600.py --dry-run --limit-contexts 1
python scripts/50_openrouter_preannotate_dev600.py --limit-contexts 5
python scripts/50_openrouter_preannotate_dev600.py
python scripts/_llm_postprocess.py
```

The model never receives `gold_label`, real example IDs, or the original within-context claim order.

## Review application

```bash
streamlit run annotation/review_app.py
```

The application supports Dev600 and Dev960. Full train Gate 1 (`train_4800`), leftover train candidate-label review, and `train_audit_600` are cancelled / out of scope; those presets are not offered.

## Train-4800 prefix-150 freeze (silver, 2026-09-20)

The first 150 context groups of the seed-42 shuffle of `results/annotation/train_4800/` are frozen:

- 600 examples, one annotator, `blind_human`
- 45/600 mismatch vs construction-time labels (7.50%; Wilson 95% CI 5.65–9.89%)
- **not** gold; do not write these labels into `data/primary/train.jsonl`
- **not** IAA (single annotator)
- **not** `train_audit_600`

Public summary: `results/annotation/TRAIN_4800_PREFIX150_FREEZE.md`. Gold-480 IAA is evaluation-gold evidence, not evidence about this training audit.

## Cancelled train packs (not in scope)

The following were prepared as files but will **not** be annotated:

- full `train_4800` Gate 1 (Reviewer A/B + adjudication beyond the frozen prefix-150)
- leftover 24/4,800 train candidate-label review
- stratified two-person `train_audit_600`

`results/annotation/train_audit_600/` is an unused construction artifact. Prefix-150 is not a substitute for that pack, and that pack will not be completed. Do not gold-relabel `data/primary/train.jsonl`.

## Dev600 reporting

```bash
python scripts/51_annotation_report_dev600.py \
  --candidate-review results/annotation/dev600/dev_stress_candidate_label_review.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_1.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_2.jsonl
```

Candidate-label refinement results and human-human IAA are reported separately.
