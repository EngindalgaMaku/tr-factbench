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

The application supports Dev600, Dev960, the full train split, and the context-grouped train audit sample.

## Train audit sample

Generate or refresh the 600-example audit sample:

```bash
python scripts/41_iaa_sampling.py
```

This creates 150 context groups, 50 per domain, retaining all four claims per context. Blind input uses opaque IDs. The candidate reference file must not be shown to annotators.

After two independent blind passes:

```bash
python scripts/42_compute_iaa.py \
  --annotator1 results/annotation/train_audit_600/train_audit_600_blind_annotator_1.jsonl \
  --annotator2 results/annotation/train_audit_600/train_audit_600_blind_annotator_2.jsonl
```

The report includes pre-adjudication percent agreement, unweighted Cohen's kappa, confusion matrices, domain/class diagnostics, disagreements, and context-level bootstrap confidence intervals.

## Dev600 reporting

```bash
python scripts/51_annotation_report_dev600.py \
  --candidate-review results/annotation/dev600/dev_stress_candidate_label_review.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_1.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_2.jsonl
```

Candidate-label refinement results and human-human IAA are reported separately.
