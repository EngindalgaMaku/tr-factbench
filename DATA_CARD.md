# TR-FactBench Data Card

## Summary

TR-FactBench is a Turkish source-grounded factuality-verification dataset covering finance, legal, and medical domains. Each example contains a context passage, a question, a claim, and one of four labels: `supported`, `partially_supported`, `contradicted`, or `unverifiable`.

This is a **v0.1.0 research preview**. Candidate labels were produced through controlled LLM-assisted construction and automated validation. The first independent blind-human review pass is complete for both evaluation sets: DEV960 960/960 and Dev600 600/600. A second annotation pass and adjudication are planned for a future release.

## Primary data

| Split | File | Examples | Context groups | Domain balance |
|---|---|---:|---:|---|
| Train | `data/primary/train.jsonl` | 4,800 | 1,200 | 1,600 per domain |
| General development — archived experiment version | `data/evaluation/legacy_experiment_version/dev_general.jsonl` | 960 | 240 | 320 per domain |
| Stress development — archived experiment version | `data/evaluation/legacy_experiment_version/dev_stress.jsonl` | 600 | 150 | 200 per domain |
| **Total** |  | **6,360** | **1,590** |  |

Every context group contains exactly four examples, one per label, in the archived experimental files. New preview evaluations should use `data/evaluation/preview_v0.1/dev_general.jsonl` and `data/evaluation/preview_v0.1/dev_stress.jsonl`. Human corrections produce only minor class-count deviations from exact balance. Files under `data/derived/` are deterministic views of the archived primary data and do not add unique examples.

## Labels

| Label | Operational definition |
|---|---|
| `supported` | All material propositions in the claim are supported by the context. |
| `partially_supported` | The claim contains multiple propositions; at least one is supported and at least one is absent or contradicted. |
| `contradicted` | No atomic proposition is supported and at least one explicitly conflicts with the context. |
| `unverifiable` | All atomic propositions are absent or underdetermined and no explicit contradiction exists. |

## Schema

The authoritative schema is `data/schema.json`. The primary JSONL files contain the core example fields. `data/example_metadata.csv` describes the archived experiment version without modifying its files or hashes. Preview evaluation records include their review metadata directly; Dev600 legacy-to-preview IDs are mapped in `data/manifests/dev600_legacy_to_preview_id_map.csv`.

The field name `gold_label` is retained for pipeline compatibility. In the preview evaluation files it represents the completed first blind-human pass and is marked `single_annotator_provisional`; training labels remain construction-time candidate labels.

## Annotation status

| Stage | Current status |
|---|---|
| Dev600 candidate-label refinement | 600/600 complete; candidate and LLM suggestion were intentionally visible |
| Dev600 blind annotator 1 | 600/600 complete; candidate and LLM outputs hidden |
| Dev600 blind annotator 2 | 80/600 complete; candidate and LLM outputs hidden |
| Dev600 adjudication | not started |
| DEV960 blind annotator 1 | 960/960 complete; provisional preview released |
| Train LLM pre-annotation | 4,800/4,800 complete |
| Train candidate-label review | 24/4,800 complete |
| Train two-person quality audit | 600-example sample prepared; both blind passes not started |

The authoritative progress snapshot is `annotation_status.json`.

The recommended reviewed preview files are `data/evaluation/preview_v0.1/dev_general.jsonl` and `data/evaluation/preview_v0.1/dev_stress.jsonl`. Audit copies and manifests are retained under `results/annotation/dev960/` and `results/annotation/dev600/`. The DEV960 preview changes 2/960 labels. The Dev600 preview changes 3/600 labels and includes one documented claim correction. Archived encoder and LLM scores remain tied to the original pre-review files. The canonical 144-run encoder archive and API-based LLM experiments were not repeated. A separate single-seed retraining of the selected configuration was conducted on the reviewed preview files and is archived under `runs/preview_evaluations/H1_DROP10_S45_PREVIEW` (DEV960=0.8965, DEV600=0.8522).

### Train audit sample

`results/annotation/train_audit_600/` contains a stratified, context-grouped quality-audit sample:

- 150 context groups
- 50 contexts per domain
- all four claims retained for each selected context
- 600 examples total
- context-length stratification within each domain
- opaque IDs in the blind input
- candidate labels stored in a separate reference file that must not be shown to annotators

After two independent blind-human passes, `scripts/42_compute_iaa.py` computes pre-adjudication percent agreement, unweighted Cohen's κ, class- and domain-level diagnostics, and context-level bootstrap confidence intervals. Agreement must be reported before adjudication.

## Provenance

`data/sources_manifest.csv` contains one row for each of the 1,590 context groups.

- Stress-set contexts include source URLs and retrieval metadata.
- Train and general-development contexts were constructed by rewriting, simplifying, and controllably adapting publicly available Turkish domain materials. They are marked `provenance_status=source_adapted_no_item_url`; item-level URLs were not retained for every transformed context.
- Labels represent the relationship between a claim and the supplied context, not an independent judgment of real-world factual correctness.

Item-level URL absence in the source-adapted splits is not treated as an annotation-validity failure. Applicable source terms still need to be reviewed before any rights-cleared commercial redistribution claim. See `data/NOTICE.md` and `data/DATA_LICENSE.md`.

## Construction methodology

1. **Context preparation:** Publicly available Turkish domain materials were rewritten, simplified, and controllably adapted into self-contained context groups.
2. **Claim construction:** Each context group was paired with four candidate claims designed to represent the four factuality relations.
3. **LLM-assisted drafting:** Language models were used as construction aids rather than independent gold annotators.
4. **Automated validation:** Scripts checked JSONL structure, label and domain balance, context-group completeness, identifier uniqueness, split overlap, and deterministic regeneration of derived views.
5. **Candidate-label refinement:** For Dev600, a human reviewer compared the current candidate label with a normalized LLM analysis to identify apparent construction errors quickly. This stage was not used as human-human IAA evidence.
6. **Independent blind validation:** Annotator 1 completed all 600 Dev600 examples and all 960 DEV960 examples without seeing candidate labels or LLM suggestions. Dev600 annotator 2 completed 80/600. Agreement statistics will be reported after paired blind annotation files are complete and before adjudication.

## LLM annotation artifacts

Current normalized LLM artifacts are stored as `llm_preannotations_normalized.jsonl`. The legacy filename `llm_preannotations.jsonl` is retained as a compatibility alias. The historical raw artifact was not included in this snapshot; this limitation is documented in each `llm_postprocess_manifest.json`. Future runs write raw artifacts separately before deterministic normalization.

## LLM benchmark baselines

Four LLMs were evaluated in zero-shot and fixed 8-shot conditions on both development sets. Exact model IDs, prompt hashes, coverage, and strict metrics are documented in `reports/LLM_BASELINES.md` and `results/llm_baselines/experiment_matrix.csv`.

## Context-dependence diagnostic

An inference-only diagnostic was conducted on the human-reviewed preview evaluation files using one fixed trained ELECTRA checkpoint (see `runs/ablations/context_ablation_manifest.json` for provenance). The model was evaluated with the full input, with the context removed (original question and claim retained), and with the context replaced by a shuffled context while preserving the original question and claim.

| Condition | DEV960-GEN Macro-F1 | DEV600-STR Macro-F1 |
|---|---:|---:|
| **Full input** | **0.8994** | **0.8686** |
| No context: question + claim | 0.4043 | 0.3858 |
| Shuffled context: question + claim | 0.3740 | 0.3508 |

The large performance reductions when context is removed or replaced provide evidence that the task and model predictions depend materially on the supplied passage–claim relation. They do not show that all lexical artifacts have been eliminated, but they indicate that context-free question–claim cues are insufficient to explain the full-input result. Full definitions and machine-readable values are in `results/tables/table_J_context_dependence_ablation.csv` and `results/tables/table_J_context_dependence_ablation.md`.

## Intended uses

- Turkish factuality-verification research
- Source-grounded hallucination detection
- Encoder and PEFT benchmarking
- Cross-domain transfer and robustness analysis

## Out-of-scope uses

- Autonomous medical, legal, or financial decision-making
- Production fact checking without human review
- Commercial redistribution of adapted contexts without reviewing applicable source terms

## Known limitations

1. Both evaluation sets have one complete blind-human review pass; the second pass and adjudication are planned for a future release.
2. Dev600 annotator 2 is 80/600; DEV960 annotator 2 has not started. The train audit has also not started.
3. Train and general-development contexts are source-adapted synthetic examples; item-level URLs were not retained for every context.
4. The dataset is Turkish-only and covers three domains.
5. The stress set is deliberately harder and is not distribution-identical to the general development set.
6. The four claims associated with a context are structurally balanced; this should be considered when designing splits and evaluation protocols.
7. The current development splits have been used during model selection and are not untouched final test sets.

## Licensing

- Project-created claims, questions, labels, and metadata: CC BY 4.0
- Context passages: source-dependent rights
- Code: MIT

See `data/DATA_LICENSE.md`, `data/NOTICE.md`, and `LICENSE`.

## Citation

See `CITATION.cff`.
