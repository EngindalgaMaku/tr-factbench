# TR-FactBench Data Card

## Summary

TR-FactBench is a Turkish source-grounded factuality-verification dataset covering finance, legal, and medical domains. Each example contains a context passage, a question, a claim, and one of four labels: `supported`, `partially_supported`, `contradicted`, or `unverifiable`.

This is **release v1.1.0**; the data files are frozen as of v1.0.0. It succeeds the v0.1.0 research preview and adds the Gold-480 held-out evaluation records. Candidate labels were produced through controlled LLM-assisted construction and automated validation. The first independent blind-human review pass is complete for both evaluation sets: DEV960 960/960 and Dev600 600/600. A second development-set pass is a known limitation of this release, **not** remaining work required to close the resource.

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
| Dev600 blind annotator 2 | 80/600 leftover; **not required** to close this resource |
| Dev600 adjudication | not done; **not required** to close this resource |
| DEV960 candidate-label refinement | not started; **not required** to close this resource (only Dev600 ran this QC pass) |
| DEV960 blind annotator 1 | 960/960 complete; provisional preview released |
| DEV960 blind annotator 2 | 0/960 leftover; **not required** to close this resource |
| DEV960 adjudication | not done; **not required** to close this resource |
| Gold-480 dual-annotator freeze | 480/480 A and 480/480 B complete; 48 disagreements adjudicated; frozen as `data/evaluation/gold_v1.0/` |
| Train LLM pre-annotation | 4,800/4,800 complete |
| Train candidate-label review | 24/4,800 leftover; cancelled / not in scope |
| Train prefix-150 silver audit | 600/600 frozen (seed-42 prefix of `train_4800`; one annotator; 45/600 mismatch vs construction). Training labels remain silver. See `results/annotation/TRAIN_4800_PREFIX150_FREEZE.md`. |
| Train two-person quality audit (`train_audit_600`) | Pack prepared but never annotated; **cancelled / out of scope**. Distinct from prefix-150. Full Gate 1 4,800 is also cancelled. |

The authoritative progress snapshot is `annotation_status.json`.

The recommended reviewed preview files are `data/evaluation/preview_v0.1/dev_general.jsonl` and `data/evaluation/preview_v0.1/dev_stress.jsonl`. Audit copies and manifests are retained under `results/annotation/dev960/` and `results/annotation/dev600/`. The DEV960 preview changes 2/960 labels. The Dev600 preview changes 3/600 labels and includes one documented claim correction. Archived encoder and LLM scores remain tied to the original pre-review files. The canonical 144-run encoder archive and API-based LLM experiments were not repeated. A separate single-seed retraining of the selected configuration was conducted on the reviewed preview files and is archived under `runs/preview_evaluations/H1_DROP10_S45_PREVIEW` (DEV960=0.8965, DEV600=0.8522).

### Gold-480 freeze (v1.0)

`data/evaluation/gold_v1.0/` is a frozen 480-example dual-annotator gold set (120 context groups). Annotators A and B each labelled 480/480 independently. Raw agreement is 432/480 (90.00%, Cohen's κ = 0.8666). The 48 disagreements were adjudicated; the canonical file is `gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl`. Raw annotator files, agreement tables, adjudication log, and provenance sit in the same package.

### Train prefix-150 silver audit (frozen 2026-09-20)

A predeclared freeze of the first 150 context groups of the seed-42 shuffle of the full `train_4800` blind pack:

- 150 groups / 600 examples (12.5% of train)
- one annotator, `blind_human`, candidate labels hidden
- 555/600 agree with construction (92.50%); 45/600 mismatch (7.50%; Wilson 95% CI 5.65–9.89%)
- **not** gold; `data/primary/train.jsonl` is unchanged
- **not** IAA (single annotator; Cohen's κ is not computed)
- **not** `train_audit_600`

Public summary: `results/annotation/TRAIN_4800_PREFIX150_FREEZE.md`. Gold-480 IAA is evaluation-gold evidence and must not be cited as evidence about this training audit.

### Train audit sample (`train_audit_600`, cancelled)

`results/annotation/train_audit_600/` is an unused construction artifact: a stratified 150-context / 600-example sample that was prepared but never annotated. It is **cancelled / out of scope**. Prefix-150 is not a substitute, and this pack will not be completed. Full Gate 1 4,800 (beyond the frozen prefix) is also cancelled. Training labels remain construction-time (silver).

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
6. **Independent blind validation:** Annotator 1 completed all 600 Dev600 examples and all 960 DEV960 examples without seeing candidate labels or LLM suggestions. Dev600 annotator 2 is 80/600 leftover; DEV960 annotator 2 was not started. That second pass is a known limitation of this frozen release, not remaining work. Formal human-human Cohen's κ on the development sets is therefore not reported.

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

1. Both evaluation sets have one complete blind-human review pass. A second pass and adjudication are **not** part of this frozen release (Dev600 A2 is 80/600 leftover; DEV960 A2 was not started).
2. Training labels remain construction-time (silver). A 600-example prefix of `train_4800` is frozen as a single-annotator noise estimate (45/600 mismatch). Full Gate 1 4,800 and the stratified two-person `train_audit_600` pack are cancelled / out of scope.
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
