# Batch Workflow v0.1

## Batch size and IDs

Each batch contains 10 context groups and therefore 40 claim candidates.

Examples:

- `F01-001` ... `F01-010`
- `L01-001` ... `L01-010`
- `M01-001` ... `M01-010`

Each claim ID appends its intended label, for example `F01-001-supported`.

## Gate 0 — Protocol version

Record the exact protocol version used by the batch. No batch may be accepted under an unspecified protocol version.

## Gate 1 — Source and provenance

For each group:

- collect public URL, title, organization, retrieval time;
- save raw or faithful source snapshot;
- compute source/snapshot hash;
- run exact URL/text and near-duplicate screen against train, DEV960, DEV600, and previously accepted final-test groups;
- resolve every REVIEW flag before context selection.

## Gate 2 — Context selection

The selected context must:

- be understandable without hidden document sections;
- contain sufficient factual content for meaningful relation construction;
- avoid navigation/cookie/boilerplate text;
- preserve wording from the source rather than being model-rewritten;
- have exact source offsets or a reproducible extraction record.

Freeze the context before the question and claims are finalized.

## Gate 3 — Question freeze

Draft one natural Turkish question that is relevant to the context. Review it without reference to target labels. Once approved, freeze it for all four claims in the group.

Questions must not be modified to make a particular relation easier to construct after label-conditioned drafting begins.

## Gate 4 — Source atom map

Record the context facts that are most relevant to the question. Atom maps are construction metadata, not evaluation input.

## Gate 5 — Four candidate claims

Construct one candidate for each intended label using the rulebook and the variation taxonomy. Store atomic relation decomposition and evidence spans for every candidate.

## Gate 6 — Automated structural/rule QA

Run `scripts/validate_construction_batch.py`. Any rule failure blocks acceptance.

## Gate 7 — Blind semantic re-check

Create a view containing only context, question, and claim. Reclassify without exposing the construction target. A target/re-check disagreement becomes REVIEW and cannot be silently overridden.

This is an internal consistency check, not a substitute for independent human annotation.

## Gate 8 — Human batch review

Review all 10 groups together. Reject awkward, trivial, ambiguous, templated, or weak candidates even if they pass structural validation. Every edit/replacement must be entered in `change_log.csv`; rejected attempts go to `rejection_log.csv`.

## Gate 9 — Bias audit

Run `scripts/audit_construction_bias.py` on all accepted batches to date. Inspect class differences in:

- claim length,
- sentence count,
- digit rate,
- negation/modality markers,
- conjunction markers,
- context–claim lexical similarity,
- contradiction subtype,
- unverifiable subtype,
- partial relation subtype,
- claim-only linear classification signal.

The bias audit is a dataset-level diagnostic. It must not be used to tune examples against the final verifier.

## Gate 10 — Batch acceptance

A batch is accepted only when all unresolved REVIEW items are closed and QA artifacts are stored with the batch.
