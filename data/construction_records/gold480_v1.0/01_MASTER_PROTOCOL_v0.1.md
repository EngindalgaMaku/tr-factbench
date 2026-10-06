# TR-FactBench Untouched Controlled Test — Master Construction Protocol v0.1

## 1. Objective

Construct a new, source-disjoint final holdout for four-way context–claim verification in Turkish. The test is designed to measure whether a verifier can distinguish `supported`, `partially_supported`, `contradicted`, and `unverifiable` relations under controlled evidence conditions.

The test is **not** intended to estimate the natural prevalence of hallucination categories in deployed RAG systems. A separate naturalistic transfer study may be reported independently.

## 2. Unit of construction

A context group contains:

1. one public Turkish source record,
2. one fixed context passage,
3. one fixed natural question,
4. four claim candidates targeting the four relation labels.

The context and question are identical across all four claims in the group.

## 3. Target size

- Finance: 40 context groups / 160 examples.
- Legal: 40 context groups / 160 examples.
- Medical: 40 context groups / 160 examples.
- Total: 120 context groups / 480 examples.
- Intended construction balance: 120 candidates per relation label.

The final count may be lower only when quality control rejects examples and replacements are not yet available. No low-quality example is retained merely to satisfy a quota.

## 4. Construction sequence

The mandatory order is:

`public source -> provenance capture -> leakage screen -> context freeze -> question freeze -> source atom map -> four label-targeted claim candidates -> automated rule validation -> blind semantic re-check -> human batch review -> dataset-level bias audit -> independent blind annotation -> agreement analysis -> adjudication -> release freeze/hash -> one-time verifier evaluation`

Changing this order in a way that exposes verifier behavior before freeze is prohibited.

## 5. Construction labels are not gold

`intended_label` records the relation targeted during candidate construction. It is never treated as final gold solely because the candidate was drafted for that class.

Final gold is determined only after construction through blind human annotation. Annotators do not see intended labels, construction taxonomies, source atoms, evidence maps, model identities, or verifier outputs.

If independent annotation disagrees with the construction target, the human annotation/adjudication result takes precedence or the example is rejected.

## 6. Ten-group batch discipline

Construction proceeds in batches of 10 context groups / 40 claims. A batch must complete all required QA stages before the next batch of the same domain is considered accepted.

The batch structure is designed to make drift visible early and to support deliberate review rather than bulk generation.

## 7. Model-assisted drafting

A language model may assist with:

- extracting candidate atomic facts from the frozen context,
- drafting natural questions before label construction,
- drafting relation-conditioned claim candidates,
- proposing evidence spans and relation decompositions,
- rewriting a rejected candidate when explicitly instructed.

The model is a construction assistant, not a gold annotator. Structured audit fields and review decisions are retained.

## 8. Prohibited construction behavior

The following are not allowed:

- generating synthetic contexts and presenting them as source-derived;
- changing a context or claim because the final verifier predicts it incorrectly;
- using verifier confidence to select, reject, rewrite, or adjudicate examples;
- constructing the question after seeing which label is difficult to produce;
- treating a target label as gold without independent review;
- keeping known duplicate or near-duplicate source material because the URL differs;
- copying supported claims verbatim from the context as the default construction strategy;
- creating unverifiable claims by adding obviously unrelated facts;
- creating contradicted claims only through trivial number swaps;
- creating partial claims through a single repeated surface template;
- silently editing previously accepted batches after a protocol rule changes.

## 9. Untouched status

The final test becomes untouched with respect to the verifier only after:

1. all final examples are adjudicated,
2. the release JSONL is generated,
3. a SHA-256 manifest is written,
4. the evaluation checkpoint/configuration is declared,
5. no subsequent construction decision depends on verifier output.

After freeze, example wording, selection, context, labels, and adjudication cannot be changed because of model results.

## 10. Separate naturalistic transfer study

If a naturalistic RAG transfer set is later collected, it must be reported separately. Its observed label prevalence must not be conflated with the balanced controlled-test composition.
