# Atom-Count Diversity and Structural Shortcut Rule v1.0.4

## Rationale

Semantic atom count is part of claim structure and can become a label shortcut if one relation class is constructed with a fixed number of propositions. In particular, `unverifiable` must not collapse to one-atom claims while other classes contain multi-atom claims.

## Frozen rule

- Atom count is **not** a label-defining property except that `partially_supported` necessarily requires at least two factual atoms.
- `supported`, `contradicted`, and `unverifiable` may each contain one or multiple factual atoms when the wording remains natural.
- `unverifiable` requires **all** of its atoms to be `not_in_context`, regardless of whether the claim has one, two, or more atoms.
- `contradicted` retains its existing rule: zero entailed atoms and at least one contradicted atom; additional atoms may be contradicted or not in context.
- `supported` retains its existing rule: every atom is entailed.
- `partially_supported` retains its existing rule: at least one entailed atom plus at least one contradicted or not-in-context atom.

## Batch-level anti-shortcut policy

For each accepted 10-group batch:

1. semantic atom count is recorded for every claim;
2. no class should be intentionally constructed with one fixed atom count;
3. for classes with 10 examples in the batch, a single atom count in 100% of the class triggers `REVIEW`;
4. the class-wise distributions should overlap rather than encode the target label;
5. exact matching of atom-count histograms is **not** required and examples must not be made awkward merely to satisfy a quota;
6. an atom-count-only group-aware diagnostic is run alongside the text-only shortcut audit;
7. if atom count alone predicts labels above its within-group permutation null, construction practice receives `REVIEW`.

The diagnostic is a construction-level guardrail. It must not use ELECTRA or final-verifier outputs.

## Atomization discipline

Atom counts are assigned using the frozen label rulebook's semantic decomposition principle: the smallest factual propositions necessary to judge evidential status. Surface conjunction count is not a substitute for semantic atom count. A conditional proposition may remain one atom when the condition and outcome jointly express one factual relation; independent factual assertions are separated.

## Change control

This rule applies prospectively to F01 revisions and all subsequent batches. Earlier accepted wording is not silently changed. Any wording revision caused by this rule requires semantic re-check and renewed constructor review before batch acceptance.
