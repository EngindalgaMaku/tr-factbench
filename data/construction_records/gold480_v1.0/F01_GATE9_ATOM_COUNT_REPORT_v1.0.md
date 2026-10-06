# F01 Gate 9 Atom-Count Shortcut Audit — v1.0

## Decision

**PASS (construction pre-check).** Atom count is no longer fixed by relation class in F01. No ELECTRA or final-verifier output was used.

## Class-wise semantic atom-count distribution

| Label | 1 atom | 2 atoms | 3 atoms |
|---|---:|---:|---:|
| supported | 0 | 8 | 2 |
| partially_supported | 0 | 8 | 2 |
| contradicted | 2 | 7 | 1 |
| unverifiable | 3 | 5 | 2 |

The `partially_supported` class cannot contain one-atom claims by definition. The other classes are not assigned an atom count by label. In particular, `unverifiable` now deliberately contains one-, two-, and three-atom claims, with every atom remaining `not_in_context`.

## Atom-count-only group-aware diagnostic

- accuracy: **0.200**
- Macro-F1: **0.172**
- permutation null Macro-F1 p95: **0.336**
- empirical p: **0.522**
- fixed-count classes: **none**

Atom count alone therefore does not separate the four classes above the within-group permutation null in this 10-group batch.

## Interpretation

This is an anti-shortcut construction diagnostic, not a statistical claim about the final 480-example test. It is rerun cumulatively as later batches are added. Exact histogram matching is not required; semantic naturalness takes precedence over quotas.
