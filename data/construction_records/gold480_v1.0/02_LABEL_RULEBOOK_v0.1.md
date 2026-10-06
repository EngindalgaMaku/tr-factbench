# Four-Way Label Rulebook v0.1

Labels describe the relation between the supplied **context** and **claim**. Broad world truth is not the labeling target.

## Atomic decomposition

A claim is decomposed into the smallest factual propositions needed to determine its evidential status. Each atom receives one relation:

- `entailed`: the context supports the proposition.
- `contradicted`: the context explicitly supports an incompatible proposition.
- `not_in_context`: the supplied context is insufficient to support or contradict the proposition.

The question may disambiguate the intended referent, but the label is fundamentally a context–claim relation.

## supported

A claim is `supported` when **all** relevant factual atoms are entailed by the context.

Rule:

`all atoms = entailed`

A supported candidate should normally paraphrase rather than copy the evidence verbatim. Numbers, entities, modality, scope, negation, and temporal conditions must remain faithful to the context.

## partially_supported

A claim is `partially_supported` when it contains at least two factual atoms, with:

- at least one `entailed` atom, and
- at least one `contradicted` or `not_in_context` atom.

Rule:

`atom_count >= 2 AND entailed >= 1 AND (contradicted + not_in_context) >= 1`

A single-atom claim cannot be partially supported.

## contradicted

A claim is `contradicted` when:

- no factual atom is entailed, and
- at least one factual atom is explicitly contradicted by the context.

Any remaining atoms may be `not_in_context`.

Rule:

`entailed = 0 AND contradicted >= 1`

This prevents a mixed claim containing one true and one false proposition from being mislabeled as contradicted; such a claim is partially supported.

## unverifiable

A claim is `unverifiable` when all factual atoms are outside the supplied context and none is explicitly contradicted.

Rule:

`all atoms = not_in_context`

The claim must remain topically plausible and relevant to the question/context. Unrelated trivia is not an acceptable unverifiable construction.

## Evidence-span policy

- Every `entailed` atom must point to at least one supporting context span.
- Every `contradicted` atom must point to at least one context span establishing the incompatible fact.
- `not_in_context` atoms must not contain fabricated evidence spans.

Evidence spans are construction/audit metadata; they are not shown to blind annotators or the evaluated verifier.
