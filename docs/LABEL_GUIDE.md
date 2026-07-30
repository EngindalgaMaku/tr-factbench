# TR-FactBench Label Guide

> Guide version: 1.0  
> Label schema: TR-FactBench four-label schema v1  
> Last updated: 2026-07-27  
> Status: Research preview

This document is the canonical English reference for the four label classes:
their definitions, decision rules, and boundary examples.
README.md, DATA_CARD.md, annotation/README.md and the annotation interface
all link to this document. They may include short summaries for quick reference,
but this document is canonical for detailed decision rules and boundary examples.

---

## 1. Core rule

**During annotation, use only the provided context.**

Whether the claim is true or false in the real world does not determine the label unless
it is supported or refuted by the context. Annotators must not use their own domain
knowledge, web searches, or any external source as evidence.

This rule is especially important for legal, financial, and medical examples.

---

## 2. Label definitions (short)

| Label | Short definition |
|---|---|
| `supported` | All atomic propositions in the claim are supported by the context. |
| `partially_supported` | A multi-atom claim has at least one supported atom and at least one unsupported or contradicted atom. |
| `contradicted` | None of the claim's atomic propositions are supported by the context, and at least one atomic proposition explicitly conflicts with information in the context. |
| `unverifiable` | All atomic propositions in the claim are absent from or underdetermined by the context; no explicit contradiction exists. |

---

## 3. Atomic proposition approach

Break the claim into independent verifiable propositions (atoms) before assigning a label.

**Example:**

> "Limited review applies to interim financial statements. The special audit report
> takes effect after general assembly approval."

This contains two atoms:
1. Limited review applies to interim financial statements.
2. The special audit report takes effect after general assembly approval.

For each atom determine one of the following relationships with the context:

| Atom relationship | Meaning |
|---|---|
| `entailed` | The context supports this atom. |
| `contradicted` | The context explicitly states the opposite of this atom. |
| `not_in_context` | The context is insufficient to decide about this atom. |

The final label is derived from atom relationships as follows:

| Atom relationships | Final label |
|---|---|
| All atoms `entailed` | `supported` |
| At least one `entailed`, at least one `contradicted` or `not_in_context` | `partially_supported` |
| No `entailed`, at least one `contradicted` (remaining atoms may be `not_in_context`) | `contradicted` |
| All atoms `not_in_context` | `unverifiable` |

> **Note:** A `contradicted` decision does not require all atoms to conflict.
> It is sufficient that no atom is supported and at least one atom is explicitly contradicted;
> the remaining atoms may be `not_in_context`.

---

## 4. Distinguishing `contradicted` from `unverifiable`

This is the most frequently confused boundary.

### `contradicted`

The context must provide **explicit** information that opposes the claim.
No atom in the claim may be supported.

**Context:**
> Buy orders with higher prices have priority; sell orders with lower prices have priority.
> At the same price level, time priority applies.

**Claim:**
> Orders are matched regardless of arrival time and price.

**Decision:** `contradicted`

The context explicitly states that price and time priority apply; the claim asserts the opposite.

---

### `unverifiable`

The context **neither supports nor explicitly refutes** the claim.

**Context:**
> Contacts are examined and subjected to the necessary tests.

**Claim:**
> Contact examination definitively determines whether a person will be infectious in the future.

**Decision:** `unverifiable`

The context does not say definitiveness is achieved, but it also does not say it is
impossible or not done.

---

### Key decision question

> Does the context **explicitly** state the opposite of the claim, or does it simply
> not mention it?

- Explicit contradicting information, no supported atom → `contradicted`
- Simply absent information → `unverifiable`

---

## 5. Modal expressions: the `contradicted` / `unverifiable` boundary

The following words require careful attention:

`only` · `always` · `never` · `certainly` · `completely` · `all` · `necessarily` · `solely` · `at least` · `at most` · `must`

**Core rule:**

If the context only states that an action is **possible** but provides no information about
whether it is **required**, a claim asserting it "must" happen is generally **`unverifiable`**.

If the context explicitly states that an action is optional, applies only under certain
conditions, or may not apply, a claim asserting it "must always" happen is **`contradicted`**.

**Example pair:**

| Context | Claim | Decision |
|---|---|---|
| "Preventive treatment may be given." | "Preventive treatment must be given." | `unverifiable` |
| "Preventive treatment is given only based on risk assessment and does not apply to every contact." | "Preventive treatment must be given to all contacts." | `contradicted` |

---

## 6. The `partially_supported` boundary

This label can **only** be used when a claim contains more than one independent verifiable atom.

**Context:**
> Limited independent review is conducted for interim financial statements. Special audit
> may apply in cases such as public offering, merger, or demerger.

**Claim:**
> Limited review applies to interim statements. The special audit report takes effect
> after general assembly approval.

**Decision:** `partially_supported`

- First atom (limited review for interim…) is supported.
- Second atom (special audit report after general assembly…) is absent from context.

> **Important:** A single-atom claim cannot be `partially_supported`. Mere lexical overlap
> of part of a single atom is not sufficient; the atom must be evaluated as a whole.

---

### The supported atom must be meaningful

If the **main proposition** of a claim is unsupported but only a **peripheral detail** is
supported, this does not automatically yield `partially_supported`. The supported atom
must be independent, verifiable, and a genuine part of the claim's substantive content.

**Context:**
> Locke distinguished legislative, executive, and federative powers.

**Claim:**
> Locke argued, while living in Paris, that the legislative power cannot be removed by election.

The overlap of the subject "Locke" or the general topic "powers" does not count as a
supported atom. The claim is likely `unverifiable`.

---

### What if the supported part is very small?

**Claim:**
> Locke addressed federative power and argued that it covers only the authority to collect taxes.

Atoms:
1. Locke addressed federative power. → `entailed`
2. Federative power covers only the authority to collect taxes. → `contradicted`

**Decision:** `partially_supported`

Even if the first atom is brief, it constitutes an independent and meaningful fact.

---

## 7. Minimal pair examples

### Example A — `unverifiable`

**Context:**
> BCG vaccine provides protection against severe forms of tuberculosis in children.

**Claim:**
> The BCG scar indicates the duration of protective treatment.

The context does not mention the scar. Information is simply absent.

---

### Example B — `contradicted`

**Context:**
> BCG vaccine does not completely prevent all types of tuberculosis or transmission at every age.

**Claim:**
> BCG vaccine completely prevents all types of tuberculosis and transmission at every age.

The context explicitly states the opposite of the claim.

---

## 8. Effect of the question on labelling

The question may help interpret the context and claim, but **the label is determined
solely by the claim–context relationship.**

A claim that does not fully answer the question is not `unverifiable` or `contradicted`
for that reason alone.

---

## 9. Ambiguous and defective examples

If the meaning of a claim cannot be reliably determined, if the context is internally
inconsistent, if a grammatical defect changes the meaning, or if the labelling decision
rests on a single strong interpretation only, the annotator should not force a label.

Such examples should be flagged as `needs_adjudication` with a brief explanation.

`needs_adjudication` is **not** part of the four-class label schema; it is a quality
control status.

---

## 10. Decision tree

```
1. Decompose the claim into atomic propositions.

2. Is every atom supported by the context?
   └─ All atoms entailed → supported

3. Is at least one atom supported?
   └─ Yes, but another atom is absent or contradicted → partially_supported

4. No supported atoms — does the context explicitly contradict any atom?
   └─ Yes (remaining atoms may be not_in_context) → contradicted

5. No supported or explicitly contradicted atoms
   └─ unverifiable

6. Decision cannot be made reliably
   └─ needs_adjudication (quality control status)
```

---

## 11. Canonical source policy

This file (`docs/LABEL_GUIDE.md`) is the **single canonical source** for English label definitions.
The following documents link here. They may include short label summaries for quick reference,
but this document is canonical for detailed decision rules and boundary examples.

- `README.md`
- `DATA_CARD.md`
- `annotation/README.md`
- Streamlit annotation interface (short rules displayed in the interface are kept in sync with this guide)
- LLM pre-annotation prompt (label definitions and decision rules in the prompt are kept semantically
  aligned with this guide; changes to this guide are reflected in the prompt version)

Turkish canonical source: [`docs/LABEL_GUIDE_TR.md`](LABEL_GUIDE_TR.md)