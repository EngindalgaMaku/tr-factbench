# F01 Gate 9 Combined Shortcut Pre-check — v1.0

## Text-only claim cue

- word TF-IDF accuracy: **0.325**
- word Macro-F1: **0.284**
- char Macro-F1: **0.241**
- permutation null Macro-F1 p95: **0.369**
- empirical p: **0.303**
- status: **PASS**

Word-model class recalls are supported=0.10, partial=0.20, contradicted=0.20, unverifiable=0.80. The unverifiable recall remains higher in this very small batch, so it is retained as a cumulative monitoring item even though the overall claim-only signal is below the frozen permutation threshold. We do not rewrite individual claims merely to defeat this diagnostic.

## Atom-count cue

- atom-count-only accuracy: **0.200**
- atom-count-only Macro-F1: **0.172**
- null p95: **0.336**
- empirical p: **0.522**
- fixed-count classes: **none**
- status: **PASS**

## Other QA

- exact reference-tokenizer maximum: **206 / 512 tokens**
- warning count >480: **0**
- failure count >512: **0**
- self-containment audit: **PASS, 0 unresolved openings**
- final verifier/ELECTRA: **NOT RUN**

## Gate status

Gate 9 remains a **pre-check PASS**. Formal F01 closure still requires Gate 8 re-review of the wording changed after the user's earlier approval.
