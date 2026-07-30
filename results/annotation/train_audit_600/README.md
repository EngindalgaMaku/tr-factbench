# Train audit 600

This directory contains a context-grouped quality-audit sample from the training set. It includes 150 contexts (50 per domain) and all four claims from each context, for 600 examples.

- Use `train_audit_600_blind_input.jsonl` in blind-human mode.
- Do not show `train_audit_600_candidate_reference.jsonl` to annotators.
- Produce two independent blind annotation files, then run `scripts/42_compute_iaa.py`.
- Cohen's kappa is computed before adjudication.
