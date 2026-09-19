# TR-FactBench Pre-Annotation Release Guardrail v1.1

This checklist is a hard sequencing guardrail. Downstream work may not bypass an open BLOCK item.

- [x] Controlled construction 120 groups / 480 examples: DONE
- [x] Historical F01/F02 kept immutable: DONE
- [x] F01/F02 deterministic entity-safe release derivative created: DONE
- [x] Canonical merged 480 evaluation surface created: DONE
- [x] 480 quartet/ID/domain/duplicate checks: PASS
- [x] Cross-group near-duplicate review: DONE
- [x] Release-wide holistic automated audit: DONE; residual claim-only lexical signal retained as documented REVIEW/limitation
- [x] Release-wide human-readable read of all 480 claims: DONE
- [x] Changed-surface Gate 6 / Gate 7 recheck: PASS
- [x] Renewed human review of all changed evaluation surfaces: **APPROVED**
- [x] Exact pre-annotation text freeze + SHA-256: **DONE** `a134bdecb40ca2c8340dca2402b62940793d63e7ff9428f7b3234ea1e81c1b73`
- [x] Blind annotator views generated from frozen surface: **DONE**
- [x] Annotator eligibility/blinding rule frozen: **DONE**
- [ ] Independent annotation pass 1 by eligible blind human: BLOCK - NOT STARTED
- [ ] Independent annotation pass 2 by eligible blind human: BLOCK - NOT STARTED
- [ ] Agreement analysis before adjudication: BLOCK - NOT STARTED
- [ ] Adjudication: BLOCK - NOT STARTED
- [ ] Gold release freeze/hash: BLOCK - NOT DONE
- [ ] One-time final verifier evaluation: BLOCK - PROHIBITED UNTIL GOLD FREEZE

If any context/question/claim wording changes now, the pre-annotation freeze is invalidated and the changed-surface semantic/holistic QA plus renewed human review must be repeated before annotation.
