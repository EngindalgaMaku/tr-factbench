# Train-4800 prefix-150 freeze (public summary)

**Status: FROZEN** (2026-09-20). Full hashes and the blind JSONL live in the local pack `results/annotation/train_4800/` (session files are not published with this preview).

A predeclared freeze of the first **150 context groups** (600 examples, 12.5% of train) from the seed-42 shuffle of the full `train_4800` blind pack. One annotator, blind to construction labels. Comparison is human vs construction-time candidate label.

| | |
|---|---|
| Agree | 555 / 600 (92.50%) |
| Mismatch | 45 / 600 (7.50%); Wilson 95% CI 5.65–9.89% |
| Protocol | `blind_human`, annotator_1, order seed 42 |
| IAA / Cohen's κ | **not computed** (single annotator) |
| Gold train | **no** — `data/primary/train.jsonl` is unchanged |
| `train_audit_600` | **not this sample**; that two-person pack is cancelled / unused |

This is a silver-train noise estimate. It is not Gate 1 complete (4,800/4,800), not Gold-480, and not human-human IAA. Gold-480 IAA (90.0%, κ = 0.8666) is an evaluation-gold result and must not be cited as evidence about this training audit.

See `DATA_CARD.md` and `annotation_status.json`.
