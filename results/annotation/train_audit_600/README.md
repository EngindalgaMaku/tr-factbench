# Train audit 600 — cancelled / unused pack

This directory contains a **stratified** context-grouped quality-audit sample from the training set (150 contexts, 50 per domain, 600 examples). The input files were generated but **never annotated**.

**Status: cancelled / out of scope.** Do not start two blinds. Do not treat this as a live annotation task.

It is **not** the frozen train-4800 prefix-150 silver audit (`train_4800_prefix150_seed42_annotator_1_v1`, 2026-09-20). Prefix-150 is a random seed-42 prefix of the full Gate 1 pack, one annotator, human vs construction. Prefix-150 is not a substitute for this pack, and this pack will not be completed.

- Files are retained as unused construction artifacts.
- Do not copy prefix-150 labels here or into `data/primary/train.jsonl`.
- Full Gate 1 4,800 beyond prefix-150 is also cancelled.
