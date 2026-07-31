# Table J — Context-dependence ablation (inference-only)

The same trained ELECTRA checkpoint was evaluated on the human-reviewed preview development sets under three input conditions. No retraining was performed between conditions; only the inference input was changed.

| Condition | Input at inference | DEV960-GEN Macro-F1 | DEV600-STR Macro-F1 | Δ vs. full (DEV960) | Δ vs. full (DEV600) |
|---|---|---:|---:|---:|---:|
| **Full input** | Original context + question + claim | **0.8994** | **0.8686** | — | — |
| Claim only | Empty context; original question + claim retained | 0.4043 | 0.3858 | −0.4951 | −0.4828 |
| Shuffled context | A different shuffled context; original question + claim retained | 0.3740 | 0.3508 | −0.5254 | −0.5177 |

**Evaluation files:** `data/evaluation/preview_v0.1/dev_general.jsonl` and `data/evaluation/preview_v0.1/dev_stress.jsonl`.

**Shuffling protocol:** contexts were randomly permuted across each complete evaluation set with seed 0, while preventing an example from receiving its own context. Shuffling was not restricted to the same domain.

These rows are stored separately from `results/csv/metrics_long_canonical.csv` because they are inference-time transformations of one fixed checkpoint rather than independent training runs.
