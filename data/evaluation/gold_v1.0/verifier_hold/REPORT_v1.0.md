# TR-FactBench Gold v1.0 — Final Verifier Evaluation Report

**Model:** H1_DROP10_S45 (ELECTRA-TR LoRA r=16, seed 45)
**Dataset:** TR-FactBench_controlled480_GOLD_v1.0.jsonl (N=480)
**Date:** 2026-08-19 20:29 UTC

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8313 (399/480) |
| Macro F1 | 0.8301 |
| Macro Precision | 0.8424 |
| Macro Recall | 0.8314 |
| Cohen kappa | 0.7750 |
| Inference time | 11.7s (24.3ms/örnek) |

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| supported | 0.8647 | 0.9583 | 0.9091 | 120 |
| partially_supported | 0.7574 | 0.8655 | 0.8078 | 119 |
| contradicted | 0.7823 | 0.8017 | 0.7918 | 121 |
| unverifiable | 0.9655 | 0.7000 | 0.8116 | 120 |

## Confusion Matrix

Rows=gold, Cols=predicted. Label order: supported, partially_supported, contradicted, unverifiable

| gold \ pred | supported | partially_supported | contradicted | unverifiable |
|---|---|---|---|---|
| **supported** | 115 | 3 | 1 | 1 |
| **partially_supported** | 11 | 103 | 5 | 0 |
| **contradicted** | 4 | 18 | 97 | 2 |
| **unverifiable** | 3 | 12 | 21 | 84 |

## Label Distributions

| Label | Gold | Predicted |
|-------|------|-----------|
| supported | 120 | 133 |
| partially_supported | 119 | 136 |
| contradicted | 121 | 124 |
| unverifiable | 120 | 87 |

---
*Gold labels değiştirilemez. Bu sonuç tek seferlik final evaluation'dır.*