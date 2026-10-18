# Gate 2 paired context-group bootstrap

- Dataset: gold_v1.0 (N=480, 120 context groups × 4)
- Resampling unit: context group, with replacement
- Replicates: 20,000; seed=42; percentile 95% CI
- Δ = system Macro-F1 − K1 Macro-F1

## Absolute Macro-F1

| System | Macro-F1 | 95% CI |
|---|---:|---:|
| K1 ELECTRA-TR LoRA (H1_DROP10_S45) | 0.8301 | [0.7983, 0.8604] |
| GPT-4.1-mini zero-shot | 0.7621 | [0.7301, 0.7939] |
| GPT-4.1-mini 8-shot | 0.9098 | [0.8822, 0.9352] |
| Gemma-4-26B zero-shot | 0.9022 | [0.8755, 0.9269] |
| Gemma-4-26B 8-shot | 0.8991 | [0.8738, 0.9226] |
| Llama-3.3-70B zero-shot | 0.7016 | [0.6677, 0.7348] |
| Llama-3.3-70B 8-shot | 0.9085 | [0.8831, 0.9317] |
| Qwen-2.5-72B zero-shot | 0.8160 | [0.7834, 0.8475] |
| Qwen-2.5-72B 8-shot | 0.8655 | [0.8343, 0.8946] |

## Paired Δ vs K1

| System | Δ Macro-F1 | 95% CI | 0 excluded |
|---|---:|---:|:---:|
| GPT-4.1-mini zero-shot | -0.0680 | [-0.1135, -0.0227] | yes |
| GPT-4.1-mini 8-shot | 0.0797 | [0.0416, 0.1183] | yes |
| Gemma-4-26B zero-shot | 0.0721 | [0.0355, 0.1097] | yes |
| Gemma-4-26B 8-shot | 0.0690 | [0.0365, 0.1027] | yes |
| Llama-3.3-70B zero-shot | -0.1284 | [-0.1700, -0.0873] | yes |
| Llama-3.3-70B 8-shot | 0.0784 | [0.0435, 0.1136] | yes |
| Qwen-2.5-72B zero-shot | -0.0141 | [-0.0550, 0.0278] | no |
| Qwen-2.5-72B 8-shot | 0.0354 | [-0.0041, 0.0751] | no |

## Paired 8-shot − zero-shot

| Contrast | Δ Macro-F1 | 95% CI | 0 excluded |
|---|---:|---:|:---:|
| GPT-4.1-mini 8-shot − GPT-4.1-mini zero-shot | 0.1477 | [0.1135, 0.1823] | yes |
| Gemma-4-26B 8-shot − Gemma-4-26B zero-shot | -0.0032 | [-0.0313, 0.0248] | no |
| Llama-3.3-70B 8-shot − Llama-3.3-70B zero-shot | 0.2069 | [0.1776, 0.2370] | yes |
| Qwen-2.5-72B 8-shot − Qwen-2.5-72B zero-shot | 0.0495 | [0.0159, 0.0824] | yes |
