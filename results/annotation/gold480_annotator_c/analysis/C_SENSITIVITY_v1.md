# Gold-480 annotator C sensitivity analysis (post-release, pre-registered)

Care check (32 fillers): 31/32 = 0.969, kappa 0.958, low reliability: False

Primary (48 disagreements): C = A 46, C = B 2, neither 0

| A / B pair type | C=A | C=B | neither |
|---|---:|---:|---:|
| A=partially_supported / B=contradicted | 24 | 1 | 0 |
| A=unverifiable / B=partially_supported | 6 | 1 | 0 |
| A=supported / B=partially_supported | 5 | 0 | 0 |
| A=contradicted / B=partially_supported | 4 | 0 | 0 |
| A=unverifiable / B=contradicted | 3 | 0 | 0 |
| A=partially_supported / B=supported | 3 | 0 | 0 |
| A=unverifiable / B=supported | 1 | 0 | 0 |

Majority labels: 2 of 480 gold labels change; {'partially_supported->contradicted': 1, 'unverifiable->partially_supported': 1}

| System | Acc gold | Acc majority | Macro-F1 gold | Macro-F1 majority |
|---|---:|---:|---:|---:|
| K1 (ELECTRA-TR) | 0.831 | 0.831 | 0.830 | 0.830 |
| K2 soft-prob | 0.804 | 0.800 | 0.806 | 0.802 |
| Hybrid P0 (V4 judge) | 0.931 | 0.927 | 0.931 | 0.927 |
| GPT-4.1-mini 0-shot | 0.790 | 0.790 | 0.762 | 0.762 |
| GPT-4.1-mini 8-shot | 0.910 | 0.906 | 0.910 | 0.906 |
| Gemma-4-26B 0-shot | 0.904 | 0.904 | 0.902 | 0.902 |
| Gemma-4-26B 8-shot | 0.900 | 0.900 | 0.899 | 0.899 |
| Llama-3.3-70B 0-shot | 0.717 | 0.717 | 0.702 | 0.701 |
| Llama-3.3-70B 8-shot | 0.908 | 0.912 | 0.908 | 0.913 |
| Qwen-2.5-72B 0-shot | 0.825 | 0.825 | 0.816 | 0.816 |
| Qwen-2.5-72B 8-shot | 0.867 | 0.867 | 0.866 | 0.866 |

| P0 vs | gold: P0 only / other only / p | majority: P0 only / other only / p |
|---|---|---|
| K1 (ELECTRA-TR) | 54 / 6 / 0.0 | 53 / 7 / 0.0 |
| K2 soft-prob | 63 / 2 / 0.0 | 63 / 2 / 0.0 |
| GPT-4.1-mini 0-shot | 91 / 23 / 0.0 | 90 / 24 / 0.0 |
| GPT-4.1-mini 8-shot | 35 / 25 / 0.2451 | 35 / 25 / 0.2451 |
| Gemma-4-26B 0-shot | 37 / 24 / 0.1237 | 36 / 25 / 0.2 |
| Gemma-4-26B 8-shot | 39 / 24 / 0.0769 | 38 / 25 / 0.1299 |
| Llama-3.3-70B 0-shot | 120 / 17 / 0.0 | 119 / 18 / 0.0 |
| Llama-3.3-70B 8-shot | 39 / 28 / 0.2215 | 37 / 30 / 0.4638 |
| Qwen-2.5-72B 0-shot | 68 / 17 / 0.0 | 67 / 18 / 0.0 |
| Qwen-2.5-72B 8-shot | 55 / 24 / 0.0006 | 54 / 25 / 0.0015 |
