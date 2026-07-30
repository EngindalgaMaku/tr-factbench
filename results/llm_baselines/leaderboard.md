# TR-FactBench LLM Baseline Results

Generated from `llm_baselines/collect_results.py`.

Metric: **strict_macro_f1** (coverage-penalised macro-F1 over all attempted examples).

### dev_general

| Model | Prompt | Macro-F1 | Accuracy | Coverage |
|---|---|---:|---:|---:|
| GPT-4.1 mini | 8-shot | **0.9443** | 0.9437 | 1.0000 |
| GPT-4.1 mini | zero-shot | **0.8106** | 0.8146 | 1.0000 |
| Gemma 4 26B MoE | 8-shot | **0.9699** | 0.9698 | 1.0000 |
| Gemma 4 26B MoE | zero-shot | **0.9433** | 0.9427 | 1.0000 |
| LLaMA 3.3 70B | 8-shot | **0.9678** | 0.9677 | 1.0000 |
| LLaMA 3.3 70B | zero-shot | **0.8067** | 0.8052 | 1.0000 |
| Qwen2.5 72B | 8-shot | **0.9310** | 0.9302 | 1.0000 |
| Qwen2.5 72B | zero-shot | **0.8738** | 0.8750 | 1.0000 |

### dev_stress

| Model | Prompt | Macro-F1 | Accuracy | Coverage |
|---|---|---:|---:|---:|
| GPT-4.1 mini | 8-shot | **0.8885** | 0.8883 | 1.0000 |
| GPT-4.1 mini | zero-shot | **0.6581** | 0.6850 | 1.0000 |
| Gemma 4 26B MoE | 8-shot | **0.9168** | 0.9183 | 1.0000 |
| Gemma 4 26B MoE | zero-shot | **0.9599** | 0.9600 | 1.0000 |
| LLaMA 3.3 70B | 8-shot | **0.8481** | 0.8567 | 1.0000 |
| LLaMA 3.3 70B | zero-shot | **0.5953** | 0.6300 | 1.0000 |
| Qwen2.5 72B | 8-shot | **0.7812** | 0.8017 | 0.9983 |
| Qwen2.5 72B | zero-shot | **0.7800** | 0.7933 | 0.9983 |

### Best per Model (avg of zero-shot and 8-shot, both datasets)

| Model | dev_stress ZS | dev_stress FS8 | dev_general ZS | dev_general FS8 |
|---|---:|---:|---:|---:|
| LLaMA 3.3 70B | 0.5953 | 0.8481 | 0.8067 | 0.9678 |
| GPT-4.1 mini | 0.6581 | 0.8885 | 0.8106 | 0.9443 |
| Qwen2.5 72B | 0.7800 | 0.7812 | 0.8738 | 0.9310 |
| Gemma 4 26B MoE | 0.9599 | 0.9168 | 0.9433 | 0.9699 |
