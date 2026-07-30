# Table I — LLM baselines

## DEV960-GEN

| Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid/missing |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B A4B IT | 8-shot | 0.9699 | 0.9698 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | 8-shot | 0.9678 | 0.9677 | 1.0000 | 0 |
| GPT-4.1 mini | 8-shot | 0.9443 | 0.9437 | 1.0000 | 0 |
| Gemma 4 26B A4B IT | Zero-shot | 0.9433 | 0.9427 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | 8-shot | 0.9310 | 0.9302 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | Zero-shot | 0.8738 | 0.8750 | 1.0000 | 0 |
| GPT-4.1 mini | Zero-shot | 0.8106 | 0.8146 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | Zero-shot | 0.8067 | 0.8052 | 1.0000 | 0 |

## DEV600-STR

| Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid/missing |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B A4B IT | Zero-shot | 0.9599 | 0.9600 | 1.0000 | 0 |
| Gemma 4 26B A4B IT | 8-shot | 0.9168 | 0.9183 | 1.0000 | 0 |
| GPT-4.1 mini | 8-shot | 0.8885 | 0.8883 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | 8-shot | 0.8481 | 0.8567 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | 8-shot | 0.7812 | 0.8017 | 0.9983 | 1 |
| Qwen2.5 72B Instruct | Zero-shot | 0.7800 | 0.7933 | 0.9983 | 1 |
| GPT-4.1 mini | Zero-shot | 0.6581 | 0.6850 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | Zero-shot | 0.5953 | 0.6300 | 1.0000 | 0 |
