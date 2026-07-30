# TR-FactBench LLM Baselines

Four general-purpose LLMs were evaluated in two prompting conditions: instruction-only
zero-shot and fixed 8-shot. Each request contains exactly one context-claim pair. Gold
labels, candidate labels, example IDs, and context IDs are not exposed to the model.
Strict Macro-F1 counts missing or invalid outputs as errors.

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

Selected ELECTRA checkpoint (`H1_DROP10_S45`): **0.9014 Macro-F1**.

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

Selected ELECTRA checkpoint (`H1_DROP10_S45`): **0.8675 Macro-F1**.

## Interpretation

- Few-shot prompting strongly improves GPT-4.1 mini and Llama 3.3 70B on both sets.
- Qwen2.5 72B improves on DEV960 but changes little on DEV600; one output is invalid or
  missing in each DEV600 prompting condition.
- Gemma is strongest overall in these runs. On DEV600 its zero-shot score exceeds its
  8-shot score, so demonstrations are not uniformly beneficial.
- The selected local ELECTRA checkpoint remains competitive with several API baselines,
  especially on DEV600, while avoiding per-example API inference.

## Protocol notes

- The exact requested and resolved model IDs are preserved in each run manifest and in
  `results/llm_baselines/experiment_matrix.csv`.
- DEV600 Gemma runs requested the OpenRouter `:free` route, whereas DEV960 requested the
  standard model route. These are reported transparently rather than treated as an
  identical provider-routing condition.
- All few-shot runs used the same fixed demonstration file with SHA-256
  `2b30bc4360ea5922d3b969a1fcc65528cf10df481afecdd7ace968142d78e8ba`.
- Archived scores were produced on the original pre-review evaluation files. The single-annotator preview changes 2/960 DEV960 labels and 3/600 Dev600 labels, with one additional Dev600 claim correction. Reported values may therefore shift slightly on the reviewed preview files; API calls and model training were not repeated.

Machine-readable results are available in `results/tables/table_I_llm_baselines.csv` and
`results/llm_baselines/leaderboard.csv`.
