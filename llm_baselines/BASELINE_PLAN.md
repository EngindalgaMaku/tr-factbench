# LLM Baseline Koşu Planı

## Amaç

Fine-tuned ELECTRA-TR LoRA modelinin (DEV960 F1=0.903, DEV600 F1=0.865) açık ağırlıklı ve kapalı kaynak büyük dil modelleriyle kıyaslanması. Bu karşılaştırma makalenin ana argümanını destekler: alan-spesifik fine-tuning, tek başına prompting'den çok daha iyi sonuç üretiyor.

## Model Seçimi

| # | Model | OpenRouter ID | Tür | Neden? |
|---|---|---|---|---|
| M1 | LLaMA 3.3 70B | `meta-llama/llama-3.3-70b-instruct` | Open-weight | Açık kaynak referans modeli, toplulukta standart 70B baseline |
| M2 | Qwen2.5 72B | `qwen/qwen-2.5-72b-instruct` | Open-weight | Türkçe dahil çok dilli performans, LLaMA ile karşılaştırma |
| M3 | GPT-4.1 | `openai/gpt-4.1` | Proprietary | OpenRouter'da mevcut en güncel OpenAI frontier modeli |

## Koşu Matrisi

Her model için 4 koşu:

| Koşu | Dataset | Prompt | Örnekler | Tahmini Maliyet |
|---|---|---|---|---|
| 1 | dev_stress | zero_shot | 600 | ~$0.12 (LLaMA/Qwen) / ~$2 (GPT-4o) |
| 2 | dev_stress | few_shot_8 | 600 | ~$0.18 (LLaMA/Qwen) / ~$3 (GPT-4o) |
| 3 | dev_general | zero_shot | 960 | ~$0.19 (LLaMA/Qwen) / ~$3 (GPT-4o) |
| 4 | dev_general | few_shot_8 | 960 | ~$0.29 (LLaMA/Qwen) / ~$5 (GPT-4o) |

**Toplam:** 12 koşu = ~$30-40 (GPT-4o ağırlıklı)

## Koşu Sırası

```
# === M1: LLaMA 3.3 70B ===
# 1/4 — şu an çalışıyor
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model meta-llama/llama-3.3-70b-instruct --model-alias llama_3_3_70b --prompt-mode zero_shot

# 2/4
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model meta-llama/llama-3.3-70b-instruct --model-alias llama_3_3_70b --prompt-mode few_shot_8

# 3/4
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model meta-llama/llama-3.3-70b-instruct --model-alias llama_3_3_70b --prompt-mode zero_shot

# 4/4
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model meta-llama/llama-3.3-70b-instruct --model-alias llama_3_3_70b --prompt-mode few_shot_8


# === M2: Qwen2.5 72B ===
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model qwen/qwen-2.5-72b-instruct --model-alias qwen2_5_72b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model qwen/qwen-2.5-72b-instruct --model-alias qwen2_5_72b --prompt-mode few_shot_8
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model qwen/qwen-2.5-72b-instruct --model-alias qwen2_5_72b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model qwen/qwen-2.5-72b-instruct --model-alias qwen2_5_72b --prompt-mode few_shot_8


# === M3: GPT-4.1 ===
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model openai/gpt-4.1 --model-alias gpt_4_1 --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model openai/gpt-4.1 --model-alias gpt_4_1 --prompt-mode few_shot_8
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model openai/gpt-4.1 --model-alias gpt_4_1 --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model openai/gpt-4.1 --model-alias gpt_4_1 --prompt-mode few_shot_8
```

## Sonuç Toplama

Tüm koşular tamamlandıktan sonra:

```bash
python llm_baselines/summarize_results.py
```

Bu komut aşağıdakileri üretir:
- `results/llm_baselines/leaderboard.csv` — tüm model × dataset × prompt kombinasyonları
- Her koşu için `metrics/` klasöründe detaylı raporlar

## Beklenen Makale Tablosu

| Model | Tür | DEV960 F1 (0-shot) | DEV960 F1 (8-shot) | DEV600 F1 (0-shot) | DEV600 F1 (8-shot) |
|---|---|---|---|---|---|
| LLaMA-3.3-70B | Open-weight | TBD | TBD | TBD | TBD |
| Qwen2.5-72B | Open-weight | TBD | TBD | TBD | TBD |
| GPT-4o | Proprietary | TBD | TBD | TBD | TBD |
| **ELECTRA-TR LoRA (fine-tuned)** | **Fine-tuned** | **0.9040** | — | **0.8650** | — |

## Notlar

- Few-shot örnekleri (`prompts/fewshot_examples_tr_v1.jsonl`) nihai koşulardan önce elle incelenmeli ve dondurulumalı.
- GPT-4o için `resolved_model` alanı manifest'e kaydedilir — model versiyonu değişirse izlenebilir.
- Her koşu checkpoint desteklidir; kesilebilir ve `--limit` kaldırılarak kaldığı yerden devam edilir.
- Prompt versiyonu: `tr_factbench_llm_baseline_tr_v1` — koşular arasında sabit tutulmalı.

## Opsiyonel Ek Modeller (M4+)

Ana matris tamamlandıktan sonra değerlendirilebilir. "Küçük LLM vs fine-tuned küçük encoder" karşılaştırması tez argümanını güçlendirir.

| # | Model | OpenRouter ID | Parametreler | Beklenen F1 aralığı | Maliyet/1k |
|---|---|---|---|---|---|
| M4 | LLaMA 3.1 8B | `meta-llama/llama-3.1-8b-instruct` | 8B | dev_general ~0.60-0.70 | ~$0.06 |
| M5 | Qwen2.5 7B | `qwen/qwen-2.5-7b-instruct` | 7B | dev_general ~0.60-0.70 | ~$0.06 |
| M6 | Gemma 3 27B | `google/gemma-3-27b-it` | 27B | dev_general ~0.75-0.85 | ~$0.10 |
| M7 | Gemma 3 12B | `google/gemma-3-12b-it` | 12B | dev_general ~0.65-0.75 | ~$0.07 |

**Öncelik sırası:** M4 (LLaMA 8B) > M6 (Gemma 27B) > M5 (Qwen 7B) = M7 (Gemma 12B)

**Koşu komutları (hazır, çalıştırılmadı):**
```bash
# M4: LLaMA 3.1 8B
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model meta-llama/llama-3.1-8b-instruct --model-alias llama_3_1_8b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model meta-llama/llama-3.1-8b-instruct --model-alias llama_3_1_8b --prompt-mode few_shot_8
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model meta-llama/llama-3.1-8b-instruct --model-alias llama_3_1_8b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model meta-llama/llama-3.1-8b-instruct --model-alias llama_3_1_8b --prompt-mode few_shot_8

# M6: Gemma 3 27B
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model google/gemma-3-27b-it --model-alias gemma_3_27b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_stress.jsonl --dataset-name dev_stress --model google/gemma-3-27b-it --model-alias gemma_3_27b --prompt-mode few_shot_8
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model google/gemma-3-27b-it --model-alias gemma_3_27b --prompt-mode zero_shot
python llm_baselines/run_baseline.py --input data/evaluation/legacy_experiment_version/dev_general.jsonl --dataset-name dev_general --model google/gemma-3-27b-it --model-alias gemma_3_27b --prompt-mode few_shot_8
```