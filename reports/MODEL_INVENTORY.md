# TR-FactBench Model Inventory

Complete record of all backbone models and adaptation strategies evaluated during
the TR-FactBench benchmark development. This document exists so that negative,
exploratory, and single-seed results are not hidden from the scientific record.

---

## 1. Backbone screening — B-series (single-seed, seed 42)

All runs used LoRA r=16, all-linear target modules, lr=2e-4, multidomain train split.
Single-seed exploratory screening; models with `degenerate_class_collapse` or weak
performance were not advanced to multi-seed alignment (F-series).

| Run ID | Model | HuggingFace ID | Method | DEV960 F1 | DEV600 F1 | QC status |
|---|---|---|---|---:|---:|---|
| B1_M04 | ELECTRA-TR | `dbmdz/electra-base-turkish-cased-discriminator` | LoRA r=16 | 0.9086 | 0.8545 | success |
| B1_M03 | ConvBERT-TR (mc4) | `dbmdz/convbert-base-turkish-mc4-cased` | LoRA r=16 | 0.8857 | 0.7726 | success |
| B2_M03 | ConvBERT-TR (cased) | `dbmdz/convbert-base-turkish-cased` | LoRA r=16 | 0.8808 | 0.7946 | success |
| B1_M02 | BERT-128k (cased) | `dbmdz/bert-base-turkish-128k-cased` | LoRA r=16 | 0.8660 | 0.7870 | success |
| B2_M02 | BERT-128k (uncased) | `dbmdz/bert-base-turkish-128k-uncased` | LoRA r=16 | 0.8326 | 0.7864 | success |
| B1_M01 | BERT-32k | `dbmdz/bert-base-turkish-cased` | LoRA r=32 | 0.8212 | 0.7285 | success |
| B1_M01 | BERT-32k | `dbmdz/bert-base-turkish-cased` | Full FT | 0.8189 | 0.6720 | success |
| B1_M01 | BERT-32k | `dbmdz/bert-base-turkish-cased` | LoRA r=16 | 0.7938 | 0.7393 | success |
| B1_M07 | mBERT | `google-bert/bert-base-multilingual-cased` | LoRA r=16 | 0.7965 | 0.6774 | success |
| B1_M06 | DistilBERT-TR | `dbmdz/distilbert-base-turkish-cased` | LoRA r=16 | 0.5497 | 0.2855 | success (weak) |
| B1_M05 | RoBERTa-TR | `TURKCELL/roberta-base-turkish-uncased` | LoRA r=16 | 0.1000 | 0.1000 | degenerate_class_collapse |
| B1_M05 | RoBERTa-TR | `TURKCELL/roberta-base-turkish-uncased` | Full FT | 0.2805 | 0.2605 | degenerate_class_collapse |
| B1_M05 | RoBERTa-TR | `TURKCELL/roberta-base-turkish-uncased` | LoRA lr=5e-5 | 0.2090 | 0.2050 | degenerate_class_collapse |
| B1_M08 | XLM-RoBERTa | `FacebookAI/xlm-roberta-base` | LoRA r=16 | 0.1000 | 0.1000 | degenerate_class_collapse |
| B1_M09 | mDeBERTa-v3 | `microsoft/mdeberta-v3-base` | LoRA r=16 | 0.1000 | 0.1000 | degenerate_class_collapse |
| B1_M10 | TabiBERT | `boun-tabilab/TabiBERT` | LoRA r=16 | 0.1343 | 0.1203 | degenerate (skewed) |
| B1_M11 | Turkish-Large-BERT | `ytu-ce-cosmos/turkish-large-bert-cased` | LoRA r=16 | 0.1000 | 0.1000 | degenerate_class_collapse |

> **Note:** "success (weak)" means the run completed without class collapse but performance
> was insufficient to advance to multi-seed alignment. Scores of 0.1000 indicate class collapse
> (model predicts a single label throughout). `degenerate_class_collapse` and `degenerate_skewed`
> are QC statuses assigned by `scripts/14_validate_runs.py`.

---

## 2. Multi-seed model-family comparison (F-series, seeds 42/43/44)

Models that reached multi-seed alignment are reported in the main benchmark table
([`results/tables/table_A_model_comparison.md`](../results/tables/table_A_model_comparison.md)).

| Model | Method | Seeds | DEV960 Avg | DEV600 Avg |
|---|---|---|---:|---:|
| ELECTRA-TR | LoRA r=16 all-linear | 42/43/44 | 0.9058 | 0.8782 |
| ModernBERT-TR | LoRA r=16 all-linear, dropout=0.10 | 42/43/44 | 0.8806 | 0.7911 |
| ConvBERT-TR | LoRA r=16 all-linear | 42/43/44 | 0.8735 | 0.7995 |
| BERT-32k | Full Fine-tune | 43/44/45 | 0.8444 | 0.7598 |
| BERT-32k | LoRA r=32 all-linear | 42/43/44 | 0.8359 | 0.7255 |
| BERT-128k | LoRA r=16 all-linear | 42/43/44 | 0.8097 | 0.7187 |
| mBERT | LoRA r=16 all-linear | 42/43/44 | 0.7759 | 0.6739 |

---

## 3. ELECTRA-TR adaptation strategy screening (C-series)

ELECTRA-TR (`dbmdz/electra-base-turkish-cased-discriminator`) was evaluated under
four adaptation strategies to understand the trade-off between trainable parameters,
VRAM, and performance.

### 3.1 Partial encoder freezing (Freeze-N)

Freeze-N denotes that N% of the encoder layers are frozen (embedding + lower layers).
ELECTRA-TR has 12 encoder layers; Freeze-50 freezes layers 0–5, trains layers 6–11 and classifier.

| Configuration | Frozen layers | Trainable params | Seeds | DEV960 F1 | DEV600 F1 | Status |
|---|---|---:|---|---:|---:|---|
| Freeze-25 | 0–2 | 64,557,954 | 42 | 0.8676 | 0.8297 | success |
| Freeze-50 (lr=2e-5) | 0–5 | 43,237,059 | 42/43/44/45/46 | 0.8734 ± 0.0067 | 0.8165 ± 0.0140 | success |
| Freeze-50 (lr=3e-5) | 0–5 | 43,237,059 | 42 | 0.8827 | 0.8497 | success |
| Freeze-50 (lr=5e-5) | 0–5 | 43,237,059 | 42 | 0.8633 | 0.8059 | success |
| Freeze-75 | 0–8 | 21,916,163 | 42 | 0.8340 | 0.7239 | success |
| Freeze-100 | all encoder | 595,267 | 42 | 0.1253 | 0.1000 | degenerate_class_collapse |

> Freeze-50 five-seed averages use D-series runs (D1_M04_ELECTRA_FREEZE50_S4x) at lr=2e-5.

### 3.2 Full fine-tuning

| Run | lr | Seeds | DEV960 F1 | DEV600 F1 |
|---|---|---|---:|---:|
| C0_M04_ELECTRA_FULL_FT (best) | 2e-5 | 42 | 0.8709 | 0.8228 |
| C1_M04_ELECTRA_FULL_FT | 2e-5 | 43 | 0.8463 | 0.7896 |
| C1_M04_ELECTRA_FULL_FT | 2e-5 | 44 | 0.8305 | 0.7651 |

### 3.3 LoRA rank screening (all-linear, seed 42)

| LoRA rank | Trainable params | DEV960 F1 | DEV600 F1 |
|---:|---:|---:|---:|
| r=4 | 1,257,220 | 0.8825 | 0.8530 |
| r=8 | 1,920,772 | 0.9031 | 0.8307 |
| r=16 | 3,247,876 | 0.9086 | 0.8545 |
| r=32 | 5,902,084 | 0.8866 | 0.8525 |

Optimal rank: **r=16** (gains plateau at r=16; r=32 regresses slightly).

### 3.4 LoRA target module ablation (r=16, seed 42)

| Target modules | Trainable params | DEV960 F1 | DEV600 F1 |
|---|---:|---:|---:|
| Q+V only | 1,183,492 | 0.8833 | 0.8622 |
| All attention (Q+K+V+out) | 1,773,316 | 0.8930 | 0.8462 |
| All linear layers | 3,247,876 | 0.9086 | 0.8545 |

### 3.5 Adaptation strategy summary

| Strategy | Trainable params | DEV960 Avg | DEV600 Avg | Seeds |
|---|---:|---:|---:|---|
| LoRA r=16 all-linear (H1/H2 protocol) | 3,247,876 | **0.8930 ± 0.0071** | **0.8587 ± 0.0155** | 5 |
| Freeze-50 | 43,237,059 | 0.8734 ± 0.0067 | 0.8165 ± 0.0140 | 5 |
| Full fine-tuning | 110,620,420 | ~0.849 | ~0.794 | 3 |

> Freeze-50 and full fine-tuning results come from an earlier experimental stage and use
> slightly different hyperparameters. They are reported here as exploratory results.
> A direct controlled comparison would require re-running under the H1/H2 protocol.

---

## 4. Turkish-Large-BERT follow-up status

The registered `B1_M11_TURKLARGE_ALLLINEAR_LORA16_S42` run collapsed to one class under the tested LoRA configuration. A later sandbox full-fine-tuning attempt showed non-degenerate learning, but the complete run directory, configuration, and final two-set metrics are not included in this repository snapshot. It is therefore not part of the canonical benchmark table. The registered collapse and the unregistered pilot must not be combined into a causal claim about the model family.

---

## 5. Observed outcomes under tested configurations

These statements describe archived outcomes only. They do not establish why a model failed; causal explanations would require controlled follow-up experiments.

| Model family | Archived observation |
|---|---|
| RoBERTa-TR | The tested LoRA, lower-LR LoRA, and full-fine-tuning runs produced collapsed or severely weak predictions. |
| XLM-RoBERTa | The tested LoRA run predicted a single class. |
| mDeBERTa-v3 | The tested LoRA run predicted a single class. |
| TabiBERT | The tested LoRA run produced a severely skewed distribution. |
| Turkish-Large-BERT | The registered LoRA run predicted a single class; an unregistered full-fine-tuning pilot is not scored canonically. |
| DistilBERT-TR | The run completed without collapse but achieved DEV960=0.5497 and DEV600=0.2855. |
| Freeze-100 | Updating only the classifier head produced a collapsed DEV600 result under the tested setup. |

ModernBERT-TR is not part of the negative-results list: its registered three-seed LoRA series is included in the main model-family comparison.
