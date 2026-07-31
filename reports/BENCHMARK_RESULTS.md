# TR-FactBench — Benchmark Results

All values in this report are derived from `results/csv/metrics_long_canonical.csv`, rebuilt from run-level `predictions_*.jsonl` files by `scripts/13_collect_results.py`.

## Repository state

- 144 validated run directories
- 432 canonical metric rows
- 126 successful runs
- 17 historical class-collapse runs
- 1 severely skewed run
- 24 runs evaluated on eight domain-specific/general sets; 120 runs evaluated on the two main development sets

**Selected checkpoint candidate:** `H1_DROP10_S45` — ELECTRA-TR, LoRA r=16 all-linear, dropout=0.10, seed=45, with Trainer `seed` and `data_seed` explicitly set.

## Model-family screening

| Model | Method | Trainable params | DEV960 | DEV600 | Seeds |
|---|---|---:|---:|---:|---|
| **ELECTRA-TR** | LoRA r=16 all-linear | 3,247,876 | **0.9058** | **0.8782** | 42/43/44 |
| ModernBERT-TR | LoRA r=16 all-linear | 3,406,852 | 0.8806 | 0.7911 | 42/43/44 |
| ConvBERT-TR | LoRA r=16 all-linear | 3,026,692 | 0.8735 | 0.7995 | 42/43/44 |
| BERT-32k | Full fine-tuning | 110,620,420 | 0.8444 | 0.7598 | 43/44/45 |
| BERT-32k | LoRA r=32 all-linear | 5,360,644 | 0.8359 | 0.7255 | 42/43/44 |
| BERT-128k | LoRA r=16 all-linear | 2,681,860 | 0.8097 | 0.7187 | 42/43/44 |
| mBERT | LoRA r=16 all-linear | 2,681,860 | 0.7759 | 0.6739 | 42/43/44 |

This table belongs to the aligned `F1_*` architecture-screening stage. It is not the final H1/H2 checkpoint comparison. The full-fine-tuning row uses a different seed triplet and is therefore a screening result, not a paired final comparison.

## ELECTRA rank screening

| Rank | Trainable params | DEV960 | DEV600 |
|---:|---:|---:|---:|
| 4 | 1,257,220 | 0.8825 | 0.8530 |
| 8 | 1,920,772 | 0.9031 | 0.8307 |
| **16** | **3,247,876** | **0.9086** | **0.8545** |
| 32 | 5,902,084 | 0.8866 | 0.8525 |

This is a single-seed screening analysis.

## Target-module screening

| Target | Attention modules | FFN modules | Trainable params | DEV960 | DEV600 |
|---|---:|---:|---:|---:|---:|
| Query + Value | 24 | 0 | 1,183,492 | 0.8833 | 0.8622 |
| All attention | 48 | 0 | 1,773,316 | 0.8930 | 0.8462 |
| All linear | 48 | 24 | 3,247,876 | **0.9086** | **0.8545** |

The all-linear run uses more parameters; the comparison does not isolate a causal FFN effect.

## Data scale and domain composition

| Training set | Planned seeds | Successful | DEV960 mean | DEV600 mean |
|---|---:|---:|---:|---:|
| TMED-1600 | 3 | 2 | 0.7615 | 0.7787 |
| TFIN-1600 | 3 | 3 | 0.7890 | 0.7350 |
| TLEG-1600 | 3 | 3 | 0.7419 | 0.6079 |
| TMIX-1600 | 3 | 3 | 0.7896 | 0.7305 |
| TALL-4800 | 3 | 3 | **0.9040** | **0.8726** |

TALL combines more examples and broader context coverage; its advantage must not be described as a pure sample-count effect.

## Final seed-fixed dropout comparison

H1 and H2 use the same current pipeline, the same base-model revision and data, and explicit Trainer `seed`/`data_seed` values. The intended configuration difference is LoRA dropout.

| Series | Dropout | Seeds | DEV960 mean ± SD | DEV600 mean ± SD | Two-set mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| **H1** | **0.10** | **42–46** | 0.8930 ± 0.0071 | **0.8587 ± 0.0155** | **0.8759** | **Selected** |
| H2 | 0.00 | 42–46 | **0.9041 ± 0.0081** | 0.8433 ± 0.0280 | 0.8737 | Not selected |

H2 is stronger on DEV960. H1 is stronger and more stable on DEV600-STR and has a slightly higher two-set mean. Selection is therefore based on the five-seed configuration-level result rather than the strongest individual run.

### Per-seed results

| Seed | H1 DEV960 | H1 DEV600 | H2 DEV960 | H2 DEV600 |
|---:|---:|---:|---:|---:|
| 42 | 0.8980 | 0.8696 | 0.9055 | 0.8254 |
| 43 | 0.8842 | 0.8714 | 0.8901 | 0.8085 |
| 44 | 0.8940 | 0.8487 | 0.9108 | 0.8441 |
| 45 | 0.9014 | 0.8675 | 0.9058 | 0.8803 |
| 46 | 0.8876 | 0.8363 | 0.9083 | 0.8583 |

`H2_DROP0_S45` is a strong individual checkpoint, but `H1_DROP10_S45` remains the selected release candidate because H1 is the selected five-seed configuration and S45 has the highest two-set average within H1.

## Historical G1 dropout record

All G1 rows use the older pre-seed-fix pipeline and are retained for audit purposes rather than mixed with H1/H2.

| Dropout | Seeds | DEV960 mean ± SD | DEV600 mean ± SD | Interpretation |
|---:|---|---:|---:|---|
| 0.00 | 42–46 | 0.7443 ± 0.3603 | 0.7133 ± 0.3434 | includes one real seed-46 class collapse |
| 0.05 | 42–44 | 0.9016 ± 0.0017 | 0.8714 ± 0.0157 | three-seed screening only |
| 0.10 | 42–46 | 0.9042 ± 0.0046 | 0.8736 ± 0.0189 | historical five-seed result |

The G1 dropout=0.00 collapse was genuine in that archived run, but it did not recur in any of the five H2/dropout=0.00 reruns. Dropout=0.00 is therefore not described as inherently unstable or disqualified.

## Reproducibility sentinel

`H1_SENTINEL_DROP10_S45_RERUN` reproduced the selected configuration within the predeclared 0.010 absolute Macro-F1 tolerance:

| Set | Original | Sentinel | Delta |
|---|---:|---:|---:|
| DEV960 | 0.9014 | 0.9027 | +0.0013 |
| DEV600 | 0.8675 | 0.8581 | −0.0094 |

This is performance-level reproducibility, not exact prediction-level identity.

## Context-dependence ablation (inference-only)

A fixed trained ELECTRA checkpoint was evaluated on the human-reviewed preview development sets under three input conditions. The model was not retrained between conditions. The experiment therefore isolates the effect of the inference input rather than training variance.

| Condition | DEV960-GEN Macro-F1 | DEV600-STR Macro-F1 | Δ vs. full (DEV960) | Δ vs. full (DEV600) |
|---|---:|---:|---:|---:|
| **Full input** | **0.8994** | **0.8686** | — | — |
| Claim only | 0.4043 | 0.3858 | −0.4951 | −0.4828 |
| Shuffled context | 0.3740 | 0.3508 | −0.5254 | −0.5177 |

`full_input` uses the original context, question, and claim. In `claim_only`, the context field is empty while the original question and claim are retained. In `shuffled_context`, the original question and claim are retained but the context is replaced by a different randomly shuffled context. The shuffle uses seed 0 across each complete evaluation set, prevents self-matches, and is not constrained to the same domain.

The large reductions under both controls show that the approximately 0.90 full-input performance cannot be attributed to claim wording alone. The shuffled-context condition performs below the context-removed condition, indicating that incorrect evidence can actively alter model decisions rather than being ignored.

The underlying files are `runs/ablations/context_ablation_summary.json` and the condition-specific prediction and metric files under `runs/ablations/`. Publication-ready tables are provided as `results/tables/table_J_context_dependence_ablation.csv` and `results/tables/table_J_context_dependence_ablation.md`. These results are intentionally excluded from `metrics_long_canonical.csv`: they are inference-time transformations of one fixed checkpoint, not additional training runs.

## LLM baselines

Four LLMs were evaluated with instruction-only zero-shot and fixed 8-shot prompts on both development sets. The canonical tables, exact requested/resolved model IDs, coverage, and invalid-output counts are in [`LLM_BASELINES.md`](LLM_BASELINES.md) and `results/tables/table_I_llm_baselines.csv`.

| Model / condition | DEV960 Macro-F1 | DEV600 Macro-F1 |
|---|---:|---:|
| Gemma 4 26B A4B IT, zero-shot | 0.9433 | **0.9599** |
| Gemma 4 26B A4B IT, 8-shot | **0.9699** | 0.9168 |
| GPT-4.1 mini, zero-shot | 0.8106 | 0.6581 |
| GPT-4.1 mini, 8-shot | 0.9443 | 0.8885 |
| Llama 3.3 70B, zero-shot | 0.8067 | 0.5953 |
| Llama 3.3 70B, 8-shot | 0.9678 | 0.8481 |
| Qwen2.5 72B, zero-shot | 0.8738 | 0.7800 |
| Qwen2.5 72B, 8-shot | 0.9310 | 0.7812 |
| Selected ELECTRA S45 | 0.9014 | 0.8675 |

Archived results use the original pre-review evaluation files. The current single-annotator preview changes 2/960 DEV960 labels and 3/600 Dev600 labels, and includes one Dev600 claim correction. The canonical 144-run encoder archive and LLM experiments were not repeated. A separate single-seed retraining of the selected configuration on the reviewed preview files is archived under `runs/preview_evaluations/H1_DROP10_S45_PREVIEW` (DEV960=0.8965, DEV600=0.8522).

## Interpretation limits

- DEV960 and DEV600 are development sets used during model selection; they are not untouched final test sets.
- Annotator 1 is complete for DEV960 and Dev600. Dev600 annotator 2 is 80/600; the remaining second-pass work and adjudication are deferred to the stable release.
- F1, G1, H1, and H2 represent different experiment stages and are labeled separately.

## Machine-readable outputs

- `results/csv/metrics_long_canonical.csv`
- `results/csv/runs.csv`
- `results/tables/table_H_final_dropout_comparison.csv`
- `results/tables/table_J_context_dependence_ablation.csv`
- `results/tables/table_J_context_dependence_ablation.md`
- `results/statistics/final_dropout_comparison.json`
- `results/tables/`
- `results/statistics/`
- `results/figures/en/`
- `results/figures/tr/`
- `results/figures/historical/`
- `results/figures/supplementary/`
