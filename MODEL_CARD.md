# Model Card — TR-FactBench ELECTRA LoRA

## Status

**Model configuration selected; benchmark release remains a research preview.** The selected checkpoint candidate is `H1_DROP10_S45`. Configuration selection is based on the seed-fixed five-run H1/H2 comparison rather than the strongest individual run. The H1 reproducibility sentinel passed the predeclared 0.010 absolute Macro-F1 tolerance. Model weights are not stored in Git; the LoRA adapter may be published separately after a clean-environment load and inference test. A second annotation pass and adjudication are planned for a future benchmark release.

## Model

| Field | Value |
|---|---|
| Base model | `dbmdz/electra-base-turkish-cased-discriminator` |
| Base-model revision | `44ba696463e4d853078f3094cec76ba4e924cfd1` |
| Selected run ID | `H1_DROP10_S45` |
| Task | Four-class Turkish source-grounded factuality verification |
| PEFT method | LoRA |
| Rank / alpha | 16 / 32 |
| Dropout | 0.10 |
| Seed / Trainer seed / data seed | 45 / 45 / 45 |
| Target modules | all linear layers in the encoder |
| Trainable parameters | 3,247,876 |
| Total parameters | 113,868,296 |
| Trainable share | 2.8523% |
| Training data | `data/primary/train.jsonl` (4,800 examples) |

## Model-family and hyperparameter screening

The repository contains several experiment stages. Their values should not be interpreted as if they came from one single paired comparison.

- The `F1_*` series is an aligned three-seed model-family screening stage. It currently covers ELECTRA-TR, ModernBERT-TR, ConvBERT-TR, BERT-32k (LoRA r=32 and full fine-tune), BERT-128k, and mBERT.
- Rank and target-module studies are mainly single-seed or exploratory ablations.
- The historical `G1_*` dropout series predates the explicit Trainer `seed`/`data_seed` wiring.
- H1 and H2 are the final seed-fixed dropout candidates and are directly comparable.

**ModernBERT-TR note (`F1_M12_*`):** this backbone uses rotary position embeddings and a 1 k-token context window. The registered three-seed series uses `all-linear` LoRA with `dropout=0.10` and reaches DEV960=0.8806 and DEV600=0.7911, placing it second on DEV960 and third on DEV600 in the aligned screening. Separate dropout=0.05 pilot attempts were reported to collapse but are not part of the registered run archive; they are not used to claim that 0.10 is universally required.

## Final seed-fixed dropout comparison

H1 and H2 use seeds 42–46, the same base-model revision, data, optimizer settings, LoRA rank/alpha, target modules, and explicit Trainer `seed`/`data_seed` values. The intended configuration difference is dropout.

| Series | Dropout | DEV960 mean ± SD | DEV600 mean ± SD | Two-set mean | Decision |
|---|---:|---:|---:|---:|---|
| **H1** | **0.10** | 0.8930 ± 0.0071 | **0.8587 ± 0.0155** | **0.8759** | **Selected** |
| H2 | 0.00 | **0.9041 ± 0.0081** | 0.8433 ± 0.0280 | 0.8737 | Not selected |

H2/dropout=0.00 performs better on DEV960. H1/dropout=0.10 performs better and more consistently on DEV600-STR and has a slightly higher two-set mean. The selection therefore remains H1. This is a configuration-level decision over five seeds. `H2_DROP0_S45` is the strongest individual H2 run (DEV960=0.9058, DEV600=0.8803), but the release choice is not based on the single best seed.

### Per-seed results

| Seed | H1 DEV960 | H1 DEV600 | H2 DEV960 | H2 DEV600 |
|---:|---:|---:|---:|---:|
| 42 | 0.8980 | 0.8696 | 0.9055 | 0.8254 |
| 43 | 0.8842 | 0.8714 | 0.8901 | 0.8085 |
| 44 | 0.8940 | 0.8487 | 0.9108 | 0.8441 |
| 45 | **0.9014** | 0.8675 | 0.9058 | **0.8803** |
| 46 | 0.8876 | 0.8363 | 0.9083 | 0.8583 |

The selected checkpoint is `H1_DROP10_S45`, the highest two-set-average checkpoint within the selected H1 configuration.

## Historical G1 collapse record

The archived G1 dropout=0.00 seed-46 run produced a genuine one-class collapse. That result remains part of the historical audit trail. It is **not** treated as evidence that dropout=0.00 is inherently unstable, because all five seed-fixed H2/dropout=0.00 reruns completed successfully.

Including the failed G1 seed, the historical dropout=0.00 mean was 0.7443 ± 0.3603 on DEV960 and 0.7133 ± 0.3434 on DEV600. These values are reported rather than replaced with dashes. The historical dropout=0.05 result remains a three-seed screening result and is not directly mixed with the final H1/H2 comparison.

## Selected checkpoint configuration

```yaml
run_id: H1_DROP10_S45
base_model: dbmdz/electra-base-turkish-cased-discriminator
base_model_revision: 44ba696463e4d853078f3094cec76ba4e924cfd1
method: lora
rank: 16
alpha: 32
dropout: 0.10
seed: 45
trainer_seed: 45
trainer_data_seed: 45
learning_rate: 2.0e-4
batch_size: 8
epochs: 5
warmup_ratio: 0.06
weight_decay: 0.01
max_length: 512
input_template: "[CONTEXT] {context} [QUESTION] {question} [CLAIM] {claim}"
```

## Reproducibility sentinel

The sentinel `H1_SENTINEL_DROP10_S45_RERUN` uses the same H1 configuration and locked environment metadata.

| Evaluation set | Original H1 S45 | Sentinel | Delta |
|---|---:|---:|---:|
| DEV960-GEN | 0.9014 | 0.9027 | +0.0013 |
| DEV600-STR | 0.8675 | 0.8581 | −0.0094 |

The maximum absolute difference is 0.0094, below the predeclared 0.010 threshold. This is **performance-level reproducibility**, not exact prediction-level identity. The runs selected different best epochs and some predictions differ.

Environment recorded for the selected run:

- Python 3.14.3
- PyTorch 2.11.0+cu126
- CUDA 12.6
- Transformers 5.12.1
- PEFT 0.19.1
- NVIDIA GeForce RTX 4060 Laptop GPU

## LLM comparison

The selected checkpoint is compared with four API-based LLMs under zero-shot and fixed 8-shot prompting. Results and exact model identifiers are reported in `reports/LLM_BASELINES.md`. Archived scores use the original evaluation files; the single-annotator preview changes 2/960 DEV960 labels and 3/600 Dev600 labels, with one documented Dev600 claim correction. The runs were not repeated for these limited revisions.

## Intended use

- Research baselines for Turkish factuality verification
- Source-grounded hallucination detection experiments
- PEFT and encoder comparison studies

## Limitations

- Both evaluation sets have one completed blind-human pass. Dev600 annotator 2 is 80/600; DEV960 annotator 2 and adjudication have not started.
- DEV960 and DEV600 were used during model selection and are development sets, not untouched final test sets.
- The model is limited to Turkish and the finance, legal, and medical domains represented in the benchmark.
- It must not be used as an autonomous medical, legal, or financial decision system.
- The selected configuration prioritizes stress-set robustness; H2/dropout=0.00 has the higher general-set mean.

## Pending before Hugging Face preview release

1. Complete the clean-environment adapter load and inference test.
2. Publish the adapter, adapter configuration, label mapping, exact dataset hashes, base-model revision, and tested loading example.

For a later stable release, complete the second blind-human pass, adjudication, and disclosure of any final label changes.

## Planned Hugging Face repository

`EngindalgaMaku/tr-factbench-electra-lora`

## Citation

See `CITATION.cff`.
