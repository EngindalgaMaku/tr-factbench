# Table H — Final seed-fixed dropout comparison

H1 and H2 use the same seed-fixed training pipeline and seeds 42–46. The intended configuration difference is LoRA dropout.

| Series | Dropout | Seeds | DEV960 mean ± SD | DEV600 mean ± SD | Two-set mean | Selection |
|---|---:|---:|---:|---:|---:|---|
| **H1** | **0.10** | **42–46** | 0.8930 ± 0.0071 | **0.8587 ± 0.0155** | **0.8759** | **Selected** |
| H2 | 0.00 | 42–46 | **0.9041 ± 0.0081** | 0.8433 ± 0.0280 | 0.8737 | Not selected |

H2/dropout=0.00 is stronger on DEV960. H1/dropout=0.10 is stronger and more stable on DEV600-STR and has a slightly higher two-set mean. The selected configuration is therefore H1/dropout=0.10. The decision is based on five-seed configuration-level behavior, not the best individual seed. Although `H2_DROP0_S45` is the strongest single H2 checkpoint, it does not replace the multi-seed configuration-selection criterion.

## Per-seed results

| Seed | H1 DEV960 | H1 DEV600 | H2 DEV960 | H2 DEV600 |
|---:|---:|---:|---:|---:|
| 42 | 0.8980 | 0.8696 | 0.9055 | 0.8254 |
| 43 | 0.8842 | 0.8714 | 0.8901 | 0.8085 |
| 44 | 0.8940 | 0.8487 | 0.9108 | 0.8441 |
| 45 | 0.9014 | 0.8675 | 0.9058 | 0.8803 |
| 46 | 0.8876 | 0.8363 | 0.9083 | 0.8583 |
