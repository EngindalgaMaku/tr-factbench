# Table B2 — ELECTRA-TR LoRA Rank Scaling Analizi

Model: `dbmdz/electra-base-turkish-cased-discriminator` | Eğitim: `train_4800_context_balanced_core_v1` | Target: `all-linear` | Seed: 42

| Run ID | Model | Method | Rank | Alpha | Trainable Params | Trainable % | DEV960 F1 | DEV960 Sup | DEV960 PSup | DEV960 Cont | DEV960 Unver | DEV600 F1 | DEV600 Sup | DEV600 PSup | DEV600 Cont | DEV600 Unver | DEV960-DEV600 Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C3_M04_ELECTRA_LORA4_S42 | ELECTRA-tr | LoRA r=4 | 4 | 8 | 1,257,220 | 1.12% | 0.8825 | 0.8793 | 0.8548 | 0.8652 | 0.9305 | 0.8530 | 0.8889 | 0.7674 | 0.8491 | 0.9065 | 0.0295 |
| C3_M04_ELECTRA_LORA8_S42 | ELECTRA-tr | LoRA r=8 | 8 | 16 | 1,920,772 | 1.71% | 0.9031 | 0.9091 | 0.8928 | 0.8652 | 0.9451 | 0.8307 | 0.9024 | 0.7405 | 0.8015 | 0.8782 | 0.0724 |
| B1_M04_ELECTRA_ALLLINEAR_LORA16_S42 | ELECTRA-tr | LoRA r=16 ★ | 16 | 32 | 3,247,876 | 2.85% | 0.9086 | 0.9177 | 0.8979 | 0.8841 | 0.9345 | 0.8545 | 0.9161 | 0.7871 | 0.8333 | 0.8817 | 0.0541 |
| C3_M04_ELECTRA_LORA32_S42 | ELECTRA-tr | LoRA r=32 | 32 | 64 | 5,902,084 | 5.07% | 0.8866 | 0.8884 | 0.8709 | 0.8627 | 0.9244 | 0.8525 | 0.9187 | 0.7914 | 0.8213 | 0.8786 | 0.0341 |

## Özet

- r=16 hem DEV960-GEN hem DEV600-STR'de en yüksek Macro F1 değerini üretmiştir.
- r=4→r=8: belirgin kazanç (+2.06pp DEV960), kapasite yetersizliği giderilmiş.
- r=8→r=16: ek kazanç (+0.44pp DEV960), optimal noktaya ulaşılmış.
- r=16→r=32: performans düşmüş (-2.09pp DEV960). Ek kapasite kazanç sağlamamış;
  ELECTRA mimarisi için r=16 optimal rank olarak saptanmıştır.
