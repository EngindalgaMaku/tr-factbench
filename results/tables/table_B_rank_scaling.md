# Table B — Rank Scaling Analizi (BERT-32k, all-linear, LR=2e-4, seed=42) [Tarihsel referans]

| Run ID | Model | Method | Trainable Params | Trainable % | DEV960 F1 | DEV960 Sup | DEV960 PSup | DEV960 Cont | DEV960 Unver | DEV600 F1 | DEV600 Sup | DEV600 PSup | DEV600 Cont | DEV600 Unver | DEV960-DEV600 Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_M01_BERT32K_ALLLINEAR_LORA8_S42 | BERT-32k | LoRA r=8 | 1,342,468 | 1.20% | 0.6847 | 0.7411 | 0.7261 | 0.5714 | 0.7003 | 0.4582 | 0.5806 | 0.5681 | 0.3789 | 0.3051 | 0.2265 |
| B1_M01_BERT32K_ALLLINEAR_LORA16_S42 | BERT-32k | LoRA r=16 | 2,681,860 | 2.37% | 0.7938 | 0.7812 | 0.7738 | 0.7484 | 0.8719 | 0.7393 | 0.7781 | 0.6242 | 0.7087 | 0.8464 | 0.0545 |
| B1_M01_BERT32K_LORA32_S42 | BERT-32k | LoRA r=32 | 5,360,644 | 4.62% | 0.8212 | 0.8340 | 0.8431 | 0.7289 | 0.8786 | 0.7285 | 0.8142 | 0.6558 | 0.5909 | 0.8529 | 0.0927 |
| B1_M01_BERT32K_FULL_FT_S42 | BERT-32k | Full FT | 110,620,420 | 100.00% | 0.8189 | 0.8176 | 0.7955 | 0.7344 | 0.9280 | 0.6720 | 0.8190 | 0.6065 | 0.5344 | 0.7280 | 0.1469 |
