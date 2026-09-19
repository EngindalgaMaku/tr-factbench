# TR-FactBench — Final Verifier Run Record v1.0

**Durum: VERIFIER ÇALIŞTIRILDI / SONUÇLAR DONDUROLDU**

Run zamanı: `2026-08-19T20:29:00Z` (= `2026-08-19T23:29:00+03:00`)

> **Provenance düzeltme notu:** İlk kayıtta timestamp yanlışlıkla `20:29:00+03:00` olarak yazılmıştı (bu 17:29Z anlamına gelir ve gold freeze'den önce olurdu). Doğrusu `20:29:00Z` (= `23:29:00+03:00`). Gold freeze `2026-08-19T18:15:00Z` (`21:15+03:00`), verifier inference `2026-08-19T20:29:00Z` (`23:29+03:00`)'de çalıştırıldı. Kronoloji doğru: freeze → inference.

## Verifier Kimliği

| Alan | Değer |
|------|-------|
| Run ID | K1_H1DROP10S45_GOLD480_v1.0 |
| Verifier ailesi | K1 (ELECTRA-TR LoRA fine-tuned) |
| Model etiketi | H1_DROP10_S45 |
| Mimari | ELECTRA-TR LoRA r=16, dropout=0.10, seed=45 |
| Base model | dbmdz/electra-base-turkish-cased-discriminator |
| Base revision | 44ba696463e4d853078f3094cec76ba4e924cfd1 |
| Eğitim seti | multidomain (train.jsonl) |
| Adapter yolu | runs/ablations/H1_DROP10_S45_ABLATION_run/ckpt/…/best_model |

## Evaluation Seti

| Alan | Değer |
|------|-------|
| Dosya | TR-FactBench_controlled480_GOLD_v1.0.jsonl |
| SHA-256 (gold JSONL) | 04bcc52ad73ba98c6648ae15d6fabf695eb11bca645a375f7af9cef378df910c |
| Örnek sayısı | 480 |
| Etiket dağılımı | supported:120, partially_supported:119, contradicted:121, unverifiable:120 |

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.8313 (399/480) |
| Macro F1 | **0.8301** |
| Macro Precision | 0.8424 |
| Macro Recall | 0.8314 |
| Cohen kappa (model/gold) | 0.7750 |
| Inference süresi | 11.7s / 24.3ms per örnek / GPU |

### Per-Class F1

| Sınıf | Precision | Recall | F1 | Destek |
|-------|-----------|--------|----|--------|
| supported | 0.8647 | 0.9583 | **0.9091** | 120 |
| partially_supported | 0.7574 | 0.8655 | **0.8078** | 119 |
| contradicted | 0.7823 | 0.8017 | **0.7918** | 121 |
| unverifiable | 0.9655 | 0.7000 | **0.8116** | 120 |

### Confusion Matrix (satır=gold, sütun=tahmin)

| gold \ pred | supported | partially_supported | contradicted | unverifiable |
|-------------|-----------|---------------------|--------------|--------------|
| **supported** | 115 | 3 | 1 | 1 |
| **partially_supported** | 11 | 103 | 5 | 0 |
| **contradicted** | 4 | 18 | 97 | 2 |
| **unverifiable** | 3 | 12 | 21 | 84 |

## Çıktı Dosyaları

| Dosya | SHA-256 | Açıklama |
|-------|---------|----------|
| PREDICTIONS_v1.0.jsonl | 15322002241e34ab68fae83d724471b2a4a9730d85707b98efd7295671c8ccab | 480 tahmin + per-label olasılıklar |
| METRICS_v1.0.json | f55f7e04018de1cf7cb7a6ede9b5b00e12d43189054f21fc18c770594fc5ea11 | Makine okunabilir metrik JSON |
| REPORT_v1.0.md | 32d8a9501ea3c5d3c081a06c655d0508a8c67111856402008ee841a51a43a781 | Okunabilir özet rapor |
| VERIFIER_CONFIG_v1.0.json | — | Model/eval yapılandırması |
| run_final_eval.py | — | Evaluation scripti (reproducibility için) |

## Protokol Notları

- Gold etiketler verifier çıktısına göre **değiştirilemez**. Bu tek seferlik final evaluation'dır.
- Evaluation scripti `run_final_eval.py` bu klasörde korunmaktadır (reproducibility).
- `VERIFIER_CONFIG_v1.0.json` model ve eval ayarlarını tam olarak kayıt altına alır.
- Dev seti referans noktası: DEV600-STR Macro F1 = 86.75% → Gold-480 Macro F1 = **83.01%** (~3.7 puan genelleme farkı).

## Hata Analizi Notları

`unverifiable` sınıfında recall düşüklüğü (0.70) dikkat çekicidir: 120 örneğin 36'sı yanlış sınıflandırılmış, büyük çoğunluğu `partially_supported` (12) ve `contradicted` (21) tahmin edilmiş. Bu patern dev seti çalışmalarıyla örtüşmektedir.

`contradicted` → `partially_supported` karışıklığı (18 örnek) ve `partially_supported` → `supported` karışıklığı (11 örnek) en büyük hata kaynakları.