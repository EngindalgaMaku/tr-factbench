# Table G — D1 Historical Multi-Seed Report (DEV960-GEN Macro F1)

Protokol Bölüm 7.20. Protokol seedleri: 42, 43, 44, 45, 46.

> **Not:** ELECTRA LoRA-r16 S46 seed'inde sınıf çökmesi (degenerate_class_collapse) gerçekleşti.
> Eğitim başarı oranı: **4/5 = %80**. S47 ek kurtarma/duyarlılık koşusu olarak verilmektedir.

## ELECTRA LoRA-r16 (Tarihsel B1 Çoklu-Seed Serisi)

| Seed | DEV960 F1 | Durum |
|---|---:|---|
| S42 | 0.9086 | ✓ başarılı |
| S43 | 0.9012 | ✓ başarılı |
| S44 | 0.9021 | ✓ başarılı |
| S45 | 0.9071 | ✓ başarılı |
| S46 | 0.1000 | ✗ COLLAPSE (degenerate_class_collapse) |
| S47 (kurtarma) | 0.8931 | ✓ ek koşu |

**Başarılı 4 seed ortalaması: 0.9047 ± 0.0037**

## Konfigürasyon Karşılaştırması

| Konfigürasyon | N başarılı | Mean DEV960 F1 | Std |
|---|:---:|---:|---:|
| **ELECTRA LoRA-r16** *(tarihsel seri)* | 4/5 | **0.9047** | 0.0037 |
| ELECTRA Freeze-50 *(alternatif)* | 5/5 | 0.8734 | 0.0067 |
| BERTurk-32k Full FT *(baseline)* | 5/5 | 0.8180 | 0.0276 |

## Pairwise Karşılaştırma (LoRA: 4-seed ort. kullanıldı)

| Karşılaştırma | Δ (pp) |
|---|---:|
| LoRA-r16 vs Freeze-50 | +3.13pp |
| LoRA-r16 vs BERTurk baseline | +8.68pp |
