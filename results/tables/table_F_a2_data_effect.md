# Table F — A2 Veri Etkisi Analizi (ELECTRA-TR, all-linear LoRA r=16)

Başarılı protokol seedleri üzerinde koşullu Macro F1 ortalaması ± örneklem SS.
TMED: S43 class collapse → 2/3 başarılı seed. TALL B2 koşularından (seed=42/43/44) türetilmiştir.

| Konfigürasyon | Eğitim Boyutu | Başarılı Seed | DEV960 F1 (mean±std) | DEV600 F1 (mean±std) |
|---|---:|:---:|---|---|
| TMED (1600 tıp) | 1,600 | 2/3 | 0.7615±0.0262 | 0.7787±0.0355 |
| TFIN (1600 finans) | 1,600 | 3/3 | 0.7890±0.0107 | 0.7350±0.0197 |
| TLEG (1600 hukuk) | 1,600 | 3/3 | 0.7419±0.0495 | 0.6079±0.1437 |
| TMIX (1600 karma) | 1,600 | 3/3 | 0.7896±0.0205 | 0.7305±0.0317 |
| TALL (4800 çok-alan) | 4,800 | 3/3 | **0.9040±0.0040** | **0.8726±0.0205** |

## A2 Karşılaştırma Özeti

| Karşılaştırma | Δ (pp) | Yorum |
|---|---:|---|
| Alan çeşitliliği: TMIX vs tek-alan ort. | +2.5pp | Çeşitlilik faydalı |
| Veri ölçeği: TALL-4800 vs TMIX-1600 | +11.4pp | 3x veri artışının etkisi |
| Çok-alan üst sınırı: TALL vs TFIN | +11.5pp | TALL'ın tek-alan üstünlüğü |
