# Q1 Benchmark — Limitations

Bu bölüm olası hakem eleştirilerine karşı dürüst kısıtlamaları belgelemektedir.

## L1: Model ve Çıkarım Türlerinin Karşılaştırılabilirliği (Orta Risk)
Encoder sonuçları eğitilmiş yerel sınıflandırıcılardan, LLM sonuçları ise API üzerinden
zero-shot veya sabit 8-shot prompting ile elde edilmiştir. Dört LLM baseline'ı raporlanmış
olsa da parametre sayısı, sağlayıcı yönlendirmesi, maliyet ve çıkarım koşulları aynı değildir.
Bu nedenle karşılaştırma pratik bir benchmark karşılaştırmasıdır; eşit hesaplama bütçeli
kontrollü bir model ölçekleme deneyi değildir.

## L2: Tek Dil (Düşük-Orta Risk)
Sistem yalnızca Türkçe doğrulama görevi için tasarlanmıştır.
Cross-lingual veya multilingual transfer değerlendirilmemiştir.
**Gerekçe:** Türkçe-spesifik benchmark hedefi.
**İleri çalışma:** mBERT ve XLM-R ile cross-lingual transfer.

## L3: Preview Anotasyon Durumu (Belgelenmiş Sınırlılık)
Dev600 için aday etiket kalite kontrolü ve birinci kör insan geçişi (600/600),
DEV960 için de birinci kör insan geçişi (960/960) tamamlanmıştır. Bu iki küme
tek anotatörlü araştırma ön sürümü olarak yayımlanabilir. İkinci Dev600 geçişi
80/600 düzeyindedir; ikinci DEV960 geçişi ve uzlaştırma kararlı sürüme bırakılmıştır.
Bu nedenle formal insan-insan Cohen κ değeri henüz raporlanmamaktadır.

Eğitim kümesinin tamamı çift anotasyona sokulmayacaktır. Bunun yerine 150 bağlam grubu
ve 600 örnekten oluşan, alan ve bağlam uzunluğuna göre tabakalandırılmış bir kalite denetimi
örneklemi iki farklı kişi tarafından kör biçimde değerlendirilecektir. Kappa uzlaştırmadan
önce, güven aralıkları bağlam grubu düzeyinde bootstrap ile hesaplanacaktır.

## L4: Hiperparametre Arama Kapsamı (Orta Risk)
Learning rate araması yalnızca lr=2e-4 noktasında gerçekleştirilmiştir (ELECTRA için).
Diğer lr değerleri (1e-4, 5e-5) yalnızca BERT-32k üzerinde denenmiştir.
Dropout ve warmup oranı sabit tutulmuştur.
**Bkz.:** `hp_search_summary.json`, `fig_h3_hp_grid_heatmap.png`

## L5: Reproducibility Kısıtları (Düşük-Orta Risk)
- LoRA eğitimi fully deterministic değil (GPU operasyonları)
- trainer_state.json bazı koşularda saklanmamış (ara checkpoint yokluğu)
- ConvBERT S42 ve BERT-128k S42 farklı checkpoint varyantları kullanmış
  (bilinen P2/P3 sorunu, kayıtlara geçirilmiş)
**Bkz.:** `d2_reproducibility.json`, `final_config.lock.yaml`

## L6: Hata Analizi Derinliği (Düşük Risk)
Claim uzunluğu analizi verimizdeki tüm claim'lerin <20 kelime olduğunu ortaya çıkarmış
(claim uzunluk aralığı anlamlı değil).
Multi-hop veya complex reasoning örneklerine ayrılmış bir analiz yapılmamıştır.
**Bulgular:** Mevcut hata analizinde `partially_supported` sınıfı DEV600-STR'de
F1=0.787 ile en zor sınıf (DEV960-GEN'de 0.898). Error pairs baskın: gold=supported → pred=partially_supported.

## L7: Tek Görev (Düşük Risk)
Sonuçlar Türkçe claim verification görevine özgüdür.
Diğer NLI görevlerine transferi gösterilmemiştir.
