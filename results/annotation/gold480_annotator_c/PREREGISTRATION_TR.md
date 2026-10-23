# Gold-480 Üçüncü Anotatör (C) — Ön Kayıt Protokolü

**Tarih:** 2026-10-22 (sonuçlar görülmeden önce yazılmıştır)
**Durum:** Yayım sonrası duyarlılık analizi. Gold-480 v1.0 etiketleri değiştirilmeyecektir.

## 1. Gerekçe

Gold-480'in 480 iddiasında iki anotatör (A ve B) 48 iddiada farklı etiket vermiştir. Bu uyuşmazlıklar birinci yazar (A) tarafından uzlaştırılmış ve 48'inin tamamında A'nın etiketi seçilmiştir. Uzlaştırmayı yapan kişinin aynı zamanda anotatörlerden biri olması bir yanlılık riski oluşturduğu için 48 uyuşmazlık iddiası, A ve B'den bağımsız üçüncü bir anotatöre (C) kör olarak etiketletilecektir.

Ön gösterge (2026-10-22, API'siz, mevcut tahminler): sekiz LLM koşulu 48 iddiada toplam 250 kez A, 115 kez B ile uyuşmuştur. Veri setiyle eğitilmemiş K2 ise 35 kez A, 7 kez B ile uyuşmuştur. Kısmi destek–çelişki sınırındaki 25 iddiada sistemler bölünmüş, doğrulanamayan–kısmi destek sınırındaki 7 iddiada B'ye daha yakın kalmıştır. LLM'ler veri setinin hazırlanmasında kullanılan modelle yanlılık paylaşabileceğinden bu gösterge insan anotatörün yerini tutmaz.

## 2. Tasarım

- **Maddeler:** 48 uyuşmazlık iddiası ve A ile B'nin uzlaştığı iddialardan her Gold sınıfından 8 (toplam 32) dolgu iddiası; toplam 80 madde. Dolgu maddeleri C'nin hangi maddelerin tartışmalı olduğunu bilmemesi ve özeninin ölçülmesi içindir.
- **Seçim ve sıra:** Dolgu maddeleri ve 80 maddenin sırası sabit tohumla (42) belirlenmiştir (`scripts/95_build_annotator_c_package.py`).
- **Körlük:** C'ye yalnızca kör kimlik, bağlam, soru ve iddia verilir. A, B ve Gold etiketleri, madde türü ve veri setinin hazırlanmasına ilişkin bilgiler verilmez. Anahtar dosyası (`private/blind_key.csv`, git dışında) C ile paylaşılmaz.
- **Kılavuz:** A ve B'nin kullandığı kılavuz (`docs/LABEL_GUIDE_TR.md`, sürüm 1.0; kopyası `package/LABEL_GUIDE_TR.md`) ve kısa talimat (`package/INSTRUCTIONS_TR.md`).
- **Yapay zekâ aracı yasağı:** C'den yapay zekâ aracı kullanmaması istenir ve teslimde bunu yazılı olarak teyit etmesi istenir.
- **C'nin nitelikleri:** Türkçeye hâkim, kılavuzu okuyabilecek, A ve B'den bağımsız, veri setinin hazırlanmasına katılmamış bir kişi. Yayınlarda adı verilmez; eğitim düzeyi ve alanı kısaca belirtilir.

## 3. Analiz planı (değiştirilmeyecek)

1. **Özen denetimi:** C'nin 32 dolgu maddesinde A/B'nin ortak etiketiyle uyum oranı ve Cohen kappa değeri. Uyum %75'in altında kalırsa C'nin etiketleri "düşük güvenilirlikli" olarak işaretlenir ama yine de raporlanır.
2. **Birincil analiz:** 48 uyuşmazlık iddiasında C'nin A ile, B ile ya da ikisinden farklı bir etiketle uyuştuğu madde sayıları; uyuşmazlık türüne (A etiketi, B etiketi) göre dağılım.
3. **Duyarlılık analizi:** 48 iddia için üç anotatörün çoğunluk etiketi (A, B, C); üçü de farklıysa Gold etiketi korunur. Bu etiketlerle Gold-480 üzerindeki ana sonuçlar (K1, K2, hibrit P0, sekiz LLM koşulu; doğruluk ve Macro-F1, eşleştirilmiş karşılaştırmalar) yeniden hesaplanır ve yayımlanan sonuçlarla yan yana verilir.
4. **Raporlama:** Sonuç ne olursa olsun tezde ve makalelerde, yayım sonrasında yapılmış bir duyarlılık analizi olarak tarihiyle raporlanır. Gold-480 v1.0 etiketleri değiştirilmez.

## 4. Dosyalar ve SHA-256 değerleri

`package_manifest.json` dosyasında. Paket bu değerlerle C'ye gönderilecek; teslim edilen dosya ayrı bir adla saklanacaktır.

## 5. Değişiklik kaydı

- 2026-10-22: Paket ilk olarak depo dışında (`hls_new/gold480_ucuncu_anotator_v1/`) hazırlandı; C'ye gönderilmeden, aynı gün bu klasöre taşındı ve dosya adları depo düzenine göre İngilizceleştirildi. Paket aynı betik ve tohumla yeniden üretildi; kör kimlikler ve madde sırası değişmedi (anahtar dosyası birebir aynı).
- 2026-10-22: C ilk iki maddeyi doğrudan depodaki `package/ANNOTATION_C.xlsx` üzerinde etiketledi. Bu ara kayıt `returned/ANNOTATION_C_interim_2026-10-22_rows1-2.xlsx` olarak saklandı; paketteki boş dosya aynı betik ve tohumla yeniden üretildi (anahtar dosyası birebir aynı; yeni SHA-256 değerleri `package_manifest.json`'da). Yalnızca biçim ve kural uygulaması denetlendi (iki satırda da geçerli etiket, eminlik ve not var); etiketler A ve B ile karşılaştırılmadı. C çalışmaya ara kayıt dosyası üzerinden, depo dışında devam edecek.
- 2026-10-23: C 80 maddenin tamamını, depo dışında ara kayıt dosyasının üzerine kaydederek tamamladı (dosya 2026-10-23 22:23'te güncellenmiş). Dosya değiştirilmeden `returned/ANNOTATION_C_final_2026-10-23.xlsx` olarak kopyalandı (SHA-256 `3a40abc62cb019e79e1f8e26c899daf5a07e04aec8c6a4bc5096e6f0eb32f2c5`). Usul denetimi: 80/80 geçerli etiket, sıra ve kör kimlikler anahtarla eşleşiyor, metinler değişmemiş. Eminlik sütunu C tarafından bilerek kullanılmamıştır (80 satırda da 2); ön kayıttaki analizlerde eminlik kullanılmadığından analiz planı değişmez. Etiketler bu kayıt yazılırken A ve B ile karşılaştırılmamıştır.
- 2026-10-23: Engin dosyanın C açısından son hâli olduğunu ve etiketlere dokunulmayacağını teyit etti. Analiz bundan sonra, plan değiştirilmeden `scripts/96_annotator_c_sensitivity.py` ile çalıştırıldı; çıktılar `analysis/C_SENSITIVITY_v1.{json,md}`. C'nin yapay zekâ aracı kullanmadığına dair yazılı teyidi henüz alınmadı.
- 2026-10-23: Analiz tamamlandıktan sonra anahtar dosyası (`private/blind_key.csv`) analizin yeniden üretilebilmesi için depoya eklendi; içeriği yayımlanmış A, B ve Gold etiketlerinden oluşur.
