# Anotatör rolleri ve uzlaşma yöntemi — düzeltme kaydı (erratum) v1.0.1

- Tarih: 2026-10-06
- Kapsam: yalnızca provenance belgeleri. Gold etiketleri, anotatör ham dosyaları, uzlaşma logu ve
  verifier sonuçları **değişmemiştir** (`TR-FactBench_controlled480_GOLD_v1.0.jsonl`
  SHA-256 `04bcc52ad73ba98c6648ae15d6fabf695eb11bca645a375f7af9cef378df910c`).

## Düzeltilen ifade

`ANNOTATOR_PROVENANCE_REPAIR_LOG_TR_v1.0.md` içindeki şu ifade **hatalıdır ve geri çekilmiştir**:

> "Annotator A sonuçları kullanıcı tarafından iletilmiş ayrı annotator çıktılarıdır;
> kullanıcı/constructor tarafından yapılmış annotation olarak yorumlanmamalıdır."

## Doğru bilgi

| Rol | Kişi | Not |
|---|---|---|
| Annotator A | Veri setini oluşturan araştırmacı (Engin Dalga) | Etiketleme, aday etiketlerin gizlendiği kör görünümler üzerinden yapıldı. Ancak A örnekleri inşa eden kişi olduğundan inşa sürecine önceden maruz kalmıştır; tam bağımsız bir anotatör değildir. |
| Annotator B | Araştırmacıdan bağımsız bir bilişim öğretmeni | Veri inşasına katılmadı; kör görünümler üzerinden etiketledi. |
| Uzlaşma (adjudication) | Üçüncü kişi yok | 48 uyuşmazlık A ve B'nin birlikte tartışmasıyla çözüldü. 48 vakanın tamamında A'nın etiketi benimsendi. |

`PREANNOTATION_RELEASE_GUARDRAIL_v1.1.md` dosyasında "Annotator eligibility/blinding rule frozen"
satırı işaretlidir; ancak uygunluk kuralının metni bu pakette yer almamaktadır.

## Sonuçlara etkisi ve sağlamlık analizi

Ham uyuşma (432/480, %90.00) ve Cohen κ (0.8666) uzlaşmadan **önce** hesaplanmıştır; bu düzeltmeden etkilenmez.

Uzlaşmanın A lehine sonuçlanmasından kaynaklanabilecek yanlılık, sistemlerin (i) B etiketlerine ve
(ii) yalnızca A ile B'nin uzlaştığı 432 örneğe göre de puanlanmasıyla sınanmıştır.

K1 (`verifier_hold/PREDICTIONS_v1.0.jsonl`, `agreement/AB_ALIGNED_480_v1.0.csv` üzerinden):

| Referans | n | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Gold v1.0 (= A) | 480 | 0.8313 | 0.8301 |
| Annotator B | 480 | 0.7812 | 0.7841 |
| A = B uzlaşılan alt küme | 432 | 0.8449 | 0.8425 |

K2 ve hibrit sistemler dahil tam tablo ve yeniden üretim scripti, eşlik eden
`tr-factbench-k2-nli` deposunda `reports/GOLD480_ANNOTATOR_ROBUSTNESS.md` altındadır.
Üç referansta da K1 > K2 sıralaması ve hibrit sistemlerin K1 ile K2'den üstünlüğü korunur.
Hibrit sistemler ile tek başına güçlü LLM'ler arasındaki sıra ise referansa göre değişebilir;
bu sistemler arasındaki farklar zaten istatistiksel olarak anlamlı değildir.

Gelecekte üçüncü bir bağımsız kişi uzlaşma yaparsa, bu v1.0 değiştirilmeyecek; sonuç ayrı bir v1.1 olarak yayımlanacaktır.

## Checksum notu

Bu düzeltmeyle değişen dosyaların SHA-256 değerleri `SHA256SUMS.txt` içinde güncellenmiştir. Ayrıca
`gold/GOLD_FREEZE_RECORD_TR_v1.0.md` satırı, 2026-09-20 tarihli `1a492df` commit'indeki durum satırı
güncellemesinden ("verifier çalıştırıldı") beri eski hash'i gösteriyordu; bu satır da yenilenmiştir.
Freeze kaydında gold etiketi, bağlam, soru veya iddia değişikliği yoktur.
