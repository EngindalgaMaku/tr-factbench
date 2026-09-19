# TR-FactBench A/B Ham Uyum Raporu v1.0

Durum: **İKİ ANNOTATOR PASS TAMAMLANDI — ADJUDICATION ÖNCESİ**

## Tamlık ve bütünlük
- Annotator A: 480/480, benzersiz blind_id 480
- Annotator B: 480/480, benzersiz blind_id 480
- Frozen context/question/claim yüzeyi: A 480/480 PASS, B 480/480 PASS
- B açık self-QC kaydı `B_342dc0b9ac79`: annotator nihai kararı `contradicted` olarak kapatıldı.
- A provenance düzeltmesi: `A_1586f79b23a9` için daha önce assistant tarafından eklenmiş `partially_supported` müdahalesi kaldırıldı; annotatorın ham kararı `contradicted` geri yüklendi.

## Raw agreement (adjudication öncesi)
- Aynı etiket: **432/480 = 90.00%**
- Uyuşmazlık: **48/480 = 10.00%**
- Cohen's kappa: **0.866637**
- Beklenen şans uyumu (Pe): 0.250165

## Etiket dağılımları
- A: {'supported': 120, 'partially_supported': 119, 'contradicted': 121, 'unverifiable': 120}
- B: {'supported': 119, 'partially_supported': 107, 'contradicted': 145, 'unverifiable': 109}

## Bilimsel sıra
Bu rapor construction target veya verifier sonuçları kullanılmadan üretildi. Raw agreement ve kappa adjudication'dan önce dondurulmuştur. `ADJUDICATION_QUEUE_v1.0.jsonl` içindeki 48 uyuşmazlık insan adjudication'ı tamamlanmadan final gold oluşturulamaz. Consensus olan 432 kayıt için geçici consensus etiketi yazılmıştır; bu dosya **gold değildir**.
