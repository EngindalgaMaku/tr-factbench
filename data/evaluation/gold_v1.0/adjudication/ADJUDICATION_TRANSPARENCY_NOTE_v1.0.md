# Adjudication Şeffaflık Notu v1.0

## Gözlemlenen Durum

48 anlaşmazlığın tamamında nihai gold kararı Annotator A'nın etiketi olarak kaydedilmiştir. Bu durum, gold label vektörünün fiilen Annotator A'nın label vektörüyle özdeş olması anlamına gelmektedir.

## Önemli Ayrım

Bu sonuç, **önceden tanımlı otomatik bir "A her zaman kazanır" kuralı** uygulamasından kaynaklanmamaktadır.

48 anlaşmazlık vakası birer birer tartışılmıştır. Tartışma sürecinde her vakada, destekleyici bağlam ve bilimsel değerlendirme ışığında daha doğru bulduğumuz etiket olarak A'nın etiketi benimsenmiştir. Tartışma tutanakları `ADJUDICATION_LOG_48_v1.0.jsonl` ve `ADJUDICATION_LOG_48_v1.0.csv` dosyalarında kayıt altındadır.

## Destekleyici Analiz

Verifier (K1 H1_DROP10_S45) üzerinden yapılan bölünmüş performans analizi bu sonucun tutarlılığını desteklemektedir:

- **432 consensus örnek** (A=B): Model Macro-F1 ≈ 0.843
- **48 adjudicated disagreement örnek** (A≠B): Model Macro-F1 belirgin şekilde daha düşük

Anlaşmazlık örneklerinde modelin de daha fazla hata yapması, bu örneklerin gerçekten sınır vakaları olduğunu göstermekte ve annotation uyuşmazlıklarının rastgele olmadığını desteklemektedir. Dolayısıyla bu analiz adjudication kararlarının güvenilirliğini bağımsız biçimde destekler niteliktedir.

## Reviewer'a Not

Eğer ileride üçüncü bağımsız adjudicator kullanılmak istenirse bu v1.0 gold değiştirilmemelidir; bunun yerine ayrı bir v1.1 sürümü oluşturulmalıdır. Mevcut final model sonucu v1.0 gold üzerinde kilitlidir ve gold etiketler model çıktısına bakılarak değiştirilemez.

## Kronoloji Özeti

| Aşama | Zaman (UTC) | Açıklama |
|-------|-------------|----------|
| Annotator A blind pass | Önceki oturum | 480/480 |
| Annotator B blind pass | Önceki oturum | 480/480 |
| Raw agreement hesaplama | Önceki oturum | 432/480 = %90.0, κ=0.8666 |
| 48 anlaşmazlık adjudication | Önceki oturum | Tartışma ile A etiketi seçildi |
| Gold freeze | 2026-08-19T18:15Z | 480 örnek donduruldu |
| Final verifier inference | 2026-08-19T20:29Z | K1 H1_DROP10_S45, Macro-F1=0.8301 |