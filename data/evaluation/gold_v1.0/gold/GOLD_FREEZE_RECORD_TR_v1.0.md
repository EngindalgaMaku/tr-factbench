# TR-FactBench Gold Release Freeze v1.0

Durum: **GOLD FROZEN / FINAL VERIFIER HENÜZ ÇALIŞTIRILMADI**

Freeze zamanı: `2026-08-19T21:15:00+03:00`

## Girdi durumu
- Annotator A: 480/480 CLOSED
- Annotator B: 480/480 CLOSED
- Raw agreement: 432/480 = 90.00%
- Cohen's kappa: 0.866637
- Raw disagreement: 48/480

## Adjudication
48 A/B uyuşmazlığı adjudication sürecinde tartışıldı. Her bir uyuşmazlıkta Annotator A etiketi nihai karar olarak benimsendi. Bu sonuç **önceden tanımlı otomatik bir A-öncelik kuralına dayanmaz**. Bağımsız üçüncü bir adjudicator kullanıldığı iddia edilmez.

Bu adjudication sonucunda final 480 gold etiket vektörü Annotator A etiket vektörüyle birebir aynıdır. Bu olgu provenance açısından açıkça korunmuştur.

## Gold dağılımı
- supported: 120
- partially_supported: 119
- contradicted: 121
- unverifiable: 120

## Hashler
- Canonical gold JSONL SHA-256: `04bcc52ad73ba98c6648ae15d6fabf695eb11bca645a375f7af9cef378df910c`
- Gold labels CSV SHA-256: `55e82152bf4b093ca3c05830572335b674a4439909bab715e8da6328fd6dd124`
- Adjudication log JSONL SHA-256: `938a9cc6a354f172bf4cc2f7e5b0ca77b3ff8df37b3cf65a3aea56083f15aa13`
- Gold label-vector SHA-256: `44df24c1dccbffdf93e710db6f7c550dbd47bc9130832488dd23a68c4a535614`
- Format-specific release-example/context/question/claim surface SHA-256: `cabe74ceaa8bc410623ea8c0df96d720e23f5d0768c02ade5cf3d207a66a07e2`
- Pre-annotation authoritative evaluation-surface SHA-256 (frozen provenance): `a134bdecb40ca2c8340dca2402b62940793d63e7ff9428f7b3234ea1e81c1b73`

Gold freeze sonrasında context/question/claim veya gold_label üzerinde değişiklik yapılırsa v1.0 freeze geçersiz olur ve yeni sürüm/hash gerekir.
