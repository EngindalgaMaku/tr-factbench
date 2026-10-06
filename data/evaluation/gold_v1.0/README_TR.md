# TR-FactBench GOLD RELEASE v1.0

**Durum: GOLD FROZEN / VERIFIER ÇALIŞTIRILDI — SONUÇLAR DONDUROLDU**

Bu paket 480 kontrollü değerlendirme örneğinin iki annotator pass'i, adjudication kaydı, raw agreement raporları ve final gold etiketlerini içerir.

## Son durum

| Aşama | Durum |
|-------|-------|
| Annotator A | 480/480 CLOSED |
| Annotator B | 480/480 CLOSED |
| Raw agreement | 90.00% |
| Cohen's kappa (A/B) | 0.866637 |
| Uyuşmazlık | 48 |
| Adjudication | TAMAMLANDI |
| Gold | 480/480 FROZEN |
| Final verifier | **ÇALIŞTIRILDI** — K1_H1DROP10S45_GOLD480_v1.0 |

## Final Verifier Sonuçları — K1 H1_DROP10_S45 (N=480)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.8313 (399/480) |
| Macro F1 | **0.8301** |
| Macro Precision | 0.8424 |
| Macro Recall | 0.8314 |
| Cohen kappa (model/gold) | 0.7750 |

| Sınıf | F1 |
|-------|----|
| supported | 0.9091 |
| partially_supported | 0.8078 |
| contradicted | 0.7918 |
| unverifiable | 0.8116 |

## Önemli provenance notu
**Düzeltme (2026-10-06):** Annotator A veri setini oluşturan araştırmacıdır; Annotator B bağımsız bir bilişim öğretmenidir; 48 uyuşmazlık üçüncü kişi olmadan A ve B'nin tartışmasıyla çözülmüştür. Ayrıntı: `provenance/ANNOTATOR_ROLES_ERRATUM_TR_v1.0.1.md`.

Uyuşmazlıklarda A etiketinin seçilmesi önceden tanımlı bir kural olarak uygulanmadı. Adjudication sonucunda 48 uyuşmazlığın tamamında A kararı benimsendi. Bu nedenle final gold label vektörü A'nın label vektörüyle aynıdır; bu durum raporlarda açıkça belirtilmiştir.

Gold etiketler verifier sonucuna göre değiştirilemez.

## Dosya Referansları

| Dosya | Açıklama |
|-------|----------|
| `gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl` | Canonical gold |
| `gold/GOLD_FREEZE_RECORD_TR_v1.0.md` | Gold freeze kaydı |
| `adjudication/ADJUDICATION_LOG_48_v1.0.jsonl` | Adjudication logu |
| `qa/GOLD_RELEASE_QA_TR_v1.0.md` | QA raporu |
| `verifier_hold/VERIFIER_RUN_RECORD_TR_v1.0.md` | Verifier çalışma kaydı |
| `verifier_hold/VERIFIER_CONFIG_v1.0.json` | Model/eval yapılandırması |
| `verifier_hold/PREDICTIONS_v1.0.jsonl` | 480 tahmin |
| `verifier_hold/METRICS_v1.0.json` | Metrik JSON |
| `verifier_hold/REPORT_v1.0.md` | Özet rapor |
