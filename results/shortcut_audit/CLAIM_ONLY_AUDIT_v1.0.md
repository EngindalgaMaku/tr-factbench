# Kısmi girdi kısayol denetimi — dört küme, tek yöntem (v1.0)

GroupKFold k=10 (bağlam grubu), TF-IDF + LinearSVC, grup içi permütasyon sıfır dağılımı (200 tekrar, seed 42). Şans düzeyi Macro-F1 ≈ 0,25. Doğrulayıcı kullanılmadı; veri değişmedi.

| Küme | n | İddia (sözcük) | İddia (karakter) | Sıfır p95 (sözcük/karakter) | Soru | Soru + iddia |
|---|---:|---:|---:|---|---:|---:|
| train | 4800 | 0.516 | 0.574 | 0.261 / 0.260 | 0.382 | 0.504 |
| dev960 | 960 | 0.513 | 0.531 | 0.276 / 0.273 | 0.305 | 0.469 |
| dev600 | 600 | 0.547 | 0.617 | 0.285 / 0.285 | 0.180 | 0.490 |
| gold480 | 480 | 0.480 | 0.483 | 0.285 / 0.283 | 0.184 | 0.442 |

## Sınıf bazında iddia uzunluğu ve bağlam örtüşmesi

| Küme | Sınıf | Ort. sözcük | Bağlamla örtüşme |
|---|---|---:|---:|
| train | supported | 13.05 | 0.665 |
| train | partially_supported | 15.55 | 0.517 |
| train | contradicted | 10.88 | 0.523 |
| train | unverifiable | 10.38 | 0.33 |
| dev960 | supported | 16.61 | 0.766 |
| dev960 | partially_supported | 15.69 | 0.527 |
| dev960 | contradicted | 13.16 | 0.518 |
| dev960 | unverifiable | 12.77 | 0.264 |
| dev600 | supported | 18.5 | 0.685 |
| dev600 | partially_supported | 16.74 | 0.552 |
| dev600 | contradicted | 15.54 | 0.453 |
| dev600 | unverifiable | 14.19 | 0.272 |
| gold480 | supported | 18.68 | 0.754 |
| gold480 | partially_supported | 17.86 | 0.615 |
| gold480 | contradicted | 15.27 | 0.541 |
| gold480 | unverifiable | 17.38 | 0.331 |
