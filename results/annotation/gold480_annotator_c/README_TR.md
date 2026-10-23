# Gold-480 üçüncü anotatör (C) — yayım sonrası duyarlılık analizi

**Durum (2026-10-22):** Paket hazırlandı, henüz anotatöre gönderilmedi.

Gold-480'de A ve B'nin uyuşmadığı 48 iddia birinci yazar (A) tarafından uzlaştırılmış ve tamamında A'nın etiketi seçilmiştir. Bu klasör, bu uyuşmazlıkların A ve B'den bağımsız üçüncü bir anotatöre kör olarak etiketletilmesine ilişkin dosyaları içerir. Gold-480 v1.0 etiketleri değiştirilmez; sonuçlar duyarlılık analizi olarak raporlanır.

| Dosya | İçerik |
|---|---|
| `PREREGISTRATION_TR.md` | Gerekçe, tasarım ve sonuçlar görülmeden yazılmış analiz planı |
| `package/` | Anotatöre gönderilen dosyalar: `ANNOTATION_C.xlsx` (80 madde), `INSTRUCTIONS_TR.md`, `LABEL_GUIDE_TR.md` |
| `package_manifest.json` | Paket dosyalarının ve kaynak Gold dosyalarının SHA-256 değerleri |
| `private/blind_key.csv` | Kör kimlik → yayın kimliği ve madde türü (analiz tamamlanana kadar git dışında tutuldu; 2026-10-23'te yayımlandı) |
| `returned/` | Anotatörün teslim ettiği dosya (teslimden sonra) |

Yeniden üretme: `python scripts/95_build_annotator_c_package.py` (tohum 42).
