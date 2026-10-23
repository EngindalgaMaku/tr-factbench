# İnşa kayıtları (iç kayıt)

Bu klasör, veri setinin ChatGPT web arayüzü (GPT-5.5) yardımıyla inşa edildiği oturumların kayıtlarını içerir. Kayıtlar 2026-10-07'de kopyalanmıştır. Özgün dosyalar yazarda saklanmaktadır.

- `gold480_v1.0/`: Gold-480 kontrollü test kümesinin kapılı (Gate 0–10) inşa protokolü, önerme sayısı kuralı, F01 kısayol raporları, parti paketleri ve ön anotasyon dondurma anlık görüntüsü (v4.3; 12 partinin tamamı, protokol v0.5, yöntem notları, varlık güvenliği politikası).
- `train_dev/`: DEV600 kısayol temizliği paketi, eğitim kümesi yapaylık incelemesi ve eğitim kümesi tek anotatör paketi.

**Yayın durumu:** Paketler maskelenmemiş bağlamlar, takma ad eşlemeleri ve iç köken bilgisi içerir. Varlık ve itibar güvenliği politikası (v0.2) gereği bu bilgiler herkese açık yayımlanmaz; zip ve csv dosyaları `.gitignore` ile depo dışında tutulur. Depoda yalnızca bu açıklama ile `SHA256SUMS.txt` ve protokol metinleri yer alır.

Bütünlük için `SHA256SUMS.txt` dosyasına bakınız.
