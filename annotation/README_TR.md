# Anotasyon ve aday etiket inceleme akışları

TR-FactBench iki farklı süreci birbirinden ayırır.

## 1. Aday etiket iyileştirme

Bu aşama bağımsız anotasyon değil, hızlı kalite kontrolüdür. Arayüzde bilinçli olarak şunlar birlikte gösterilir:

- mevcut aday etiket,
- normalize edilmiş LLM etiketi,
- LLM gerekçesi ve atomik ilişki analizi.

İnsan denetçi adayı kabul eder, düzeltir veya işaretler. Bu kayıtlar veri üretim hatalarını bulmak için kullanılır. İkinci insan anotörü gibi sayılmaz ve insan-insan IAA hesabına katılmaz.

## 2. Bağımsız kör insan anotasyonu

Kör modda yalnız bağlam, soru, iddia ve etiket tanımları gösterilir. Şunlar gizlenir:

- aday etiketler,
- LLM önerileri,
- iç örnek kimlikleri,
- diğer anotörün kararları.

Gerçek iki-insan IAA için iki farklı kişi aynı örnekleri birbirinden bağımsız biçimde kör modda tamamlamalıdır. Cohen κ uzlaştırmadan önce hesaplanır.

## Kurulum

```bash
pip install -r requirements-annotation.txt
```

## Ham ve normalize edilmiş LLM çıktıları

Yeni OpenRouter çalışmaları ham çıktıyı şu dosyaya yazar:

```text
results/annotation/dev600/llm_preannotations_raw.jsonl
```

Ardından:

```bash
python scripts/_llm_postprocess.py
```

çalıştırılır. Bu işlem:

```text
llm_preannotations_normalized.jsonl
llm_postprocess_manifest.json
```

dosyalarını üretir. Eski betiklerle uyumluluk için `llm_preannotations.jsonl` aynı normalize içeriğin takma yolu olarak korunur. Mevcut snapshot'ta tarihsel ham dosyalar bulunmadığı için bu durum postprocess manifestlerinde açıkça belirtilmiştir.

## OpenRouter ön anotasyonu

```bash
python scripts/50_openrouter_preannotate_dev600.py --dry-run --limit-contexts 1
python scripts/50_openrouter_preannotate_dev600.py --limit-contexts 5
python scripts/50_openrouter_preannotate_dev600.py
python scripts/_llm_postprocess.py
```

LLM'ye mevcut aday etiket, gerçek `example_id` veya bağlam içindeki özgün iddia sırası gönderilmez.

## İnceleme arayüzü

```bash
streamlit run annotation/review_app.py
```

Arayüz Dev600 ve Dev960 ile çalışır. Tam train Gate 1 (`train_4800`), kalan 24/4.800 aday etiket incelemesi ve `train_audit_600` iptal / kapsam dışı; bu ön ayarlar sunulmaz.

## Train-4800 önek-150 dondurması (gümüş, 2026-09-20)

`results/annotation/train_4800/` paketinin seed-42 sırasındaki ilk 150 bağlam grubu dondurulmuştur:

- 600 örnek, tek anotör, `blind_human`
- yapı etiketine göre 45/600 uyuşmazlık (%7,50; Wilson %95 GA %5,65–%9,89)
- **gold değil;** bu etiketler `data/primary/train.jsonl` dosyasına yazılmaz
- **IAA değil** (tek anotör)
- **`train_audit_600` değil**

Genel özet: `results/annotation/TRAIN_4800_PREFIX150_FREEZE.md`. Gold-480 IAA değerlendirme-gold kanıtıdır; bu eğitim denetiminin kanıtı değildir.

## İptal edilen eğitim paketleri (kapsam dışı)

Dosya olarak hazırlanmış ama **anotasyon yapılmayacak** paketler:

- tam `train_4800` Gate 1 (dondurulmuş önek-150 dışındaki Reviewer A/B + uzlaştırma)
- kalan 24/4.800 eğitim aday etiket incelemesi
- tabakalı iki-kişilik `train_audit_600`

`results/annotation/train_audit_600/` kullanılmayan bir inşa artığıdır. Önek-150 onun yerine geçmez; o paket tamamlanmayacaktır. `data/primary/train.jsonl` gold ile yeniden etiketlenmez.

## Dev600 raporu

```bash
python scripts/51_annotation_report_dev600.py \
  --candidate-review results/annotation/dev600/dev_stress_candidate_label_review.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_1.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_2.jsonl
```

Aday etiket iyileştirme sonuçları ile iki-kör-insan IAA sonuçları ayrı bölümlerde raporlanmalıdır.
