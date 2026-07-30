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

Arayüz Dev600, Dev960, tam eğitim kümesi ve 600 örneklik eğitim kalite örneklemiyle çalışır.

## Eğitim kalite örneklemi

Örneklemi oluşturmak veya yenilemek için:

```bash
python scripts/41_iaa_sampling.py
```

Üretilen örneklem:

- 150 bağlam grubu,
- her alandan 50 bağlam,
- her bağlamdaki dört iddianın tamamı,
- toplam 600 örnek,
- kör girişte opak kimlikler

içerir. `train_audit_600_candidate_reference.jsonl` anotörlere gösterilmemelidir.

İki kör insan geçişinden sonra:

```bash
python scripts/42_compute_iaa.py \
  --annotator1 results/annotation/train_audit_600/train_audit_600_blind_annotator_1.jsonl \
  --annotator2 results/annotation/train_audit_600/train_audit_600_blind_annotator_2.jsonl
```

Rapor uzlaştırma öncesi ham uyumu, ağırlıksız Cohen κ değerini, karmaşıklık matrisini, sınıf/alan tanılarını, anlaşmazlıkları ve bağlam düzeyinde bootstrap güven aralıklarını üretir.

## Dev600 raporu

```bash
python scripts/51_annotation_report_dev600.py \
  --candidate-review results/annotation/dev600/dev_stress_candidate_label_review.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_1.jsonl \
  --blind results/annotation/dev600/dev_stress_blind_annotator_2.jsonl
```

Aday etiket iyileştirme sonuçları ile iki-kör-insan IAA sonuçları ayrı bölümlerde raporlanmalıdır.
