# TR-FactBench LLM Baselines

Bu klasör, TR-FactBench üzerinde OpenRouter aracılığıyla üretken LLM baseline'ları çalıştırmak için hazırlanmıştır. Mevcut LLM destekli anotasyon kodunun ağ, retry, structured-output ve kayıt yaklaşımı korunmuştur. Benchmark için gerekli ayrım ise uygulanmıştır:

- Her istek yalnız **bir bağlam ve bir iddia** içerir.
- Aynı bağlama ait dört iddia birlikte gönderilmez.
- `gold_label`, aday etiket, gerçek `example_id` ve `context_id` modele gösterilmez.
- Zero-shot ve sabit 8-shot koşulları ayrı tutulur.
- Ham cevap, normalize tahmin, istek günlüğü ve manifest ayrı dosyalara yazılır.

## Klasörü projeye yerleştirme

Bu klasörü TR-FactBench depo köküne kopyalayın:

```text
tr-factbench/
├── data/
├── results/
├── llm_baselines/
└── ...
```

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r llm_baselines/requirements.txt
```

Depo kökünde `.env` oluşturun:

```text
OPENROUTER_API_KEY=...
```

## Tek model ve tek veri kümesi

Zero-shot:

```bash
python llm_baselines/run_baseline.py \
  --input data/evaluation/legacy_experiment_version/dev_stress.jsonl \
  --dataset-name dev_stress \
  --model meta-llama/llama-3.3-70b-instruct \
  --model-alias llama_3_3_70b \
  --prompt-mode zero_shot
```

Few-shot:

```bash
python llm_baselines/run_baseline.py \
  --input data/evaluation/legacy_experiment_version/dev_stress.jsonl \
  --dataset-name dev_stress \
  --model meta-llama/llama-3.3-70b-instruct \
  --model-alias llama_3_3_70b \
  --prompt-mode few_shot_8
```

Önce tek örneklik dry-run yapılması önerilir:

```bash
python llm_baselines/run_baseline.py \
  --input data/evaluation/legacy_experiment_version/dev_stress.jsonl \
  --model meta-llama/llama-3.3-70b-instruct \
  --prompt-mode few_shot_8 \
  --limit 1 \
  --dry-run
```

## Dört model × iki prompt koşulu

`configs/models.example.yaml` dosyasını `configs/models.yaml` adıyla kopyalayın, dört kesin model kimliğini girin ve hepsini etkinleştirin. Ardından:

```bash
python llm_baselines/run_matrix.py \
  --models-config llm_baselines/configs/models.yaml \
  --benchmark-config llm_baselines/configs/benchmark.example.yaml \
  --evaluate
```

Bu komut iki veri kümesi için her etkin modelin zero-shot ve 8-shot koşularını sırayla çalıştırır.

## Değerlendirme

```bash
python llm_baselines/evaluate_baseline.py \
  --dataset data/evaluation/legacy_experiment_version/dev_stress.jsonl \
  --predictions results/llm_baselines/dev_stress/llama_3_3_70b/zero_shot/predictions.jsonl
```

Toplu liderlik tablosu:

```bash
python llm_baselines/summarize_results.py
```

Raporlanan temel değerler:

- strict Macro-F1 ve accuracy
- geçerli çıktı kapsamı ve invalid oranı
- sınıf bazlı precision, recall ve F1
- domain bazlı sonuçlar
- confusion matrix
- `contradicted ↔ unverifiable` karışmaları
- `supported ↔ partially_supported` karışmaları

Geçersiz veya eksik bir LLM çıktısı strict değerlendirmede yanlış tahmin sayılır. Ayrıca yalnız geçerli çıktılardaki performans ayrı olarak kaydedilir.

## Few-shot örnekleri

`prompts/fewshot_examples_tr_v1.jsonl` dosyasında her etiketten iki olmak üzere sekiz farklı train bağlamı bulunur. Örnekler değerlendirme kümelerinden alınmamıştır. Raporlanan koşuların tamamı aynı dosya ve aynı SHA-256 değeriyle çalıştırılmıştır. Dosya içeriği sonuçlar görüldükten sonra değiştirilmemelidir.

Nihai bilimsel koşuldan önce:

1. Sekiz örneği elle inceleyin.
2. Gerekirse yalnız deneyler başlamadan önce değiştirin.
3. Dosyayı dondurun ve son SHA-256 değerini saklayın.
4. LLM sonuçları görüldükten sonra örnekleri modele göre değiştirmeyin.

Manifest, kullanılan few-shot dosyasının SHA-256 değerini otomatik kaydeder.

## Çıktı yapısı

```text
results/llm_baselines/
└── dev_stress/
    └── llama_3_3_70b/
        ├── zero_shot/
        │   ├── predictions.jsonl
        │   ├── raw_responses.jsonl
        │   ├── request_log.jsonl
        │   ├── manifest.json
        │   └── metrics/
        └── few_shot_8/
            └── ...
```

`predictions.jsonl` gold etiketi içermez. Gold yalnız `evaluate_baseline.py` tarafından ayrı veri dosyasından okunur.

## Testler

```bash
pytest -q llm_baselines/tests
```

Testler prompt içine gold/ID sızıntısı olmadığını, few-shot kümesinin dengeli olduğunu ve çıktı doğrulamasını kontrol eder.
