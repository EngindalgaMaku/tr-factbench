# TR-FactBench LLM Baseline Sonuçları

Dört genel amaçlı LLM, yalnız yönerge içeren zero-shot ve sabit 8-shot olmak üzere iki
prompt koşulunda değerlendirilmiştir. Her istek yalnızca tek bir bağlam-iddia çifti içerir.
Gold etiket, aday etiket, örnek kimliği ve bağlam kimliği modele gösterilmez. Strict
Macro-F1 hesabında eksik veya geçersiz çıktılar hata kabul edilir.

## DEV960-GEN

| Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid/missing |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B A4B IT | 8-shot | 0.9699 | 0.9698 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | 8-shot | 0.9678 | 0.9677 | 1.0000 | 0 |
| GPT-4.1 mini | 8-shot | 0.9443 | 0.9437 | 1.0000 | 0 |
| Gemma 4 26B A4B IT | Zero-shot | 0.9433 | 0.9427 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | 8-shot | 0.9310 | 0.9302 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | Zero-shot | 0.8738 | 0.8750 | 1.0000 | 0 |
| GPT-4.1 mini | Zero-shot | 0.8106 | 0.8146 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | Zero-shot | 0.8067 | 0.8052 | 1.0000 | 0 |

Seçilen ELECTRA checkpointi (`H1_DROP10_S45`): **0.9014 Macro-F1**.

## DEV600-STR

| Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid/missing |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B A4B IT | Zero-shot | 0.9599 | 0.9600 | 1.0000 | 0 |
| Gemma 4 26B A4B IT | 8-shot | 0.9168 | 0.9183 | 1.0000 | 0 |
| GPT-4.1 mini | 8-shot | 0.8885 | 0.8883 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | 8-shot | 0.8481 | 0.8567 | 1.0000 | 0 |
| Qwen2.5 72B Instruct | 8-shot | 0.7812 | 0.8017 | 0.9983 | 1 |
| Qwen2.5 72B Instruct | Zero-shot | 0.7800 | 0.7933 | 0.9983 | 1 |
| GPT-4.1 mini | Zero-shot | 0.6581 | 0.6850 | 1.0000 | 0 |
| Llama 3.3 70B Instruct | Zero-shot | 0.5953 | 0.6300 | 1.0000 | 0 |

Seçilen ELECTRA checkpointi (`H1_DROP10_S45`): **0.8675 Macro-F1**.

## Yorum

- Few-shot, GPT-4.1 mini ve Llama 3.3 70B sonuçlarını iki kümede de belirgin biçimde artırmıştır.
- Qwen2.5 72B, DEV960'ta iyileşmiş; DEV600'de ise çok sınırlı değişmiştir. DEV600'de her
  iki prompt koşulunda birer geçersiz veya eksik çıktı vardır.
- Bu koşularda en yüksek sonuçları Gemma vermiştir. DEV600'de zero-shot sonucu 8-shot
  sonucundan yüksektir; dolayısıyla örnek eklemek her model ve kümede yarar sağlamamıştır.
- Seçilen yerel ELECTRA checkpointi, özellikle DEV600'de bazı API tabanlı LLM koşullarıyla
  rekabetçi kalırken örnek başına API çıkarımı gerektirmez.

## Protokol notları

- İstenen ve çözümlenen tam model kimlikleri run manifestlerinde ve
  `results/llm_baselines/experiment_matrix.csv` dosyasında korunur.
- DEV600 Gemma koşuları OpenRouter `:free` rotasını, DEV960 koşuları ise standart model
  rotasını istemiştir. Bu fark gizlenmemiş ve aynı sağlayıcı yönlendirme koşuluymuş gibi
  sunulmamıştır.
- Bütün few-shot koşularında SHA-256 değeri
  `2b30bc4360ea5922d3b969a1fcc65528cf10df481afecdd7ace968142d78e8ba` olan sabit örnek dosyası kullanılmıştır.
- Arşivdeki skorlar özgün, inceleme öncesi değerlendirme dosyalarında üretilmiştir. Tek
  anotatörlü preview sürümünde DEV960'ta 2/960, Dev600'de 3/600 etiket değişmiş ve Dev600'de
  bir iddia düzeltmesi yapılmıştır. Güncel preview dosyalarında skorlar küçük ölçüde değişebilir;
  bu sınırlı revizyonlar için API çağrıları ve model eğitimleri tekrarlanmamıştır.

Makine-okunur sonuçlar `results/tables/table_I_llm_baselines.csv` ve
`results/llm_baselines/leaderboard.csv` dosyalarındadır.
