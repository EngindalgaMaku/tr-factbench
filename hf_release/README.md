---
language:
- tr
license: mit
library_name: peft
base_model: dbmdz/electra-base-turkish-cased-discriminator
tags:
- lora
- fact-verification
- turkish
- nli
- sequence-classification
pipeline_tag: text-classification
---

# tr-factbench-electra-lora

Turkish factuality-verification LoRA adapter for [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator).

The adapter was trained on [TRFactBench v0.1.0-preview](https://github.com/EngindalgaMaku/tr-factbench), a Turkish source-grounded factuality benchmark covering finance, legal, and medical domains.

## Model details

| Property | Value |
|---|---|
| Base model | `dbmdz/electra-base-turkish-cased-discriminator` |
| Base model revision | `44ba696463e4d853078f3094cec76ba4e924cfd1` |
| Method | LoRA (PEFT) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.10 |
| Target modules | all-linear (query, key, value, dense, out_proj) |
| Classifier head | saved via `modules_to_save` (classifier, score) |
| Trainable parameters | 3,247,876 / 113,868,296 (2.85%) |
| Seed | 45 |
| Learning rate | 2e-4 |
| Epochs | 5 |
| Batch size | 8 |
| Max sequence length | 512 |
| Training set | `data/primary/train.jsonl` (4,800 examples, balanced across 4 labels × 3 domains) |

## Labels

| ID | Label | Description |
|---|---|---|
| 0 | `supported` | Claim is directly supported by the evidence context. |
| 1 | `partially_supported` | Claim is partially supported; some aspects do not align or cannot be confirmed. |
| 2 | `contradicted` | Claim is explicitly contradicted by the evidence context. |
| 3 | `unverifiable` | Claim cannot be verified or refuted based on the supplied context alone. |

## Performance

Results on the human-reviewed preview development sets (see [TRFactBench v0.1.0-preview](https://github.com/EngindalgaMaku/tr-factbench)):

| Evaluation set | Macro-F1 | Accuracy |
|---|---:|---:|
| DEV960-GEN (general dev, 960 examples) | 0.8994 | 0.8990 |
| DEV600-STR (stress dev, 600 examples) | 0.8686 | 0.8650 |

> **Note:** The legacy (pre-review) archived scores for the originally selected H1_DROP10_S45 checkpoint are DEV960=0.9014 / DEV600=0.8675. This adapter uses the same configuration and seed but was separately trained (the original checkpoint was not retained). Small score differences reflect training stochasticity.

## Input format

The model expects two text segments concatenated with the tokenizer's separator:

```
text_a = "[CONTEXT] {context}"
text_b = "[QUESTION] {question} [CLAIM] {claim}"
```

The tokenizer handles truncation to 512 tokens.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
BASE_MODEL_REVISION = "44ba696463e4d853078f3094cec76ba4e924cfd1"
ADAPTER_REPO = "angeldust007/tr-factbench-electra-lora"

LABEL_MAPPING = {0: "supported", 1: "partially_supported", 2: "contradicted", 3: "unverifiable"}

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, revision=BASE_MODEL_REVISION,
    num_labels=4, ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

context  = "Türkiye'de lisanslı hekimlerin tıp fakültesinden mezun olması zorunludur. Tıp fakülteleri en az altı yıllık eğitim vermektedir."
question = "Türkiye'de hekim olmak için ne kadar eğitim gereklidir?"
claim    = "Türkiye'de hekim olmak için en az altı yıllık tıp eğitimi zorunludur."

enc = tokenizer(f"[CONTEXT] {context}", f"[QUESTION] {question} [CLAIM] {claim}",
                max_length=512, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits
pred = LABEL_MAPPING[logits.argmax(-1).item()]
print(pred)  # → supported
```

A complete loading and verification script is available in `inference_example.py`.

## Context-dependence diagnostic

The same configuration was evaluated under controlled input ablations (inference-only, no retraining):

| Condition | DEV960-GEN | DEV600-STR |
|---|---:|---:|
| Full input | **0.8994** | **0.8686** |
| No context: question + claim | 0.4043 | 0.3858 |
| Shuffled context: question + claim | 0.3740 | 0.3508 |

Removing or corrupting the context drops Macro-F1 by approximately 0.50. This confirms that the model relies on the supplied evidence rather than question/claim shortcuts.

## Limitations

- Trained and evaluated exclusively on Turkish text.
- Covers three domains: finance, legal, medical.
- DEV960-GEN and DEV600-STR are development sets used during model selection, not held-out final test sets.
- Must not be used as an autonomous medical, legal, or financial decision system.
- Second blind-human annotation pass is a known limitation of this frozen preview, not remaining work required to close the resource.

## Training data

[TRFactBench v0.1.0-preview](https://github.com/EngindalgaMaku/tr-factbench). Train split: 4,800 examples (balanced across 4 labels × 3 domains).

## Citation

```bibtex
@misc{dalga2026trfactbench,
  author = {Dalga, Engin},
  title  = {{TRFactBench}: A Turkish Source-Grounded Factuality Benchmark},
  year   = {2026},
  url    = {https://github.com/EngindalgaMaku/tr-factbench}
}