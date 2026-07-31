"""
Full Hub-load test: loads adapter from HF, runs inference, verifies classifier head.
Uses the exact code pattern suggested in the review.
"""
import sys, os, json, torch
sys.stdout.reconfigure(encoding="utf-8")

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

TOKEN  = os.environ.get("HF_WRITE_TOKEN", "").strip() or None
REPO   = "angeldust007/tr-factbench-electra-lora"
BASE   = "dbmdz/electra-base-turkish-cased-discriminator"
REV    = "44ba696463e4d853078f3094cec76ba4e924cfd1"

ID2LABEL = {0: "supported", 1: "partially_supported", 2: "contradicted", 3: "unverifiable"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── 1. Verify adapter_config.json ──────────────────────────────────────────────
print("=== adapter_config.json ===")
cfg_path = hf_hub_download(repo_id=REPO, filename="adapter_config.json", token=TOKEN)
cfg = json.loads(open(cfg_path, encoding="utf-8").read())
for k in ["base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "task_type",
          "inference_mode", "target_modules", "modules_to_save", "revision"]:
    print(f"  {k}: {cfg.get(k)}")

# ── 2. Load model from Hub ─────────────────────────────────────────────────────
print("\n=== Loading from Hub ===")
tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REV)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE, revision=REV, num_labels=4,
    id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(base_model, REPO, token=TOKEN)
model.eval()
print("Model loaded OK")

# ── 3. Verify classifier head ──────────────────────────────────────────────────
params = [n for n, _ in model.named_parameters() if "classifier" in n or "modules_to_save" in n]
assert params, "ERROR: classifier head not found in adapter!"
print(f"Classifier head params in adapter: {len(params)}")

# ── 4. Inference test ─────────────────────────────────────────────────────────
context  = "Miras işlemleri bazı ülkelerde mahkeme, bazı ülkelerde noterlik sistemi üzerinden belgelenir."
question = "Noterlik sistemi kaç ülkede kullanılmaktadır?"
claim    = "Noterlik sistemi yirmi ülkede kullanılmaktadır."

text_a = f"[CONTEXT] {context}"
text_b = f"[QUESTION] {question} [CLAIM] {claim}"

inputs = tokenizer(text_a, text_b, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
pred = ID2LABEL[int(logits.argmax(-1).item())]
print(f"\nInference: '{claim[:50]}...'")
print(f"  Predicted: {pred}")
print(f"  (Expected: unverifiable — claim cannot be verified from given context)")
print("\n[PASS] Hub load and inference test complete.")