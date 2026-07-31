"""Local adapter test: loads from local path and evaluates 10 fixed examples."""
import sys, json, torch
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ADAPTER = str(ROOT / "runs/ablations/H1_DROP10_S45_ABLATION_run/ckpt/electra-base-turkish-cased-discriminator/multidomain/seed_45/best_model")

sys.path.insert(0, str(Path(__file__).parent))
import inference_example as ie

print(f"Loading from: {ADAPTER}")
model, tokenizer = ie.load_model(ADAPTER)
ie.verify_classifier_head(model)

records = [json.loads(l) for l in (ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:10]
correct = 0
for r in records:
    res = ie.predict(model, tokenizer, r["context"], r["question"], r["claim"])
    match = res["predicted_label"] == r["gold_label"]
    correct += int(match)
    print(f"  {'OK' if match else 'XX'}  pred={res['predicted_label']:<22} gold={r['gold_label']}")
print(f"\nLocal test: {correct}/10 correct")
if correct >= 8:
    print("[PASS] Adapter loads and infers correctly.")
else:
    print("[WARN] Less than 8/10 correct — check adapter.")