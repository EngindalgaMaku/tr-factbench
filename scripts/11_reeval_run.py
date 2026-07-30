"""
scripts/11_reeval_run.py

Kaydedilmiş bir checkpoint'ten DEV960 ve DEV600 üzerinde inference'ı
yeniden çalıştırır. predictions_DEV960-GEN.jsonl, predictions_DEV600-STR.jsonl
ve metrics.json'u (accuracy, macro_f1, per-class F1) günceller.

LoRA koşuları için best_model/ klasöründe adapter_model.safetensors beklenir.
Full fine-tune koşuları için best_model/ klasöründe model.safetensors beklenir.

Kullanım:
  python scripts/11_reeval_run.py --run_id B1_M01_BERT32K_FULL_FT_S42
"""
import sys, json, argparse, torch
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR = Q1_DIR / "runs/completed"

LABEL2ID    = {"supported": 0, "partially_supported": 1, "contradicted": 2, "unverifiable": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
BATCH_SIZE  = 32
MAX_LEN     = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def infer(model, tokenizer, data_file: Path):
    records = [json.loads(l) for l in open(data_file, encoding="utf-8")]
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(records), BATCH_SIZE):
            batch   = records[i:i+BATCH_SIZE]
            texts_a = [f"[CONTEXT] {r.get('context', '')}" for r in batch]
            texts_b = [f"[QUESTION] {r.get('question', '')} [CLAIM] {r.get('claim', '')}" for r in batch]
            enc     = tokenizer(texts_a, texts_b, max_length=MAX_LEN, truncation=True,
                                padding=True, return_tensors="pt")
            enc     = {k: v.to(DEVICE) for k, v in enc.items()}
            out     = model(**enc)
            probs   = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds   = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            for r in batch:
                lbl = r.get("gold_label", "")
                if lbl not in LABEL2ID:
                    raise ValueError(f"Record {r.get('example_id','?')}: unknown gold_label '{lbl}'")
                all_labels.append(LABEL2ID[lbl])

    acc    = sum(g == p for g, p in zip(all_labels, all_preds)) / len(all_labels)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    pred_out = []
    for r, p, prob in zip(records, all_preds, all_probs):
        pred_out.append({
            "example_id":      r.get("example_id"),
            "domain":          r.get("domain"),
            "gold_label":      r.get("gold_label"),
            "predicted_label": ID2LABEL[p],
            "probabilities":   {ID2LABEL[i]: float(prob[i]) for i in range(4)}
        })

    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(report["macro avg"]["f1-score"], 4),
        "per_class": {ln: round(report[ln]["f1-score"], 4) for ln in LABEL_NAMES},
        "pred_dist": dict(Counter(ID2LABEL[p] for p in all_preds)),
        "predictions": pred_out
    }


def reeval(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        print(f"[HATA] {run_dir} bulunamadı.")
        sys.exit(1)

    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"[HATA] metrics.json yok — bu koşunun config'i eksik, yeniden eğitim gerekli.")
        sys.exit(1)

    m = json.loads(open(metrics_file, encoding="utf-8").read())
    model_name = m.get("model_id", "")
    method     = m.get("method", "lora")
    seed       = m.get("seed", 42)

    ckpt_dir = run_dir / "ckpt" / model_name.split("/")[-1] / "multidomain" / f"seed_{seed}" / "best_model"
    if not ckpt_dir.exists():
        print(f"[HATA] Checkpoint bulunamadı: {ckpt_dir}")
        sys.exit(1)

    dev960_file = Q1_DIR / "data/evaluation/legacy_experiment_version/dev_general.jsonl"
    dev600_file = Q1_DIR / "data/evaluation/legacy_experiment_version/dev_stress.jsonl"

    print(f"Run: {run_id}")
    print(f"Method: {method} | Checkpoint: {ckpt_dir}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    if method == "lora":
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))

    model.to(DEVICE)
    model.eval()
    print("Model yüklendi.")

    print("DEV960-GEN inference...")
    r960 = infer(model, tokenizer, dev960_file)
    print(f"  accuracy={r960['accuracy']:.4f}  macro_f1={r960['macro_f1']:.4f}")
    print(f"  pred_dist: {r960['pred_dist']}")

    print("DEV600-STR inference...")
    r600 = infer(model, tokenizer, dev600_file)
    print(f"  accuracy={r600['accuracy']:.4f}  macro_f1={r600['macro_f1']:.4f}")
    print(f"  pred_dist: {r600['pred_dist']}")

    # predictions dosyaları güncelle
    with open(run_dir / "predictions_DEV960-GEN.jsonl", "w", encoding="utf-8") as f:
        for item in r960["predictions"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(run_dir / "predictions_DEV600-STR.jsonl", "w", encoding="utf-8") as f:
        for item in r600["predictions"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # metrics.json güncelle
    m["d960_accuracy"]               = r960["accuracy"]
    m["d960_macro_f1"]               = r960["macro_f1"]
    m["d960_supported_f1"]           = r960["per_class"]["supported"]
    m["d960_partially_supported_f1"] = r960["per_class"]["partially_supported"]
    m["d960_contradicted_f1"]        = r960["per_class"]["contradicted"]
    m["d960_unverifiable_f1"]        = r960["per_class"]["unverifiable"]
    m["d600_accuracy"]               = r600["accuracy"]
    m["d600_macro_f1"]               = r600["macro_f1"]
    m["d600_supported_f1"]           = r600["per_class"]["supported"]
    m["d600_partially_supported_f1"] = r600["per_class"]["partially_supported"]
    m["d600_contradicted_f1"]        = r600["per_class"]["contradicted"]
    m["d600_unverifiable_f1"]        = r600["per_class"]["unverifiable"]

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    print(f"\nDosyalar güncellendi: {run_dir}")
    print("  - predictions_DEV960-GEN.jsonl")
    print("  - predictions_DEV600-STR.jsonl")
    print("  - metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    reeval(args.run_id)
