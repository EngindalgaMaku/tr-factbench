"""
scripts/44_context_ablation_eval.py

Seçilen H1_DROP10_S45 checkpoint'i üzerinde iki ablasyon koşusu çalıştırır:
  1. claim_only        — context alanı boş bırakılır ([CONTEXT] )
  2. shuffled_context  — her örneğin context'i rastgele başka bir örneğin
                         context'iyle değiştirilir (aynı alan-içi değil, tüm
                         değerlendirme kümesinden)

Her koşul için DEV960-GEN ve DEV600-STR değerlendirme setlerinde inference yapılır.
Sonuçlar runs/ablations/ altına kaydedilir ve bir özet tablo yazdırılır.

Kullanım:
  python scripts/44_context_ablation_eval.py
  python scripts/44_context_ablation_eval.py --seed 0  # shuffle seed'i
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR     = PROJECT_ROOT / "runs/completed"
OUT_DIR      = PROJECT_ROOT / "runs/ablations"

BASE_RUN_ID  = "H1_DROP10_S45_ABLATION"
BASE_MODEL   = "dbmdz/electra-base-turkish-cased-discriminator"

LABEL2ID     = {"supported": 0, "partially_supported": 1, "contradicted": 2, "unverifiable": 3}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES  = ["supported", "partially_supported", "contradicted", "unverifiable"]
BATCH_SIZE   = 32
MAX_LEN      = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_SETS = {
    "DEV960-GEN": "data/evaluation/preview_v0.1/dev_general.jsonl",
    "DEV600-STR": "data/evaluation/preview_v0.1/dev_stress.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )


def load_model(run_dir: Path):
    """Load base model + LoRA adapter from best_model/."""
    adapter_path = run_dir / "ckpt" / BASE_MODEL.split("/")[-1] / "multidomain" / "seed_45" / "best_model"
    if not adapter_path.exists():
        # Fallback: search anywhere under ckpt/
        hits = list(run_dir.glob("ckpt/**/best_model"))
        if not hits:
            raise FileNotFoundError(f"No best_model/ found under {run_dir}")
        adapter_path = hits[0]
    print(f"[INFO] Loading adapter from: {adapter_path.relative_to(PROJECT_ROOT)}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_NAMES),
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval().to(DEVICE)
    return model, tokenizer


def run_inference(model, tokenizer, records: list[dict], mode: str, rng_seed: int = 0) -> tuple[list, list, list]:
    """
    mode:
      'full'     — original context
      'claim_only' — empty context
      'shuffled' — context from a different random example
    """
    if mode == "shuffled":
        rng = random.Random(rng_seed)
        contexts = [r["context"] for r in records]
        shuffled = contexts[:]
        rng.shuffle(shuffled)
        # Ensure no example gets its own context
        for i in range(len(shuffled)):
            if shuffled[i] == contexts[i] and len(records) > 1:
                j = (i + 1) % len(shuffled)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        get_context = lambda i, r: shuffled[i]
    elif mode == "claim_only":
        get_context = lambda i, r: ""
    else:
        get_context = lambda i, r: r.get("context", "")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            texts_a = [f"[CONTEXT] {get_context(i + j, batch[j])}" for j in range(len(batch))]
            texts_b = [f"[QUESTION] {r.get('question', '')} [CLAIM] {r.get('claim', '')}" for r in batch]
            enc = tokenizer(texts_a, texts_b, max_length=MAX_LEN, truncation=True,
                           padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            for r in batch:
                lbl = r.get("gold_label", "")
                if lbl not in LABEL2ID:
                    raise ValueError(f"Unknown gold_label '{lbl}' for {r.get('example_id', '?')}")
                all_labels.append(LABEL2ID[lbl])
    return all_preds, all_labels, all_probs


def evaluate(all_preds, all_labels, all_probs, records: list[dict], mode: str, eval_set: str) -> dict:
    acc = sum(g == p for g, p in zip(all_labels, all_preds)) / len(all_labels)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    predictions = []
    for r, pred_id, prob in zip(records, all_preds, all_probs):
        predictions.append({
            "example_id": r.get("example_id"),
            "domain": r.get("domain"),
            "gold_label": r.get("gold_label"),
            "predicted_label": ID2LABEL[pred_id],
            "probabilities": {LABEL_NAMES[i]: round(float(p), 6) for i, p in enumerate(prob)},
            "ablation_mode": mode,
            "eval_set": eval_set,
        })
    return {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": {k: round(v["f1-score"], 6) for k, v in report.items() if k in LABEL_NAMES},
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for shuffled_context mode")
    args = parser.parse_args()

    run_dir = RUNS_DIR / BASE_RUN_ID
    if not run_dir.is_dir():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"[INFO] Loading model from run: {BASE_RUN_ID}")
    model, tokenizer = load_model(run_dir)

    summary_rows = []
    modes = ["full", "claim_only", "shuffled"]

    for eval_name, eval_rel in EVAL_SETS.items():
        eval_path = PROJECT_ROOT / eval_rel
        records = load_jsonl(eval_path)
        print(f"\n[INFO] Evaluating on {eval_name} ({len(records)} records)")

        for mode in modes:
            print(f"  mode={mode} ...", end="", flush=True)
            preds, labels, probs = run_inference(model, tokenizer, records, mode, args.seed)
            result = evaluate(preds, labels, probs, records, mode, eval_name)

            # Save predictions
            ablation_dir = OUT_DIR / f"{BASE_RUN_ID}_{mode}"
            ablation_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ablation_dir / f"predictions_{eval_name}.jsonl", result["predictions"])

            # Save metrics
            metrics_path = ablation_dir / f"metrics_{eval_name}.json"
            existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            existing[eval_name] = {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "per_class": result["per_class"]}
            metrics_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            print(f" Macro-F1={result['macro_f1']:.4f}  Acc={result['accuracy']:.4f}")
            summary_rows.append({
                "eval_set": eval_name,
                "mode": mode,
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
            })

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Eval Set':<16} {'Mode':<18} {'Macro-F1':>10} {'Accuracy':>10}")
    print("-" * 65)
    for row in summary_rows:
        print(f"{row['eval_set']:<16} {row['mode']:<18} {row['macro_f1']:>10.4f} {row['accuracy']:>10.4f}")
    print("=" * 65)

    # Save combined summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "context_ablation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] Summary saved to: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
