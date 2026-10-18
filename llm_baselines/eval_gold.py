"""
eval_gold.py — Gold v1.0 tahminlerini değerlendir.

Gold dosyasında `release_example_id` var, tahminlerde `example_id` var.
Bu script bu dönüşümü otomatik halleder.

Kullanım:
    python eval_gold.py --model openai/gpt-4.1-mini --prompt-mode zero_shot
    python eval_gold.py  # varsayılanlar
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import LABELS, read_jsonl, safe_slug

GOLD_PATH = REPO_ROOT / "data" / "evaluation" / "gold_v1.0" / "gold" / "TR-FactBench_controlled480_GOLD_v1.0.jsonl"
RESULTS_ROOT = REPO_ROOT / "results" / "llm_baselines"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="openai/gpt-4.1-mini")
    p.add_argument("--prompt-mode", "-p", default="zero_shot", choices=["zero_shot", "few_shot_8"])
    p.add_argument("--model-alias", default=None,
                   help="run_baseline.py'nin kullandığı klasör adı (varsayılan: model slug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_alias = args.model_alias or safe_slug(args.model)

    pred_path = RESULTS_ROOT / "gold_v1.0" / model_alias / args.prompt_mode / "predictions.jsonl"
    if not pred_path.exists():
        print(f"[ERROR] Tahmin dosyası bulunamadı: {pred_path}")
        sys.exit(1)

    gold_rows = read_jsonl(GOLD_PATH)
    pred_rows = read_jsonl(pred_path)
    pred_by_id = {r["example_id"]: r for r in pred_rows}

    y_true, y_pred, domains = [], [], []
    missing = 0

    for g in gold_rows:
        eid = g.get("release_example_id") or g.get("example_id", "")
        p = pred_by_id.get(eid)
        gold_label = g["gold_label"]
        if not p or p.get("status") != "success" or p.get("predicted_label") not in LABELS:
            missing += 1
            continue
        y_true.append(gold_label)
        y_pred.append(p["predicted_label"])
        domains.append(g.get("domain", "unknown"))

    n = len(y_true)
    total = len(gold_rows)

    if not y_true:
        print("[ERROR] Hiç geçerli tahmin yok.")
        sys.exit(1)

    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    report_dict = __import__("sklearn.metrics", fromlist=["classification_report"]).classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )

    print("\n" + "=" * 65)
    print(f"  {args.model} / {args.prompt_mode} — Gold v1.0 Sonuçları")
    print(f"  Model  : {args.model}")
    print(f"  Prompt : {args.prompt_mode}")
    print("=" * 65)
    print(f"  N             : {n}/{total}  (missing/failed={missing})")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Weighted F1   : {weighted_f1*100:.2f}%")
    print("-" * 65)
    print("  Per-Class F1:")
    for lbl in LABELS:
        cls = report_dict.get(lbl, {})
        print(f"    {lbl:<25}  F1={cls.get('f1-score',0)*100:.2f}%  "
              f"P={cls.get('precision',0)*100:.2f}%  R={cls.get('recall',0)*100:.2f}%  "
              f"(n={int(cls.get('support',0))})")

    # Domain bazlı
    from collections import defaultdict
    domain_data: dict[str, dict] = defaultdict(lambda: {"true": [], "pred": []})
    for t, p2, d in zip(y_true, y_pred, domains):
        domain_data[d]["true"].append(t)
        domain_data[d]["pred"].append(p2)

    if len(domain_data) > 1:
        print("-" * 65)
        print("  Domain F1 (Macro):")
        for dom in sorted(domain_data):
            dt, dp = domain_data[dom]["true"], domain_data[dom]["pred"]
            df1 = f1_score(dt, dp, labels=LABELS, average="macro", zero_division=0)
            dacc = accuracy_score(dt, dp)
            print(f"    {dom:<12}  Macro F1={df1*100:.2f}%  Acc={dacc*100:.2f}%  (n={len(dt)})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("-" * 65)
    print("  Confusion Matrix (rows=gold, cols=pred):")
    header = "  " + " " * 28 + "  ".join(f"{l[:4]:>5}" for l in LABELS)
    print(header)
    for i, lbl in enumerate(LABELS):
        row = "  " + f"{lbl:<28}" + "  ".join(f"{cm[i,j]:>5}" for j in range(len(LABELS)))
        print(row)
    print("=" * 65)

    # Metrikleri kaydet
    out_dir = pred_path.parent / "metrics"
    out_dir.mkdir(exist_ok=True)
    metrics = {
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "dataset": "gold_v1.0",
        "n_total": total,
        "n_valid": n,
        "n_missing_failed": missing,
        "coverage_rate": n / total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            lbl: {
                "f1": report_dict.get(lbl, {}).get("f1-score", 0),
                "precision": report_dict.get(lbl, {}).get("precision", 0),
                "recall": report_dict.get(lbl, {}).get("recall", 0),
                "support": int(report_dict.get(lbl, {}).get("support", 0)),
            }
            for lbl in LABELS
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "label_dist_gold": dict(Counter(y_true)),
        "label_dist_pred": dict(Counter(y_pred)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrikler kaydedildi: {out_dir / 'metrics.json'}\n")


if __name__ == "__main__":
    main()