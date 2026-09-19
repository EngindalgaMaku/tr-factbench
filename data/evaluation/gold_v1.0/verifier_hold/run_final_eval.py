#!/usr/bin/env python3
"""
TR-FactBench Gold v1.0 — Final Verifier Evaluation
====================================================
Model  : H1_DROP10_S45 (ELECTRA-TR LoRA r=16, seed 45)
Dataset: TR-FactBench_controlled480_GOLD_v1.0.jsonl  (480 örnekli untouched gold set)

Çalıştırma:
    python run_final_eval.py

Çıktılar (bu script'le aynı klasöre):
    final_eval_predictions_v1.0.jsonl  — her örnek için tahmin + olasılıklar
    final_eval_metrics_v1.0.json       — macro/per-class F1, confusion matrix
    final_eval_report_v1.0.md          — okunabilir özet rapor
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent  # hls_new/tr-factbench-v0.1.0-preview/data/evaluation/..
PREVIEW_ROOT = ROOT.parent               # hls_new/tr-factbench-v0.1.0-preview

GOLD_JSONL = HERE.parent / "gold" / "TR-FactBench_controlled480_GOLD_v1.0.jsonl"

ADAPTER_PATH = (
    ROOT
    / "runs"
    / "ablations"
    / "H1_DROP10_S45_ABLATION_run"
    / "ckpt"
    / "electra-base-turkish-cased-discriminator"
    / "multidomain"
    / "seed_45"
    / "best_model"
)

OUT_PREDS   = HERE / "final_eval_predictions_v1.0.jsonl"
OUT_METRICS = HERE / "final_eval_metrics_v1.0.json"
OUT_REPORT  = HERE / "final_eval_report_v1.0.md"

# ── Model config ──────────────────────────────────────────────────────────────
BASE_MODEL          = "dbmdz/electra-base-turkish-cased-discriminator"
BASE_MODEL_REVISION = "44ba696463e4d853078f3094cec76ba4e924cfd1"
MAX_LENGTH          = 512
LABEL_MAPPING       = {0: "supported", 1: "partially_supported", 2: "contradicted", 3: "unverifiable"}
ID2LABEL            = LABEL_MAPPING
LABEL2ID            = {v: k for k, v in LABEL_MAPPING.items()}
LABEL_ORDER         = ["supported", "partially_supported", "contradicted", "unverifiable"]
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    adapter = str(ADAPTER_PATH)
    print(f"Tokenizer: {adapter}")
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)

    print(f"Base model: {BASE_MODEL}@{BASE_MODEL_REVISION}")
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        num_labels=len(LABEL_MAPPING),
        ignore_mismatched_sizes=True,
    )

    print(f"LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval().to(DEVICE)

    # sanity check
    classifier_params = [n for n, _ in model.named_parameters() if "classifier" in n or "modules_to_save" in n]
    assert classifier_params, "Classifier head not found in adapter — adapter may be incomplete."
    print(f"[OK] Classifier head: {len(classifier_params)} params")
    return model, tokenizer


@torch.inference_mode()
def predict_batch(model, tokenizer, rows: list[dict]) -> list[dict]:
    results = []
    for i, row in enumerate(rows, 1):
        text_a = f"[CONTEXT] {row['context'].strip()}"
        text_b = f"[QUESTION] {row['question'].strip()} [CLAIM] {row['claim'].strip()}"
        enc = tokenizer(
            text_a, text_b,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred_id = int(torch.argmax(logits, dim=-1).item())
        pred_label = LABEL_MAPPING[pred_id]

        results.append({
            "release_example_id": row["release_example_id"],
            "gold_label":         row["gold_label"],
            "predicted_label":    pred_label,
            "correct":            pred_label == row["gold_label"],
            "probabilities":      {LABEL_MAPPING[j]: round(p, 4) for j, p in enumerate(probs)},
        })
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} done")
    return results


def compute_metrics(results: list[dict]) -> dict:
    golds = [r["gold_label"] for r in results]
    preds = [r["predicted_label"] for r in results]

    report = classification_report(
        golds, preds, labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(golds, preds, labels=LABEL_ORDER).tolist()
    kappa = cohen_kappa_score(golds, preds)
    accuracy = sum(1 for r in results if r["correct"]) / len(results)

    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall":    round(report[label]["recall"], 4),
            "f1":        round(report[label]["f1-score"], 4),
            "support":   int(report[label]["support"]),
        }
        for label in LABEL_ORDER
    }

    return {
        "n": len(results),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "matrix": cm,
        },
        "gold_distribution": dict(Counter(golds)),
        "pred_distribution": dict(Counter(preds)),
    }


def write_report(metrics: dict, elapsed: float) -> str:
    m = metrics
    pc = m["per_class"]
    cm = m["confusion_matrix"]

    lines = [
        "# TR-FactBench Gold v1.0 — Final Verifier Evaluation Report",
        "",
        f"**Model:** H1_DROP10_S45 (ELECTRA-TR LoRA r=16, seed 45)",
        f"**Dataset:** TR-FactBench_controlled480_GOLD_v1.0.jsonl (N=480)",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {m['accuracy']:.4f} ({int(m['accuracy']*m['n'])}/{m['n']}) |",
        f"| Macro F1 | {m['macro_f1']:.4f} |",
        f"| Macro Precision | {m['macro_precision']:.4f} |",
        f"| Macro Recall | {m['macro_recall']:.4f} |",
        f"| Cohen kappa | {m['cohen_kappa']:.4f} |",
        f"| Inference time | {elapsed:.1f}s ({elapsed/m['n']*1000:.1f}ms/örnek) |",
        "",
        "## Per-Class Metrics",
        "",
        f"| Label | Precision | Recall | F1 | Support |",
        f"|-------|-----------|--------|----|---------|",
    ]
    for label in ["supported", "partially_supported", "contradicted", "unverifiable"]:
        p = pc[label]
        lines.append(f"| {label} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {p['support']} |")

    lines += [
        "",
        "## Confusion Matrix",
        "",
        f"Rows=gold, Cols=predicted. Label order: {', '.join(cm['labels'])}",
        "",
    ]
    header = "| gold \\ pred | " + " | ".join(cm["labels"]) + " |"
    sep    = "|" + "---|" * (len(cm["labels"]) + 1)
    lines += [header, sep]
    for i, row_vals in enumerate(cm["matrix"]):
        line = f"| **{cm['labels'][i]}** | " + " | ".join(str(v) for v in row_vals) + " |"
        lines.append(line)

    lines += [
        "",
        "## Label Distributions",
        "",
        "| Label | Gold | Predicted |",
        "|-------|------|-----------|",
    ]
    for label in ["supported", "partially_supported", "contradicted", "unverifiable"]:
        g = m["gold_distribution"].get(label, 0)
        p = m["pred_distribution"].get(label, 0)
        lines.append(f"| {label} | {g} | {p} |")

    lines += [
        "",
        "---",
        "*Gold labels değiştirilemez. Bu sonuç tek seferlik final evaluation'dır.*",
    ]
    return "\n".join(lines)


def main():
    print("=== TR-FactBench Gold v1.0 Final Evaluation ===")
    print(f"Gold file : {GOLD_JSONL}")
    print(f"Adapter   : {ADAPTER_PATH}")
    print(f"Device    : {DEVICE}")
    print()

    # Load gold data
    with open(GOLD_JSONL, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(rows)} gold examples")
    label_dist = Counter(r["gold_label"] for r in rows)
    print(f"Gold distribution: {dict(label_dist)}")
    print()

    # Load model
    t0 = time.monotonic()
    model, tokenizer = load_model()
    print(f"Model loaded in {time.monotonic()-t0:.1f}s\n")

    # Run inference
    print("Running inference...")
    t1 = time.monotonic()
    results = predict_batch(model, tokenizer, rows)
    elapsed = time.monotonic() - t1
    print(f"Inference done: {elapsed:.1f}s\n")

    # Compute metrics
    metrics = compute_metrics(results)
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({int(metrics['accuracy']*metrics['n'])}/{metrics['n']})")
    print(f"Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"Cohen kappa: {metrics['cohen_kappa']:.4f}")

    # Write outputs
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PREDS, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions → {OUT_PREDS}")

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics    → {OUT_METRICS}")

    report = write_report(metrics, elapsed)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Report     → {OUT_REPORT}")


if __name__ == "__main__":
    main()