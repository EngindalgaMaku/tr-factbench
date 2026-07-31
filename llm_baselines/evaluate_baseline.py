from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import LABELS, read_jsonl, utc_now_iso, validate_dataset  # noqa: E402

INVALID = "__invalid__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one TR-FactBench LLM baseline prediction file.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_block(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "examples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else None,
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if y_true else None,
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0) if y_true else None,
    }


def main() -> None:
    args = parse_args()
    dataset = read_jsonl(args.dataset)
    validate_dataset(dataset, require_gold=True)
    predictions = read_jsonl(args.predictions)
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("Prediction row missing example_id")
        if example_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction: {example_id}")
        prediction_by_id[example_id] = row

    output_dir = args.output_dir or args.predictions.parent / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true: list[str] = []
    y_pred_strict: list[str] = []
    valid_true: list[str] = []
    valid_pred: list[str] = []
    domains: list[str] = []
    error_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for row in dataset:
        example_id = str(row["example_id"])
        gold = str(row["gold_label"])
        pred_row = prediction_by_id.get(example_id)
        predicted = pred_row.get("predicted_label") if pred_row else None
        valid = bool(pred_row and pred_row.get("status") == "success" and predicted in LABELS)
        strict_pred = str(predicted) if valid else INVALID
        y_true.append(gold)
        y_pred_strict.append(strict_pred)
        domains.append(str(row.get("domain", "unknown")))
        if valid:
            valid_true.append(gold)
            valid_pred.append(str(predicted))
        else:
            missing_ids.append(example_id)
        if strict_pred != gold:
            error_rows.append(
                {
                    "example_id": example_id,
                    "context_id": row.get("context_id"),
                    "domain": row.get("domain"),
                    "gold_label": gold,
                    "predicted_label": None if strict_pred == INVALID else strict_pred,
                    "prediction_status": pred_row.get("status") if pred_row else "missing",
                    "claim": row.get("claim"),
                    "brief_rationale": pred_row.get("brief_rationale") if pred_row else None,
                }
            )

    strict = metric_block(y_true, y_pred_strict)
    valid_only = metric_block(valid_true, valid_pred)
    coverage = len(valid_true) / len(y_true)
    report = classification_report(
        y_true,
        y_pred_strict,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    class_rows: list[dict[str, Any]] = []
    for label in LABELS:
        values = report[label]
        class_rows.append(
            {
                "label": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )

    predicted_columns = LABELS + [INVALID]
    cm = confusion_matrix(y_true, y_pred_strict, labels=predicted_columns)
    cm_rows: list[dict[str, Any]] = []
    for row_index, true_label in enumerate(predicted_columns):
        cm_rows.append(
            {"true_label": true_label, **{pred: int(cm[row_index][col]) for col, pred in enumerate(predicted_columns)}}
        )

    domain_metrics: dict[str, Any] = {}
    domain_indices: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(domains):
        domain_indices[domain].append(index)
    for domain, indices in sorted(domain_indices.items()):
        domain_metrics[domain] = metric_block(
            [y_true[index] for index in indices],
            [y_pred_strict[index] for index in indices],
        )

    pair_confusions = {
        "contradicted_to_unverifiable": sum(t == "contradicted" and p == "unverifiable" for t, p in zip(y_true, y_pred_strict)),
        "unverifiable_to_contradicted": sum(t == "unverifiable" and p == "contradicted" for t, p in zip(y_true, y_pred_strict)),
        "supported_to_partially_supported": sum(t == "supported" and p == "partially_supported" for t, p in zip(y_true, y_pred_strict)),
        "partially_supported_to_supported": sum(t == "partially_supported" and p == "supported" for t, p in zip(y_true, y_pred_strict)),
    }

    first_success = next((row for row in predictions if row.get("status") == "success"), {})
    summary = {
        "created_at_utc": utc_now_iso(),
        "dataset": str(args.dataset),
        "predictions": str(args.predictions),
        "requested_model": first_success.get("requested_model"),
        "resolved_model": first_success.get("resolved_model"),
        "model_alias": first_success.get("model_alias"),
        "prompt_mode": first_success.get("prompt_mode"),
        "examples": len(y_true),
        "valid_predictions": len(valid_true),
        "invalid_or_missing_predictions": len(y_true) - len(valid_true),
        "coverage": coverage,
        "invalid_rate": 1.0 - coverage,
        "strict_metrics": strict,
        "valid_only_metrics": valid_only,
        "class_metrics": {row["label"]: {k: v for k, v in row.items() if k != "label"} for row in class_rows},
        "domain_metrics": domain_metrics,
        "pair_confusions": pair_confusions,
        "gold_distribution": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred_strict)),
        "missing_or_invalid_example_ids": missing_ids,
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "class_metrics.csv", ["label", "precision", "recall", "f1", "support"], class_rows)
    write_csv(output_dir / "confusion_matrix.csv", ["true_label", *predicted_columns], cm_rows)
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "model": summary["model_alias"] or summary["requested_model"],
        "prompt_mode": summary["prompt_mode"],
        "coverage": summary["coverage"],
        "strict_macro_f1": summary["strict_metrics"]["macro_f1"],
        "strict_accuracy": summary["strict_metrics"]["accuracy"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
