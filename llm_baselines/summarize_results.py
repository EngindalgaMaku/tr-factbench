from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leaderboard from all LLM baseline metrics.json files.")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "llm_baselines")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.results_root.glob("**/metrics/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": Path(data["dataset"]).stem,
                "model": data.get("model_alias") or data.get("requested_model"),
                "resolved_model": data.get("resolved_model"),
                "prompt_mode": data.get("prompt_mode"),
                "examples": data.get("examples"),
                "coverage": data.get("coverage"),
                "macro_f1": data.get("strict_metrics", {}).get("macro_f1"),
                "accuracy": data.get("strict_metrics", {}).get("accuracy"),
                "weighted_f1": data.get("strict_metrics", {}).get("weighted_f1"),
                "invalid_rate": data.get("invalid_rate"),
                "metrics_file": str(path),
            }
        )
    rows.sort(key=lambda row: (row["dataset"], -(row["macro_f1"] or -1), row["model"] or ""))
    output_csv = args.output_csv or args.results_root / "leaderboard.csv"
    output_md = args.output_md or args.results_root / "leaderboard.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "resolved_model", "prompt_mode", "examples", "coverage", "macro_f1", "accuracy", "weighted_f1", "invalid_rate", "metrics_file"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# LLM baseline leaderboard",
        "",
        "| Dataset | Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['prompt_mode']} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['coverage']:.4f} | {row['invalid_rate']:.4f} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_csv} and {output_md}")


if __name__ == "__main__":
    main()
