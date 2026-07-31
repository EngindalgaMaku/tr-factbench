"""Generate the final H1/H2 dropout comparison artifacts.

H1 and H2 use the same seed-fixed training pipeline. The only intended
configuration difference is LoRA dropout (0.10 vs 0.00). Historical G1 results
are retained separately and are not mixed into this comparison.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "results/csv/metrics_long_canonical.csv"
TABLE_DIR = ROOT / "results/tables"
STAT_DIR = ROOT / "results/statistics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

SERIES = {
    "H1_DROP10": {
        "dropout": 0.10,
        "run_prefix": "H1_DROP10_S",
        "seeds": [42, 43, 44, 45, 46],
        "selected": True,
    },
    "H2_DROP0": {
        "dropout": 0.00,
        "run_prefix": "H2_DROP0_S",
        "seeds": [42, 43, 44, 45, 46],
        "selected": False,
    },
}

rows = list(csv.DictReader(METRICS_PATH.open(encoding="utf-8-sig", newline="")))
lookup = {(r["run_id"], r["eval_set"]): float(r["macro_f1"]) for r in rows}


def summarize(prefix: str, seeds: list[int], eval_set: str) -> dict:
    values = [lookup[(f"{prefix}{seed}", eval_set)] for seed in seeds]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "worst_seed": seeds[values.index(min(values))],
        "best_seed": seeds[values.index(max(values))],
    }


summary = {}
for name, cfg in SERIES.items():
    d960 = summarize(cfg["run_prefix"], cfg["seeds"], "DEV960-GEN")
    d600 = summarize(cfg["run_prefix"], cfg["seeds"], "DEV600-STR")
    per_seed = {}
    for seed, v960, v600 in zip(cfg["seeds"], d960["values"], d600["values"]):
        per_seed[str(seed)] = {
            "dev960_macro_f1": v960,
            "dev600_macro_f1": v600,
            "two_set_average": (v960 + v600) / 2,
        }
    summary[name] = {
        "dropout": cfg["dropout"],
        "seeds": cfg["seeds"],
        "selected": cfg["selected"],
        "dev960": d960,
        "dev600": d600,
        "two_set_mean": (d960["mean"] + d600["mean"]) / 2,
        "per_seed": per_seed,
    }

h1 = summary["H1_DROP10"]
h2 = summary["H2_DROP0"]
summary["selection"] = {
    "selected_series": "H1_DROP10",
    "selected_checkpoint": "H1_DROP10_S45",
    "rationale": (
        "H2/dropout=0.00 has the higher DEV960 mean, while H1/dropout=0.10 "
        "has the higher DEV600 stress mean, lower DEV600 standard deviation, and "
        "a slightly higher two-set mean. Selection is based on configuration-level "
        "five-seed behavior rather than the best single seed."
    ),
    "dev960_mean_difference_h1_minus_h2": h1["dev960"]["mean"] - h2["dev960"]["mean"],
    "dev600_mean_difference_h1_minus_h2": h1["dev600"]["mean"] - h2["dev600"]["mean"],
    "two_set_mean_difference_h1_minus_h2": h1["two_set_mean"] - h2["two_set_mean"],
}

(STAT_DIR / "final_dropout_comparison.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

csv_path = TABLE_DIR / "table_H_final_dropout_comparison.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow([
        "series", "dropout", "seeds", "dev960_mean", "dev960_sd", "dev960_min", "dev960_max",
        "dev600_mean", "dev600_sd", "dev600_min", "dev600_max", "two_set_mean", "selected", "interpretation"
    ])
    writer.writerow([
        "H1", "0.10", "42-46", f"{h1['dev960']['mean']:.4f}", f"{h1['dev960']['sd']:.4f}",
        f"{h1['dev960']['min']:.4f}", f"{h1['dev960']['max']:.4f}", f"{h1['dev600']['mean']:.4f}",
        f"{h1['dev600']['sd']:.4f}", f"{h1['dev600']['min']:.4f}", f"{h1['dev600']['max']:.4f}",
        f"{h1['two_set_mean']:.4f}", "yes", "Higher and more stable stress-set performance; slightly higher two-set mean"
    ])
    writer.writerow([
        "H2", "0.00", "42-46", f"{h2['dev960']['mean']:.4f}", f"{h2['dev960']['sd']:.4f}",
        f"{h2['dev960']['min']:.4f}", f"{h2['dev960']['max']:.4f}", f"{h2['dev600']['mean']:.4f}",
        f"{h2['dev600']['sd']:.4f}", f"{h2['dev600']['min']:.4f}", f"{h2['dev600']['max']:.4f}",
        f"{h2['two_set_mean']:.4f}", "no", "Higher general-set performance; larger stress-set variance"
    ])

md_path = TABLE_DIR / "table_H_final_dropout_comparison.md"
md_path.write_text(f"""# Table H — Final seed-fixed dropout comparison

H1 and H2 use the same seed-fixed training pipeline and seeds 42–46. The intended configuration difference is LoRA dropout.

| Series | Dropout | Seeds | DEV960 mean ± SD | DEV600 mean ± SD | Two-set mean | Selection |
|---|---:|---:|---:|---:|---:|---|
| **H1** | **0.10** | **42–46** | {h1['dev960']['mean']:.4f} ± {h1['dev960']['sd']:.4f} | **{h1['dev600']['mean']:.4f} ± {h1['dev600']['sd']:.4f}** | **{h1['two_set_mean']:.4f}** | **Selected** |
| H2 | 0.00 | 42–46 | **{h2['dev960']['mean']:.4f} ± {h2['dev960']['sd']:.4f}** | {h2['dev600']['mean']:.4f} ± {h2['dev600']['sd']:.4f} | {h2['two_set_mean']:.4f} | Not selected |

H2/dropout=0.00 is stronger on DEV960. H1/dropout=0.10 is stronger and more stable on DEV600-STR and has a slightly higher two-set mean. The selected configuration is therefore H1/dropout=0.10. The decision is based on five-seed configuration-level behavior, not the best individual seed. Although `H2_DROP0_S45` is the strongest single H2 checkpoint, it does not replace the multi-seed configuration-selection criterion.

## Per-seed results

| Seed | H1 DEV960 | H1 DEV600 | H2 DEV960 | H2 DEV600 |
|---:|---:|---:|---:|---:|
""" + "\n".join(
    f"| {seed} | {h1['per_seed'][str(seed)]['dev960_macro_f1']:.4f} | {h1['per_seed'][str(seed)]['dev600_macro_f1']:.4f} | {h2['per_seed'][str(seed)]['dev960_macro_f1']:.4f} | {h2['per_seed'][str(seed)]['dev600_macro_f1']:.4f} |"
    for seed in SERIES["H1_DROP10"]["seeds"]
) + "\n", encoding="utf-8")

print(f"[OK] {csv_path}")
print(f"[OK] {md_path}")
print(f"[OK] {STAT_DIR / 'final_dropout_comparison.json'}")
