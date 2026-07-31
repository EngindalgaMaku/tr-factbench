"""
Collect all evaluate_baseline metrics.json files and produce:
  - results/llm_baselines/leaderboard.csv
  - results/llm_baselines/leaderboard.md
  - results/llm_baselines/leaderboard.json
"""
import json
import pathlib
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).parent.parent / "results" / "llm_baselines"

MODEL_DISPLAY = {
    "llama_3_3_70b": "LLaMA 3.3 70B",
    "gpt_4_1_mini": "GPT-4.1 mini",
    "qwen2_5_72b": "Qwen2.5 72B",
    "gemma_4_26b_moe_free": "Gemma 4 26B MoE",
    "gpt_4_1": "GPT-4.1",
}

PROMPT_DISPLAY = {
    "zero_shot": "zero-shot",
    "few_shot_8": "8-shot",
}

rows = []
for metrics_file in sorted(ROOT.rglob("metrics/metrics.json")):
    # structure: ROOT / {dataset} / {model} / {prompt_mode} / metrics / metrics.json
    rel = metrics_file.relative_to(ROOT)
    parts = rel.parts  # (dataset, model, prompt_mode, 'metrics', 'metrics.json')
    if len(parts) != 5:
        continue
    dataset, model, prompt_mode = parts[0], parts[1], parts[2]
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    # Metrics are nested under "strict_metrics"
    sm = data.get("strict_metrics", {})
    rows.append({
        "dataset": dataset,
        "model": model,
        "model_display": MODEL_DISPLAY.get(model, model),
        "prompt_mode": prompt_mode,
        "prompt_display": PROMPT_DISPLAY.get(prompt_mode, prompt_mode),
        "strict_macro_f1": round(sm.get("macro_f1", data.get("strict_macro_f1", 0)), 4),
        "strict_accuracy": round(sm.get("accuracy", data.get("strict_accuracy", 0)), 4),
        "coverage": round(data.get("coverage", 0), 4),
    })

# Sort: dataset, model_display, prompt_mode
rows.sort(key=lambda r: (r["dataset"], r["model_display"], r["prompt_mode"]))

# --- CSV ---
csv_path = ROOT / "leaderboard.csv"
fieldnames = ["dataset", "model", "prompt_mode", "strict_macro_f1", "strict_accuracy", "coverage"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r[k] for k in fieldnames})
print(f"Wrote {csv_path}")

# --- JSON ---
json_path = ROOT / "leaderboard.json"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {json_path}")

# --- Markdown leaderboard ---
def md_table(dataset_rows, title):
    lines = [f"### {title}", ""]
    lines.append("| Model | Prompt | Macro-F1 | Accuracy | Coverage |")
    lines.append("|---|---|---:|---:|---:|")
    for r in dataset_rows:
        f1 = f"{r['strict_macro_f1']:.4f}"
        acc = f"{r['strict_accuracy']:.4f}"
        cov = f"{r['coverage']:.4f}"
        lines.append(f"| {r['model_display']} | {r['prompt_display']} | **{f1}** | {acc} | {cov} |")
    lines.append("")
    return "\n".join(lines)

datasets = sorted(set(r["dataset"] for r in rows))
md_sections = [
    "# TR-FactBench LLM Baseline Results",
    "",
    f"Generated from `llm_baselines/collect_results.py`.",
    "",
    "Metric: **strict_macro_f1** (coverage-penalised macro-F1 over all attempted examples).",
    "",
]
for ds in datasets:
    ds_rows = [r for r in rows if r["dataset"] == ds]
    md_sections.append(md_table(ds_rows, ds))

# Best-per-model summary across both datasets (avg of ZS + FS8)
md_sections.append("### Best per Model (avg of zero-shot and 8-shot, both datasets)")
md_sections.append("")
md_sections.append("| Model | dev_stress ZS | dev_stress FS8 | dev_general ZS | dev_general FS8 |")
md_sections.append("|---|---:|---:|---:|---:|")

def get_f1(model, dataset, prompt):
    for r in rows:
        if r["model"] == model and r["dataset"] == dataset and r["prompt_mode"] == prompt:
            return r["strict_macro_f1"]
    return None

models_order = ["llama_3_3_70b", "gpt_4_1_mini", "qwen2_5_72b", "gemma_4_26b_moe_free"]
for m in models_order:
    dszs = get_f1(m, "dev_stress", "zero_shot")
    dsfs = get_f1(m, "dev_stress", "few_shot_8")
    dgzs = get_f1(m, "dev_general", "zero_shot")
    dgfs = get_f1(m, "dev_general", "few_shot_8")
    fmt = lambda v: f"{v:.4f}" if v is not None else "—"
    md_sections.append(f"| {MODEL_DISPLAY.get(m, m)} | {fmt(dszs)} | {fmt(dsfs)} | {fmt(dgzs)} | {fmt(dgfs)} |")
md_sections.append("")

md_path = ROOT / "leaderboard.md"
md_path.write_text("\n".join(md_sections), encoding="utf-8")
print(f"Wrote {md_path}")

# Print summary to console
print("\n=== LEADERBOARD SUMMARY ===")
print(f"{'dataset':<14} {'model':<22} {'prompt':<10} {'F1':>7} {'acc':>7} {'cov':>7}")
print("-" * 75)
for r in rows:
    print(f"{r['dataset']:<14} {r['model_display']:<22} {r['prompt_display']:<10} {r['strict_macro_f1']:>7.4f} {r['strict_accuracy']:>7.4f} {r['coverage']:>7.4f}")