"""
scripts/30_collapse_analysis.py

Historical S46 Collapse Analysis
- ELECTRA LoRA S46 vs S42 karşılaştırması
- Predicted class dağılımı (collapse kanıtı)
- trainer_state.json eğitim eğrisi analizi
- Confusion matrix karşılaştırması
- Hipotez: S46'da gradient vanishing veya şanssız init → tek sınıfa collapse

Çıktı:
  results/statistics/collapse_analysis.json
  results/tables/table_collapse_class_dist.csv
  results/figures/fig_historical_s46_collapse_analysis.png
"""
import sys, json, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/historical"

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}

COLLAPSE_RUN  = "D1_M04_ELECTRA_LORA16_S46"
HEALTHY_RUNS  = {
    "S42": "D1_M04_ELECTRA_LORA16_S42",
    "S43": "D1_M04_ELECTRA_LORA16_S43",
    "S44": "D1_M04_ELECTRA_LORA16_S44",
    "S45": "D1_M04_ELECTRA_LORA16_S45",
}
# Tüm D1 run'ları (BERT ve Freeze dahil)
ALL_D1_RUNS = [COLLAPSE_RUN] + list(HEALTHY_RUNS.values())
# Freeze50 S46 da inceleyelim (farklı model, collapse mi?)
FREEZE_S46 = "D1_M04_ELECTRA_FREEZE50_S46"
BERT_S46   = "D1_M01_BERT_FULL_FT_LR2E5_S46"
H2_S46     = "H2_DROP0_S46"

print("=" * 60)
print("Historical S46 Class Collapse Analysis")
print("=" * 60)

def load_predictions(run_id, dev_set="DEV960-GEN"):
    p = RUNS_DIR / run_id / f"predictions_{dev_set}.jsonl"
    if not p.exists():
        # Alternatif: ckpt içindeki predictions.jsonl
        p2 = RUNS_DIR / run_id / "ckpt"
        for pp in p2.rglob("predictions.jsonl"):
            p = pp
            break
    if not p.exists():
        return None
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except: pass
    return rows

def analyze_preds(preds):
    if not preds: return None
    gold, pred = [], []
    for r in preds:
        for key, lst in (("gold_label", gold), ("predicted_label", pred)):
            lbl = r.get(key, "")
            if lbl not in LABEL2ID:
                raise ValueError(f"Record {r.get('example_id','?')}: unknown {key} '{lbl}'")
            lst.append(LABEL2ID[lbl])
    pred_counts = Counter(pred)
    gold_counts = Counter(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    cm = confusion_matrix(gold, pred, labels=list(range(4))).tolist()
    rep = classification_report(gold, pred, labels=list(range(4)),
                                target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    return {
        "n": len(preds),
        "accuracy": round(acc, 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "pred_class_dist": {LABEL_NAMES[k]: v for k, v in sorted(pred_counts.items())},
        "gold_class_dist": {LABEL_NAMES[k]: v for k, v in sorted(gold_counts.items())},
        "pred_class_pct":  {LABEL_NAMES[k]: round(v/len(preds)*100, 1)
                            for k, v in sorted(pred_counts.items())},
        "per_class_f1": {k: round(rep[k]["f1-score"], 4) for k in LABEL_NAMES},
        "confusion_matrix": cm,
    }

def load_training_curve(run_id):
    """trainer_state.json'dan eval_macro_f1 eğrisini yükle."""
    pattern = RUNS_DIR / run_id / "ckpt"
    states = list(pattern.rglob("trainer_state.json")) if pattern.exists() else []
    if not states:
        return None
    state = json.load(open(states[0], encoding="utf-8"))
    log_history = state.get("log_history", [])
    eval_curve = [(e.get("epoch"), e.get("eval_macro_f1") or e.get("eval_loss"))
                  for e in log_history if "eval_macro_f1" in e or "eval_loss" in e]
    return {"log_history": log_history, "eval_curve": eval_curve,
            "best_metric": state.get("best_metric")}

# ── Collapse analizi ────────────────────────────────────────────────
collapse_data = {}

# S46 ELECTRA LoRA
print(f"\n[Collapse run: {COLLAPSE_RUN}]")
preds_s46 = load_predictions(COLLAPSE_RUN)
if preds_s46:
    a = analyze_preds(preds_s46)
    collapse_data[COLLAPSE_RUN] = a
    print(f"  Accuracy: {a['accuracy']} | Macro F1: {a['macro_f1']}")
    print(f"  Predicted class distribution:")
    for cls, pct in a["pred_class_pct"].items():
        bar = "█" * int(pct / 5)
        print(f"    {cls:25s}: {pct:5.1f}% {bar}")
    # Collapse tespiti: tek sınıf %80+ mı?
    max_pct = max(a["pred_class_pct"].values())
    dominant_cls = max(a["pred_class_pct"], key=lambda k: a["pred_class_pct"][k])
    is_collapse = max_pct > 70
    print(f"  → {'[COLLAPSE CONFIRMED]' if is_collapse else '[NO COLLAPSE]'}: {dominant_cls} = {max_pct:.1f}%")
    collapse_data[COLLAPSE_RUN]["is_collapse"] = is_collapse
    collapse_data[COLLAPSE_RUN]["dominant_class"] = dominant_cls
    collapse_data[COLLAPSE_RUN]["dominant_class_pct"] = max_pct
else:
    print(f"  [WARN] predictions bulunamadı")
    collapse_data[COLLAPSE_RUN] = {"status": "predictions_not_found"}

# Healthy runs karşılaştırma
print(f"\n[Healthy runs karşılaştırması]")
for seed_name, run_id in HEALTHY_RUNS.items():
    preds = load_predictions(run_id)
    if preds:
        a = analyze_preds(preds)
        collapse_data[run_id] = a
        max_pct = max(a["pred_class_pct"].values())
        print(f"  {seed_name} ({run_id}): F1={a['macro_f1']}, max_class_pct={max_pct:.1f}%")
    else:
        print(f"  {seed_name}: predictions bulunamadı")

# Freeze50 S46
print(f"\n[Freeze50 S46 — aynı seed, farklı yöntem]")
preds_freeze46 = load_predictions(FREEZE_S46)
if preds_freeze46:
    a = analyze_preds(preds_freeze46)
    collapse_data[FREEZE_S46] = a
    max_pct = max(a["pred_class_pct"].values())
    print(f"  F1={a['macro_f1']}, max_class={max(a['pred_class_pct'], key=a['pred_class_pct'].get)} ({max_pct:.1f}%)")
    print(f"  → {'[COLLAPSE]' if max_pct > 70 else '[Sağlıklı]'}")

# Seed-fixed H2 dropout=0.00 S46 rerun
print(f"\n[H2 dropout=0.00 S46 — seed-fixed current pipeline]")
preds_h2_s46 = load_predictions(H2_S46)
if preds_h2_s46:
    a = analyze_preds(preds_h2_s46)
    collapse_data[H2_S46] = a
    max_pct = max(a["pred_class_pct"].values())
    print(f"  F1={a['macro_f1']}, max_class={max(a['pred_class_pct'], key=a['pred_class_pct'].get)} ({max_pct:.1f}%)")
    print(f"  → {'[COLLAPSE]' if max_pct > 70 else '[Healthy]'}")

# ── Eğitim eğrisi ───────────────────────────────────────────────────
print(f"\n[Eğitim eğrisi: S46 vs S42]")
curves = {}
for run_id in [COLLAPSE_RUN, HEALTHY_RUNS.get("S42","")]:
    if not run_id: continue
    curve = load_training_curve(run_id)
    curves[run_id] = curve
    if curve:
        print(f"  {run_id}: best_metric={curve['best_metric']}, {len(curve['eval_curve'])} eval nokta")
    else:
        print(f"  {run_id}: trainer_state.json yok")

# ── Hipotez özeti ────────────────────────────────────────────────────
print(f"\n[Hipotez Özeti]")
s46_data = collapse_data.get(COLLAPSE_RUN, {})
freeze46_data = collapse_data.get(FREEZE_S46, {})

hypotheses = []
if s46_data.get("is_collapse"):
    dom = s46_data.get("dominant_class", "?")
    pct = s46_data.get("dominant_class_pct", 0)
    hypotheses.append(f"ELECTRA LoRA S46 collapsed to '{dom}' ({pct:.1f}% of predictions)")

if freeze46_data and freeze46_data.get("accuracy", 1) > 0.5:
    hypotheses.append("Freeze50 S46 did NOT collapse → seed=46 alone is not sufficient to cause failure")
    hypotheses.append("The exact cause of the historical LoRA collapse cannot be determined from the archived training state")
else:
    hypotheses.append("Both LoRA and Freeze50 at S46 show issues → seed=46 may have systematic instability")

if curves.get(COLLAPSE_RUN) and curves[COLLAPSE_RUN].get("eval_curve"):
    hypotheses.append("Training curve available — see eval_curve in output JSON")
else:
    hypotheses.append("trainer_state.json unavailable — cannot confirm gradient vanishing hypothesis from loss curve")

for h in hypotheses:
    print(f"  • {h}")

# ── Kaydet ──────────────────────────────────────────────────────────
output = {
    "analysis": "S46 class collapse investigation",
    "collapse_run": COLLAPSE_RUN,
    "healthy_runs": list(HEALTHY_RUNS.values()),
    "freeze50_s46": FREEZE_S46,
    "per_run": collapse_data,
    "training_curves": {k: v for k, v in curves.items() if v},
    "hypotheses": hypotheses,
    "summary": {
        "collapse_confirmed": s46_data.get("is_collapse", False),
        "collapse_dominant_class": s46_data.get("dominant_class"),
        "collapse_dominant_pct": s46_data.get("dominant_class_pct"),
        "freeze50_s46_healthy": freeze46_data.get("macro_f1", 0) > 0.5 if freeze46_data else None,
        "interpretation": (
            "A genuine class collapse occurred in the historical D1 LoRA run. Healthy Freeze-50 "
            "and seed-fixed H2 S46 results show that seed 46 alone is not a sufficient explanation. "
            "The exact cause remains undetermined because the original training state is incomplete."
        )
    }
}
json.dump(output, open(STAT_DIR/"collapse_analysis.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\n  [OK] collapse_analysis.json")

# ── CSV: class dağılımı karşılaştırması ─────────────────────────────
csv_rows = []
for run_id, data in collapse_data.items():
    if not isinstance(data, dict) or "pred_class_pct" not in data: continue
    row = {"run_id": run_id, "macro_f1": data.get("macro_f1",""),
           "is_collapse": data.get("is_collapse", False)}
    for cls in LABEL_NAMES:
        row[f"pred_{cls}_pct"] = data["pred_class_pct"].get(cls, 0)
    csv_rows.append(row)

headers = ["run_id","macro_f1","is_collapse"] + [f"pred_{c}_pct" for c in LABEL_NAMES]
with open(TABLE_DIR/"table_collapse_class_dist.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
    w.writeheader()
    w.writerows(csv_rows)
print(f"  [CSV] table_collapse_class_dist.csv")

# ── Figür ───────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Figür: Predicted class distribution comparison
    plot_runs = {COLLAPSE_RUN: "S46 [COLLAPSE]"}
    for seed_name, run_id in HEALTHY_RUNS.items():
        if run_id in collapse_data and "pred_class_pct" in collapse_data[run_id]:
            plot_runs[run_id] = f"{seed_name} (healthy)"
    if FREEZE_S46 in collapse_data and "pred_class_pct" in collapse_data[FREEZE_S46]:
        plot_runs[FREEZE_S46] = "Freeze50 S46"
    if H2_S46 in collapse_data and "pred_class_pct" in collapse_data[H2_S46]:
        plot_runs[H2_S46] = "H2 drop0 S46 (healthy rerun)"

    valid_runs = {k: v for k, v in plot_runs.items()
                  if k in collapse_data and "pred_class_pct" in collapse_data[k]}

    n_runs = len(valid_runs)
    if n_runs > 0:
        x = np.arange(len(LABEL_NAMES))
        width = 0.7 / n_runs

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Sol: bar chart — predicted distribution
        colors_map = {}
        for i, (run_id, label) in enumerate(valid_runs.items()):
            color = "red" if "COLLAPSE" in label else f"C{i}"
            colors_map[run_id] = color
            d = collapse_data[run_id]["pred_class_pct"]
            vals = [d.get(c, 0) for c in LABEL_NAMES]
            ax1.bar(x + i * width - (n_runs - 1) * width / 2, vals,
                    width, label=label, color=color, alpha=0.8)

        ax1.set_xlabel("Predicted Class")
        ax1.set_ylabel("% of Predictions")
        ax1.set_title("S46 Class Collapse: Predicted Distribution\n(sağlıklı seedler vs çöken seed)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace("_", "\n") for c in LABEL_NAMES], fontsize=8)
        ax1.legend(fontsize=7)
        ax1.axhline(25, color="gray", linestyle="--", alpha=0.5, label="Uniform (25%)")

        # Sağ: F1 per-class karşılaştırması (S46 vs S42)
        s46_f1 = collapse_data.get(COLLAPSE_RUN, {}).get("per_class_f1", {})
        s42_run = HEALTHY_RUNS.get("S42", "")
        s42_f1 = collapse_data.get(s42_run, {}).get("per_class_f1", {})

        if s46_f1 and s42_f1:
            x2 = np.arange(len(LABEL_NAMES))
            ax2.bar(x2 - 0.2, [s42_f1.get(c, 0) for c in LABEL_NAMES], 0.35,
                    label="S42 (healthy)", color="steelblue", alpha=0.85)
            ax2.bar(x2 + 0.2, [s46_f1.get(c, 0) for c in LABEL_NAMES], 0.35,
                    label="S46 (collapse)", color="red", alpha=0.85)
            ax2.set_xlabel("Class")
            ax2.set_ylabel("F1 Score")
            ax2.set_title("Per-Class F1: S42 vs S46")
            ax2.set_xticks(x2)
            ax2.set_xticklabels([c.replace("_", "\n") for c in LABEL_NAMES], fontsize=8)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "S42/S46 per-class F1\nnot available", ha="center", va="center", transform=ax2.transAxes)

        plt.tight_layout()
        plt.savefig(FIG_DIR/"fig_historical_s46_collapse_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] fig_historical_s46_collapse_analysis.png")

except Exception as e:
    print(f"  [WARN] Figür hatası: {e}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] 30_collapse_analysis.py — historical audit")
print("=" * 60)