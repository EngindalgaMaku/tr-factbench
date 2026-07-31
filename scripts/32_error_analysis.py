"""
scripts/32_error_analysis.py

H5: Detaylı Hata Analizi
- Seçili checkpoint (H1_DROP10_S45) üzerinden
- Confusion matrix (DEV960-GEN, DEV600-STR)
- Per-class F1 heatmap (tüm başarılı D1 modelleri)
- Per-domain error rate
- Claim uzunluğu vs hata ilişkisi
- En çok yanılan örnek tipleri
- Çıktı: error_analysis.json, table_h5_error_breakdown.csv,
         fig_h5_confusion_matrix.png, fig_h5_per_class_f1.png,
         fig_h5_error_by_length.png
"""
import sys, json, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/supplementary"

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}
LABEL_SHORT = {"supported": "SUP", "partially_supported": "PART",
               "contradicted": "CONT", "unverifiable": "UNV"}

BEST_RUN = "H1_DROP10_S45"

# D1 başarılı modeller — per-class F1 heatmap için
D1_MODELS = {
    "ELECTRA_LoRA_S42": "D1_M04_ELECTRA_LORA16_S42",
    "ELECTRA_LoRA_S43": "D1_M04_ELECTRA_LORA16_S43",
    "ELECTRA_LoRA_S44": "D1_M04_ELECTRA_LORA16_S44",
    "ELECTRA_LoRA_S45": "D1_M04_ELECTRA_LORA16_S45",
    "ELECTRA_Freeze50_S42": "D1_M04_ELECTRA_FREEZE50_S42",
    "ELECTRA_Freeze50_S43": "D1_M04_ELECTRA_FREEZE50_S43",
    "ELECTRA_Freeze50_S44": "D1_M04_ELECTRA_FREEZE50_S44",
    "ELECTRA_Freeze50_S45": "D1_M04_ELECTRA_FREEZE50_S45",
    "ELECTRA_Freeze50_S46": "D1_M04_ELECTRA_FREEZE50_S46",
    "BERTurk_FullFT_S42":  "D1_M01_BERT_FULL_FT_LR2E5_S42",
}

print("=" * 60)
print("H5: Detaylı Hata Analizi")
print("=" * 60)

def load_preds(run_id, dev_set="DEV960-GEN"):
    p = RUNS_DIR / run_id / f"predictions_{dev_set}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def get_gold_pred(preds):
    gold, pred = [], []
    for r in preds:
        for key, lst in (("gold_label", gold), ("predicted_label", pred)):
            lbl = r.get(key, "")
            if lbl not in LABEL2ID:
                raise ValueError(f"Record {r.get('example_id','?')}: unknown {key} '{lbl}'")
            lst.append(LABEL2ID[lbl])
    return gold, pred

# ── 1. Seçili checkpoint: confusion matrix + per-class ──────────────────
print(f"\n[1] Seçili checkpoint: {BEST_RUN}")
preds_d960 = load_preds(BEST_RUN, "DEV960-GEN")
preds_d600 = load_preds(BEST_RUN, "DEV600-STR")

cm_d960 = cm_d600 = None
rep_d960 = rep_d600 = None

if preds_d960:
    gold, pred = get_gold_pred(preds_d960)
    cm_d960 = confusion_matrix(gold, pred, labels=list(range(4))).tolist()
    rep_d960 = classification_report(gold, pred, labels=list(range(4)),
                                     target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    print(f"  DEV960-GEN: n={len(preds_d960)}, acc={sum(g==p for g,p in zip(gold,pred))/len(gold):.4f}")
    print(f"  Per-class F1: " + ", ".join(f"{c}={rep_d960[c]['f1-score']:.3f}" for c in LABEL_NAMES))

if preds_d600:
    gold, pred = get_gold_pred(preds_d600)
    cm_d600 = confusion_matrix(gold, pred, labels=list(range(4))).tolist()
    rep_d600 = classification_report(gold, pred, labels=list(range(4)),
                                     target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    print(f"  DEV600-STR: n={len(preds_d600)}, acc={sum(g==p for g,p in zip(gold,pred))/len(gold):.4f}")
    print(f"  Per-class F1: " + ", ".join(f"{c}={rep_d600[c]['f1-score']:.3f}" for c in LABEL_NAMES))

# ── 2. Per-domain error (DEV960) ─────────────────────────────────────
print(f"\n[2] Per-domain error (DEV960-GEN)")
domain_results = {}
if preds_d960:
    domain_preds = defaultdict(list)
    for r in preds_d960:
        dom = r.get("domain", "unknown")
        domain_preds[dom].append(r)

    for dom, rows in sorted(domain_preds.items()):
        g, p = get_gold_pred(rows)
        acc = sum(gg==pp for gg,pp in zip(g,p)) / len(g)
        f1 = f1_score(g, p, average="macro", zero_division=0)
        domain_results[dom] = {
            "n": len(rows),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "per_class_f1": {LABEL_NAMES[i]: round(f1_score(g, p, labels=[i], average="macro",
                                                              zero_division=0), 4)
                              for i in range(4)}
        }
        print(f"  {dom}: n={len(rows)}, F1={f1:.4f}")

# ── 3. Claim uzunluğu vs hata ─────────────────────────────────────
print(f"\n[3] Claim uzunluğu vs hata analizi (DEV960-GEN, selected checkpoint)")
length_error = {}
if preds_d960:
    bins = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 9999)]
    bin_labels = ["<20", "20-40", "40-60", "60-100", ">100"]
    for (lo, hi), bl in zip(bins, bin_labels):
        subset = [r for r in preds_d960
                  if lo <= len(r.get("claim","").split()) < hi]
        if subset:
            g, p = get_gold_pred(subset)
            err = sum(gg != pp for gg, pp in zip(g, p)) / len(g)
            f1 = f1_score(g, p, average="macro", zero_division=0)
            length_error[bl] = {"n": len(subset), "error_rate": round(err, 4), "macro_f1": round(f1, 4)}
            print(f"  Claim words {bl}: n={len(subset)}, error_rate={err:.3f}, F1={f1:.4f}")
        else:
            print(f"  Claim words {bl}: no examples")

# ── 4. En çok yanılan gold → predicted çiftleri ─────────────────────
print(f"\n[4] En sık yanlış sınıflandırma çiftleri (DEV960-GEN, selected checkpoint)")
error_pairs = Counter()
if preds_d960:
    for r in preds_d960:
        if r.get("gold_label") != r.get("predicted_label"):
            error_pairs[(r["gold_label"], r["predicted_label"])] += 1

    for (gold, pred), count in error_pairs.most_common(8):
        print(f"  gold={gold} → pred={pred}: {count} örnek")

# ── 5. Per-class F1 — tüm D1 modelleri ──────────────────────────────
print(f"\n[5] D1 modelleri per-class F1 karşılaştırması (DEV960-GEN)")
d1_perclass = {}
for label, run_id in D1_MODELS.items():
    preds = load_preds(run_id, "DEV960-GEN")
    if preds:
        g, p = get_gold_pred(preds)
        rep = classification_report(g, p, labels=list(range(4)),
                                    target_names=LABEL_NAMES, output_dict=True, zero_division=0)
        d1_perclass[label] = {c: round(rep[c]["f1-score"], 4) for c in LABEL_NAMES}
        print(f"  {label}: " + " | ".join(f"{c[:4]}={rep[c]['f1-score']:.3f}" for c in LABEL_NAMES))
    else:
        print(f"  {label}: predictions bulunamadı")

# ── JSON çıktısı ─────────────────────────────────────────────────────
output = {
    "best_run": BEST_RUN,
    "confusion_matrix_dev960": cm_d960,
    "confusion_matrix_dev600": cm_d600,
    "per_class_f1_dev960": {c: round(rep_d960[c]["f1-score"], 4) for c in LABEL_NAMES} if rep_d960 else None,
    "per_class_f1_dev600": {c: round(rep_d600[c]["f1-score"], 4) for c in LABEL_NAMES} if rep_d600 else None,
    "per_domain_results": domain_results,
    "claim_length_vs_error": length_error,
    "top_error_pairs": [{"gold": k[0], "predicted": k[1], "count": v}
                        for k, v in error_pairs.most_common(10)],
    "d1_models_per_class_f1": d1_perclass,
    "interpretation": {
        "hardest_class": min(rep_d960[c]["f1-score"] for c in LABEL_NAMES) if rep_d960 else None,
        "hardest_class_name": min(LABEL_NAMES, key=lambda c: (rep_d960 or {}).get(c, {}).get("f1-score", 1)) if rep_d960 else None,
        "easiest_class_name": max(LABEL_NAMES, key=lambda c: (rep_d960 or {}).get(c, {}).get("f1-score", 0)) if rep_d960 else None,
    }
}
json.dump(output, open(STAT_DIR/"error_analysis.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\n  [OK] error_analysis.json")

# CSV tablo
csv_rows = []
for (gold, pred), count in error_pairs.most_common():
    csv_rows.append({"gold_label": gold, "predicted_label": pred, "count": count})
with open(TABLE_DIR/"table_h5_error_pairs.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["gold_label","predicted_label","count"], lineterminator="\n")
    w.writeheader()
    w.writerows(csv_rows)

# Per-class breakdown CSV
pc_rows = []
for label, vals in d1_perclass.items():
    row = {"model": label}
    row.update(vals)
    pc_rows.append(row)
with open(TABLE_DIR/"table_h5_per_class_f1.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model"]+LABEL_NAMES, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(pc_rows)
print(f"  [CSV] table_h5_error_pairs.csv, table_h5_per_class_f1.csv")

# ── Figürler ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fig 1: Confusion matrix (DEV960-GEN, selected checkpoint)
    if cm_d960:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        short = [LABEL_SHORT[l] for l in LABEL_NAMES]

        for ax, (cm, title) in zip(axes, [
            (cm_d960, f"Confusion Matrix — DEV960-GEN\n({BEST_RUN[:30]})"),
            (cm_d600, f"Confusion Matrix — DEV600-STR\n({BEST_RUN[:30]})") if cm_d600 else (None, "")
        ]):
            if cm is None:
                ax.set_visible(False)
                continue
            cm_arr = np.array(cm)
            cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
            annot = np.array([[f"{cm_arr[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                               for j in range(4)] for i in range(4)])
            sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                        xticklabels=short, yticklabels=short,
                        ax=ax, vmin=0, vmax=1,
                        cbar_kws={"label": "Normalized"})
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")
            ax.set_title(title, fontsize=9)

        plt.tight_layout()
        plt.savefig(FIG_DIR/"fig_h5_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] fig_h5_confusion_matrix.png")

    # Fig 2: Per-class F1 heatmap (D1 modelleri)
    if d1_perclass:
        model_labels = list(d1_perclass.keys())
        mat = np.array([[d1_perclass[m].get(c, 0) for c in LABEL_NAMES] for m in model_labels])
        fig, ax = plt.subplots(figsize=(9, max(4, len(model_labels) * 0.55)))
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[LABEL_SHORT[l] for l in LABEL_NAMES],
                    yticklabels=model_labels,
                    ax=ax, vmin=0.5, vmax=1.0,
                    cbar_kws={"label": "F1"})
        ax.set_title("Per-Class F1 — D1 Modelleri (DEV960-GEN)", fontsize=10)
        ax.set_xlabel("Class")
        ax.set_ylabel("Model (seed)")
        plt.tight_layout()
        plt.savefig(FIG_DIR/"fig_h5_per_class_f1_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] fig_h5_per_class_f1_heatmap.png")

    # Fig 3: Claim uzunluğu vs error rate
    if length_error:
        bins_x = list(length_error.keys())
        errs = [length_error[b]["error_rate"] * 100 for b in bins_x]
        ns = [length_error[b]["n"] for b in bins_x]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(bins_x, errs, color="#EF5350", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Claim Uzunluğu (kelime sayısı)")
        ax.set_ylabel("Hata Oranı (%)")
        ax.set_title(f"Claim Uzunluğu vs Hata Oranı\n({BEST_RUN}, DEV960-GEN)")
        for bar, n, err in zip(bars, ns, errs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{err:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(errs) * 1.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR/"fig_h5_error_by_length.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] fig_h5_error_by_length.png")

    # Fig 4: Per-domain error bar
    if domain_results:
        doms = sorted(domain_results.keys())
        f1s = [domain_results[d]["macro_f1"] * 100 for d in doms]
        dom_colors = {"medical": "#EF5350", "finance": "#42A5F5", "legal": "#66BB6A"}

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(doms, f1s,
                      color=[dom_colors.get(d, "#999") for d in doms],
                      edgecolor="white", alpha=0.85)
        ax.set_ylim(min(f1s) - 2, 100)
        ax.set_ylabel("Macro F1 (%)")
        ax.set_title(f"Per-Domain F1 — DEV960-GEN\n({BEST_RUN})")
        for bar, f1, d in zip(bars, f1s, doms):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{f1:.2f}%\n(n={domain_results[d]['n']})",
                    ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR/"fig_h5_per_domain_f1.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] fig_h5_per_domain_f1.png")

except Exception as e:
    print(f"  [WARN] Figür hatası: {e}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] 32_error_analysis.py — H5")
print("=" * 60)