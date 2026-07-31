"""
scripts/23_d1_five_seed_report.py

D1 — Historical 5-Seed Stability Report (Protokol Bölüm 7.20)

3 konfigürasyon, 5 geçerli seed, istatistiksel özet.
Çıktılar:
  results/tables/table_G_d1_five_seed.md
  results/tables/table_G_d1_five_seed.csv
  results/figures/fig_historical_d1_five_seed_boxplot.png
  results/statistics/d1_summary.json
"""
import sys, json, csv
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
CSV_DIR   = Q1_DIR / "results/csv"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/historical"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("D1 — Historical 5-Seed Stability Report")
print("=" * 60)

def load_metrics():
    rows = {}
    path = CSV_DIR / "metrics_long_canonical.csv"
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r["run_id"]
            if rid not in rows:
                rows[rid] = {}
            rows[rid][r["eval_set"]] = r
    return rows

metrics = load_metrics()

def get_f1(run_id, dev_set="DEV960-GEN"):
    try:
        return float(metrics[run_id][dev_set]["macro_f1"])
    except (KeyError, ValueError):
        return None

def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    a = np.array(vals)
    return {
        "n": len(vals),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1) if len(vals) > 1 else 0.0),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "seeds": vals
    }

# ── Konfigürasyon 1: ELECTRA LoRA-r16 (historical B1 series) ─────────────
# Protokol seedleri: 42, 43, 44, 45, 46
# S46 collapse (degenerate_class_collapse: DEV960=10.00%) → kayıt altına alınır
# S47 ek kurtarma/duyarlılık koşusu olarak raporlanır
ELECTRA_LORA_PROTOCOL_SEEDS = {
    "S42": ("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", "success"),
    "S43": ("B2_M04_ELECTRA_ALLLINEAR_LORA16_S43", "success"),
    "S44": ("B2_M04_ELECTRA_ALLLINEAR_LORA16_S44", "success"),
    "S45": ("D1_M04_ELECTRA_LORA16_S45", "success"),
    "S46": ("D1_M04_ELECTRA_LORA16_S46", "degenerate_class_collapse"),  # 10.00%
}
ELECTRA_LORA_RESCUE_SEED = {
    "S47": ("D1_M04_ELECTRA_LORA16_S47", "success"),  # ek kurtarma koşusu
}

# Yalnızca başarılı protokol seedleri ortalamaya girer
electra_lora_valid = {k: v[0] for k, v in ELECTRA_LORA_PROTOCOL_SEEDS.items() if v[1] == "success"}
electra_lora_f1s_valid = [get_f1(r) for r in electra_lora_valid.values()]
electra_lora_stat = stats(electra_lora_f1s_valid)

# Kurtarma seed (S47) ayrı istatistik
rescue_f1 = get_f1(list(ELECTRA_LORA_RESCUE_SEED.values())[0][0])

# Uyumluluk için (tablolar)
electra_lora_runs = {**electra_lora_valid, "S47_rescue": list(ELECTRA_LORA_RESCUE_SEED.values())[0][0]}
electra_lora_f1s = electra_lora_f1s_valid + [rescue_f1]

# ── Konfigürasyon 2: ELECTRA Freeze-50 (güçlü alternatif) ─────────
# S42=C0, S43=C1, S44=C1, S45=D1, S46=D1
freeze50_runs = {
    "S42": "C0_M04_ELECTRA_FREEZE50_LR2E5_S42",
    "S43": "C1_M04_ELECTRA_FREEZE50_LR2E5_S43",
    "S44": "C1_M04_ELECTRA_FREEZE50_LR2E5_S44",
    "S45": "D1_M04_ELECTRA_FREEZE50_S45",
    "S46": "D1_M04_ELECTRA_FREEZE50_S46",
}
freeze50_f1s = [get_f1(r) for r in freeze50_runs.values()]
freeze50_stat = stats(freeze50_f1s)

# ── Konfigürasyon 3: BERTurk-32k M01 Full FT (tarihsel baseline) ──
# S42=B1 (reeval), S43-S46=D1
bert_runs = {
    "S42": "B1_M01_BERT32K_FULL_FT_S42",
    "S43": "D1_M01_BERT_FULL_FT_LR2E5_S43",
    "S44": "D1_M01_BERT_FULL_FT_LR2E5_S44",
    "S45": "D1_M01_BERT_FULL_FT_LR2E5_S45",
    "S46": "D1_M01_BERT_FULL_FT_LR2E5_S46",
}
bert_f1s = [get_f1(r) for r in bert_runs.values()]
bert_stat = stats(bert_f1s)

print("\nELECTRA LoRA-r16 (tarihsel B1 serisi — protokol seedleri 42-46):")
print(f"  Başarılı 4/5 seed: {electra_lora_f1s_valid}")
print(f"  mean(4 seed)={electra_lora_stat['mean']:.4f} ± {electra_lora_stat['std']:.4f}")
print(f"  S46: COLLAPSE (degenerate_class_collapse, DEV960=10.00%)")
print(f"  S47 kurtarma: {rescue_f1:.4f}")
print(f"  Eğitim başarı oranı: 4/5 = %80")

print("\nELECTRA Freeze-50 (alternatif):")
print(f"  Seeds: {freeze50_f1s}")
print(f"  mean={freeze50_stat['mean']:.4f} ± {freeze50_stat['std']:.4f}")

print("\nBERTurk-32k Full FT (baseline):")
print(f"  Seeds: {bert_f1s}")
print(f"  mean={bert_stat['mean']:.4f} ± {bert_stat['std']:.4f}")

# ── İstatistiksel fark özeti ───────────────────────────────────────
gap_lora_vs_freeze = electra_lora_stat["mean"] - freeze50_stat["mean"]
gap_lora_vs_bert   = electra_lora_stat["mean"] - bert_stat["mean"]
print(f"\nLoRA vs Freeze-50 gap: {gap_lora_vs_freeze*100:+.2f} pp")
print(f"LoRA vs BERTurk gap:   {gap_lora_vs_bert*100:+.2f} pp")

# ── S46 Collapse Teşhis Özeti ─────────────────────────────────────
print("\nS46 Collapse Analizi:")
print("  Loss (epoch 1 sonu): 1.3923 ≈ ln(4)=1.3863 (tamamen rastgele, hiç öğrenemedi)")
print("  Tahmin dağılımı: tüm 960 örnek → 'unverifiable' (sınıf 3)")
print("  Gradyan normu: 1-4 arası (patlama/sönümleme yok, optimizasyon takılı)")
print("  Yorum: Optimizasyon çökmesi gözlendi; kesin neden mevcut kayıtlarla belirlenemedi.")
print("  Bu, planlanan beş seedden birinde görülen bir başarısız koşudur;")
print("  nihai konfigürasyon için doğal veya sabit bir çökme oranı olarak yorumlanmamalıdır.")

# ── JSON istatistik ────────────────────────────────────────────────
summary = {
    "electra_lora_r16": {
        "label": "ELECTRA-TR LoRA r=16 all-linear (tarihsel B1 çoklu-seed serisi)",
        "protocol_seeds": "42, 43, 44, 45, 46",
        "protocol_success_rate": "4/5 (80%)",
        "collapsed_seeds": {"S46": "degenerate_class_collapse, DEV960=10.00%"},
        "valid_seeds_f1": dict(zip(electra_lora_valid.keys(), electra_lora_f1s_valid)),
        "seed_scores_by_id": dict(zip(electra_lora_valid.keys(), electra_lora_f1s_valid)),
        "rescue_seed": {"S47": round(rescue_f1, 4)},
        "mean_4valid_seeds": round(electra_lora_stat["mean"], 4),
        "std_4valid_seeds": round(electra_lora_stat["std"], 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in electra_lora_stat.items()},
        "note": "S46 önceden belirlenmiş protokol seedi, class collapse nedeniyle başarısız. S47 ek kurtarma/duyarlılık koşusu."
    },
    "electra_freeze50": {
        "label": "ELECTRA-TR Freeze-50 (güçlü alternatif)",
        "seed_scores_by_id": dict(zip(freeze50_runs.keys(), freeze50_f1s)),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in freeze50_stat.items()},
    },
    "bert_full_ft": {
        "label": "BERTurk-32k Full FT LR=2e-5 (tarihsel baseline)",
        "seed_scores_by_id": dict(zip(bert_runs.keys(), bert_f1s)),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in bert_stat.items()},
    },
    "pairwise_gap_pp": {
        "lora_vs_freeze50": round(gap_lora_vs_freeze * 100, 2),
        "lora_vs_bert_baseline": round(gap_lora_vs_bert * 100, 2),
    }
}

stat_path = STAT_DIR / "d1_summary.json"
with open(stat_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n  [OK] {stat_path.name}")

# ── CSV ────────────────────────────────────────────────────────────
csv_path = TABLE_DIR / "table_G_d1_five_seed.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["Konfigürasyon", "N", "Mean DEV960 F1", "Std", "Min", "Max"])
    for name, st in [("ELECTRA LoRA-r16", electra_lora_stat),
                     ("ELECTRA Freeze-50", freeze50_stat),
                     ("BERTurk Full FT (baseline)", bert_stat)]:
        w.writerow([name, st["n"], f"{st['mean']:.4f}", f"{st['std']:.4f}",
                    f"{st['min']:.4f}", f"{st['max']:.4f}"])
print(f"  [CSV] {csv_path.name}")

# ── Markdown ───────────────────────────────────────────────────────
md_path = TABLE_DIR / "table_G_d1_five_seed.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Table G — D1 Historical Multi-Seed Report (DEV960-GEN Macro F1)\n\n")
    f.write("Protokol Bölüm 7.20. Protokol seedleri: 42, 43, 44, 45, 46.\n\n")
    f.write("> **Not:** ELECTRA LoRA-r16 S46 seed'inde sınıf çökmesi (degenerate_class_collapse) gerçekleşti.\n")
    f.write("> Eğitim başarı oranı: **4/5 = %80**. S47 ek kurtarma/duyarlılık koşusu olarak verilmektedir.\n\n")
    
    # ELECTRA LoRA satırı: S42-S45 başarılı, S46=COLLAPSE, S47=kurtarma
    f.write("## ELECTRA LoRA-r16 (Tarihsel B1 Çoklu-Seed Serisi)\n\n")
    f.write("| Seed | DEV960 F1 | Durum |\n")
    f.write("|---|---:|---|\n")
    for k, (run_id, status) in ELECTRA_LORA_PROTOCOL_SEEDS.items():
        f1v = get_f1(run_id)
        if status == "success":
            f.write(f"| {k} | {f1v:.4f} | ✓ başarılı |\n")
        else:
            f.write(f"| {k} | 0.1000 | ✗ COLLAPSE (degenerate_class_collapse) |\n")
    f.write(f"| S47 (kurtarma) | {rescue_f1:.4f} | ✓ ek koşu |\n")
    f.write(f"\n**Başarılı 4 seed ortalaması: {electra_lora_stat['mean']:.4f} ± {electra_lora_stat['std']:.4f}**\n\n")

    f.write("## Konfigürasyon Karşılaştırması\n\n")
    f.write("| Konfigürasyon | N başarılı | Mean DEV960 F1 | Std |\n")
    f.write("|---|:---:|---:|---:|\n")
    f.write(f"| **ELECTRA LoRA-r16** *(tarihsel seri)* | 4/5 | **{electra_lora_stat['mean']:.4f}** | {electra_lora_stat['std']:.4f} |\n")
    f.write(f"| ELECTRA Freeze-50 *(alternatif)* | 5/5 | {freeze50_stat['mean']:.4f} | {freeze50_stat['std']:.4f} |\n")
    f.write(f"| BERTurk-32k Full FT *(baseline)* | 5/5 | {bert_stat['mean']:.4f} | {bert_stat['std']:.4f} |\n")
    
    f.write("\n## Pairwise Karşılaştırma (LoRA: 4-seed ort. kullanıldı)\n\n")
    f.write(f"| Karşılaştırma | Δ (pp) |\n")
    f.write("|---|---:|\n")
    f.write(f"| LoRA-r16 vs Freeze-50 | {gap_lora_vs_freeze*100:+.2f}pp |\n")
    f.write(f"| LoRA-r16 vs BERTurk baseline | {gap_lora_vs_bert*100:+.2f}pp |\n")
print(f"  [MD]  {md_path.name}")

# ── Boxplot ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [
        [v for v in electra_lora_f1s if v],
        [v for v in freeze50_f1s if v],
        [v for v in bert_f1s if v],
    ]
    labels = ["ELECTRA\nLoRA-r16", "ELECTRA\nFreeze-50", "BERTurk-32k\nFull FT"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (d, color) in enumerate(zip(data, colors)):
        x = [i + 1 + (0.08 * (j - len(d)/2)) for j in range(len(d))]
        ax.scatter(x, d, color=color, zorder=5, s=60, edgecolors="black", linewidths=0.8)
    ax.set_ylabel("DEV960-GEN Macro F1", fontsize=12)
    ax.set_title("D1 — Final Konfigürasyonların 5-Seed Dağılımı (ELECTRA-TR)", fontsize=12)
    ax.set_ylim(0.70, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_historical_d1_five_seed_boxplot.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [PNG] {fig_path.name}")
except Exception as e:
    print(f"  [WARN] Figure: {e}")

print("\n[TAMAMLANDI]")