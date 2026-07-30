"""
scripts/21_a2_data_effect_analysis.py

A2 - Mixed ve Full Veri Etkisi Analizi (Protokol Bölüm 7.7)
A1 koşularından türetilir; ek eğitim gerektirmez.

Karşılaştırmalar:
  1. TMIX-1600 vs TMED/TFIN/TLEG-1600: Alan çeşitliliği etkisi (aynı boyut, farklı içerik)
  2. TALL-4800 vs TMIX-1600: Veri ölçeği ve bağlam kapsama etkisi
  3. TALL vs en iyi tek-alan modeli: Çok alanlı üst sınır

Çıktılar (results/):
  tables/table_F_a2_data_effect.md
  tables/table_F_a2_data_effect.csv
  figures/fig_08_a2_data_effect_bar.png
  statistics/a2_analysis.json
"""
import sys, json, csv
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
CSV_DIR   = Q1_DIR / "results/csv"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/supplementary"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("A2 — Mixed ve Full Veri Etkisi Analizi")
print("=" * 60)

# ── Metrik CSV: 8 eval setli yeni versiyon kullanılır ───────────────
def load_metrics():
    rows = {}
    path = CSV_DIR / "metrics_long_canonical.csv"
    print(f"  Metrik kaynağı: {path.name}")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r["run_id"]
            if rid not in rows:
                rows[rid] = {}
            rows[rid][r["eval_set"]] = r
    return rows

metrics = load_metrics()

# Başarılı/collapse durumunu metrics.json'dan al
def get_qc_status(run_id):
    p = Q1_DIR / "runs/completed" / run_id / "metrics.json"
    if not p.exists():
        return "unknown"
    return json.load(open(p, encoding="utf-8")).get("qc_status", "unknown")

def get_f1(run_id, dev_set="DEV960-GEN"):
    try:
        return float(metrics[run_id][dev_set]["macro_f1"])
    except (KeyError, ValueError):
        return None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

# ── A1 koşu konfigürasyonları — tüm planlanan seedler ────────────
# TMED: S42, S43 (collapse), S44, S45 (ek duyarlılık)
# TFIN: S42, S43, S44
# TLEG: S42, S43, S44
# TMIX: S42, S43, S44
# TALL: B1_S42, B2_S43, B2_S44

configs = {
    "TMED": {
        "planned_runs": ["A1_M04_ELECTRA_TMED_LORA16_S42",
                         "A1_M04_ELECTRA_TMED_LORA16_S43",  # collapse
                         "A1_M04_ELECTRA_TMED_LORA16_S44"],
        "extra_runs":   ["A1_M04_ELECTRA_TMED_LORA16_S45"],  # ek duyarlılık
        "train_size": 1600,
        "label": "TMED (1600 tıp)"
    },
    "TFIN": {
        "planned_runs": ["A1_M04_ELECTRA_TFIN_LORA16_S42",
                         "A1_M04_ELECTRA_TFIN_LORA16_S43",
                         "A1_M04_ELECTRA_TFIN_LORA16_S44"],
        "extra_runs":   [],
        "train_size": 1600,
        "label": "TFIN (1600 finans)"
    },
    "TLEG": {
        "planned_runs": ["A1_M04_ELECTRA_TLEG_LORA16_S42",
                         "A1_M04_ELECTRA_TLEG_LORA16_S43",
                         "A1_M04_ELECTRA_TLEG_LORA16_S44"],
        "extra_runs":   [],
        "train_size": 1600,
        "label": "TLEG (1600 hukuk)"
    },
    "TMIX": {
        "planned_runs": ["A1_M04_ELECTRA_TMIX_LORA16_S42",
                         "A1_M04_ELECTRA_TMIX_LORA16_S43",
                         "A1_M04_ELECTRA_TMIX_LORA16_S44"],
        "extra_runs":   [],
        "train_size": 1600,
        "label": "TMIX (1600 karma)"
    },
    "TALL": {
        "planned_runs": ["B1_M04_ELECTRA_ALLLINEAR_LORA16_S42",
                         "B2_M04_ELECTRA_ALLLINEAR_LORA16_S43",
                         "B2_M04_ELECTRA_ALLLINEAR_LORA16_S44"],
        "extra_runs":   [],
        "train_size": 4800,
        "label": "TALL (4800 çok-alan)"
    },
}

# Eski "runs" anahtarını uyumluluk için ekle: yalnızca başarılı planned
for key, cfg in configs.items():
    cfg["runs"] = [r for r in cfg["planned_runs"]
                   if get_qc_status(r) == "success"]

results = {}
print("\nKoşu sonuçları (DEV960-GEN):")
for key, cfg in configs.items():
    f1s_960 = [get_f1(r, "DEV960-GEN") for r in cfg["runs"]]
    f1s_600 = [get_f1(r, "DEV600-STR") for r in cfg["runs"]]
    mu_960, sd_960 = mean_std(f1s_960)
    mu_600, sd_600 = mean_std(f1s_600)
    n_success = len(cfg["runs"])
    n_planned = len(cfg["planned_runs"])
    results[key] = {
        "label": cfg["label"],
        "train_size": cfg["train_size"],
        "f1s_960": f1s_960,
        "f1s_600": f1s_600,
        "mu_960": mu_960,
        "sd_960": sd_960,
        "mu_600": mu_600,
        "sd_600": sd_600,
        "n_success": n_success,
        "n_planned": n_planned,
        "success_label": f"{n_success}/{n_planned}",
    }
    seeds_str = " / ".join(f"{v:.4f}" if v else "—" for v in f1s_960)
    print(f"  {key}: mean={mu_960:.4f}±{sd_960:.4f}  [{seeds_str}]  ({n_success}/{n_planned} başarılı)")

# ── Karşılaştırma 1: TMIX vs tek-alan ─────────────────────────────
single_means_960 = [results[k]["mu_960"] for k in ["TMED","TFIN","TLEG"] if results[k]["mu_960"]]
mean_single_960 = np.mean(single_means_960)
tmix_960 = results["TMIX"]["mu_960"]
comp1_delta = tmix_960 - mean_single_960

print(f"\n[A2.1] Alan çeşitliliği etkisi (TMIX-1600 vs tek-alan-1600 ortalaması):")
print(f"  Tek-alan ort.={mean_single_960:.4f}, TMIX={tmix_960:.4f}, Δ={comp1_delta:+.4f}")

# ── Karşılaştırma 2: TALL-4800 vs TMIX-1600 ───────────────────────
tall_960 = results["TALL"]["mu_960"]
comp2_delta = tall_960 - tmix_960
print(f"\n[A2.2] Veri ölçeği etkisi (TALL-4800 vs TMIX-1600):")
print(f"  TMIX={tmix_960:.4f}, TALL={tall_960:.4f}, Δ={comp2_delta:+.4f}")

# ── Karşılaştırma 3: TALL vs en iyi tek-alan ──────────────────────
# TMIX karma veridir — tek-alan karşılaştırmasına DAHİL EDİLMEZ
best_single_key = max(["TMED","TFIN","TLEG"], key=lambda k: results[k]["mu_960"] or 0)
best_single_960 = results[best_single_key]["mu_960"]
comp3_delta = tall_960 - best_single_960
print(f"\n[A2.3] Çok-alan üst sınırı (TALL vs en iyi tek-alan, TMIX hariç):")
print(f"  En iyi tek-alan: {best_single_key}={best_single_960:.4f}, TALL={tall_960:.4f}, Δ={comp3_delta:+.4f}")

# ── JSON istatistik raporu ─────────────────────────────────────────
analysis = {
    "comparison_1_diversity": {
        "description": "Alan çeşitliliği etkisi: TMIX-1600 vs tek-alan-1600 ortalaması",
        "mean_single_domain_d960": round(mean_single_960, 4),
        "tmix_d960": round(tmix_960, 4),
        "delta_pp": round(comp1_delta * 100, 2),
        "interpretation": "Pozitif ise karma veri çeşitlilik avantajı sağlıyor"
    },
    "comparison_2_scale": {
        "description": "Veri ölçeği etkisi: TALL-4800 vs TMIX-1600",
        "tmix_d960": round(tmix_960, 4),
        "tall_d960": round(tall_960, 4),
        "delta_pp": round(comp2_delta * 100, 2),
        "interpretation": "Veri 3x artışının performansa katkısı"
    },
    "comparison_3_upper_bound": {
        "description": "Çok-alan üst sınırı: TALL vs en iyi tek-alan",
        "best_single_domain": best_single_key,
        "best_single_d960": round(best_single_960, 4),
        "tall_d960": round(tall_960, 4),
        "delta_pp": round(comp3_delta * 100, 2),
        "interpretation": "Çok-alan eğitiminin tek-alan üst sınırına göre avantajı"
    },
    "per_config": {
        k: {
            "mean_d960": round(v["mu_960"], 4) if v["mu_960"] else None,
            "std_d960": round(v["sd_960"], 4) if v["sd_960"] else None,
            "mean_d600": round(v["mu_600"], 4) if v["mu_600"] else None,
            "std_d600": round(v["sd_600"], 4) if v["sd_600"] else None,
            "train_size": v["train_size"]
        }
        for k, v in results.items()
    }
}

stat_path = STAT_DIR / "a2_analysis.json"
with open(stat_path, "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print(f"\n  [OK] {stat_path.name}")

# ── CSV ────────────────────────────────────────────────────────────
csv_path = TABLE_DIR / "table_F_a2_data_effect.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Config", "Eğitim Boyutu", "DEV960 Ortalama F1", "DEV960 Std", "DEV600 Ortalama F1", "DEV600 Std"])
    for k, v in results.items():
        w.writerow([
            v["label"], v["train_size"],
            f"{v['mu_960']:.4f}" if v["mu_960"] else "—",
            f"{v['sd_960']:.4f}" if v["sd_960"] else "—",
            f"{v['mu_600']:.4f}" if v["mu_600"] else "—",
            f"{v['sd_600']:.4f}" if v["sd_600"] else "—",
        ])
print(f"  [CSV] {csv_path.name}")

# ── Markdown tablosu ───────────────────────────────────────────────
md_path = TABLE_DIR / "table_F_a2_data_effect.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Table F — A2 Veri Etkisi Analizi (ELECTRA-TR, all-linear LoRA r=16)\n\n")
    f.write("Başarılı protokol seedleri üzerinde koşullu Macro F1 ortalaması ± örneklem SS.\n")
    f.write("TMED: S43 class collapse → 2/3 başarılı seed. TALL B2 koşularından (seed=42/43/44) türetilmiştir.\n\n")
    f.write("| Konfigürasyon | Eğitim Boyutu | Başarılı Seed | DEV960 F1 (mean±std) | DEV600 F1 (mean±std) |\n")
    f.write("|---|---:|:---:|---|---|\n")
    for k, v in results.items():
        d960 = f"{v['mu_960']:.4f}±{v['sd_960']:.4f}" if v["mu_960"] else "—"
        d600 = f"{v['mu_600']:.4f}±{v['sd_600']:.4f}" if v["mu_600"] else "—"
        if k == "TALL":
            d960 = f"**{d960}**"
            d600 = f"**{d600}**"
        f.write(f"| {v['label']} | {v['train_size']:,} | {v['success_label']} | {d960} | {d600} |\n")
    f.write("\n## A2 Karşılaştırma Özeti\n\n")
    f.write(f"| Karşılaştırma | Δ (pp) | Yorum |\n")
    f.write("|---|---:|---|\n")
    f.write(f"| Alan çeşitliliği: TMIX vs tek-alan ort. | {comp1_delta*100:+.1f}pp | {'Çeşitlilik faydalı' if comp1_delta > 0 else 'Tek-alan üstün'} |\n")
    f.write(f"| Veri ölçeği: TALL-4800 vs TMIX-1600 | {comp2_delta*100:+.1f}pp | 3x veri artışının etkisi |\n")
    f.write(f"| Çok-alan üst sınırı: TALL vs {best_single_key} | {comp3_delta*100:+.1f}pp | TALL'ın tek-alan üstünlüğü |\n")
print(f"  [MD]  {md_path.name}")

# ── Şekil ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [v["label"].replace(" ", "\n", 1) for v in results.values()]
    means_960 = [v["mu_960"] or 0 for v in results.values()]
    stds_960  = [v["sd_960"] or 0 for v in results.values()]
    means_600 = [v["mu_600"] or 0 for v in results.values()]
    stds_600  = [v["sd_600"] or 0 for v in results.values()]
    x = range(len(labels))
    w = 0.35

    success_labels = [v["success_label"] for v in results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - w/2 for i in x], means_960, w, yerr=stds_960, capsize=4,
                   label="DEV960-GEN", color="#4C72B0", alpha=0.9)
    bars2 = ax.bar([i + w/2 for i in x], means_600, w, yerr=stds_600, capsize=4,
                   label="DEV600-STR", color="#DD8452", alpha=0.9)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    # Başarı oranı annotasyonu — DEV960 barı üzerine
    for i, (bar, slabel) in enumerate(zip(bars1, success_labels)):
        h = bar.get_height() + (stds_960[i] if stds_960[i] else 0)
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.025,
                slabel, ha="center", va="bottom", fontsize=8,
                color="#c0392b" if slabel != f"{len(configs[list(configs.keys())[i]]['planned_runs'])}/{len(configs[list(configs.keys())[i]]['planned_runs'])}" else "#27ae60",
                fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.4, 1.10)
    ax.set_ylabel("Başarılı protokol seedleri üzerinde koşullu Macro F1 ortalaması ± örneklem SS", fontsize=10)
    ax.set_title("A2 — Veri Boyutu ve Çeşitlilik Etkisi (ELECTRA-TR LoRA r=16)", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = FIG_DIR / "fig_08_a2_data_effect_bar.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [PNG] {fig_path.name}")
except Exception as e:
    print(f"  [WARN] Figure skipped: {e}")

print("\n[TAMAMLANDI]")
