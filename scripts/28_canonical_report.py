"""
scripts/28_canonical_report.py

SALT-OKUNUR rapor uretici. metrics_long_canonical.csv dosyasini DEGISTIRMEZ.
Kanonik CSV yalnizca 13_collect_results.py tarafindan yazilabilir.

Bu script:
- metrics_long_canonical.csv'yi okur (yazmaz)
- a2_analysis.json, domain_analysis.json, stress_vs_general.json olusturur
- Cross-domain matrix CSV ve PNG uretir
- A2 tablo CSV/MD uretir
- key_findings.md uretir

NOT: Bu script hicbir CSV'yi degistirmez. Sadece statistics/ ve tables/
altina JSON, MD, PNG yazar.
"""
import sys, csv, json, shutil
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
CSV_DIR   = Q1_DIR / "results/csv"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures"

CSV8     = CSV_DIR / "metrics_long_canonical.csv"   # tek kanonik kaynak
CSV_OLD  = CSV_DIR / "metrics_long_canonical.csv"   # artık aynı dosya
CSV_ARCH = CSV_DIR / "metrics_overall_ARCHIVED.csv" # eski arşiv

DEV_SIZES = {"DEV960-GEN": 960, "DEV960-MED": 320, "DEV960-FIN": 320, "DEV960-LEG": 320,
             "DEV600-STR": 600, "DEV600-MED": 200, "DEV600-FIN": 200, "DEV600-LEG": 200}

# ── Yardımcı: QC durumu ──────────────────────────────────────────────
def qc_status(run_id):
    p = RUNS_DIR / run_id / "metrics.json"
    if not p.exists(): return "unknown"
    try: return json.load(open(p, encoding="utf-8")).get("qc_status", "unknown")
    except: return "unknown"

# ── 1. Kanonik CSV'yi salt-okunur olarak yukle ──────────────────────
print("=" * 60)
print("1. metrics_long_canonical.csv okunuyor (degistirilmez)")
print("=" * 60)

if not CSV8.exists():
    raise FileNotFoundError(f"Kanonik CSV bulunamadi: {CSV8}\n"
                            "Once 13_collect_results.py calistirin.")

rows = list(csv.DictReader(open(CSV8, encoding="utf-8")))
print(f"  [OK] {CSV8.name} ({len(rows)} satir okundu)")

# Dogrulama: eval_set kolonu olmali, dev_set olmamali
col_names = [rows[0].keys() if rows else []]
if rows:
    assert "eval_set" in rows[0], "HATA: eval_set kolonu yok - schema hatasi"
    assert "dev_set" not in rows[0], "HATA: dev_set kolonu var - eski schema"

# ── 2. Kanonik metrik sözlüğü oluştur ───────────────────────────────
def load_canonical():
    """(run_id, eval_set) -> float macro_f1"""
    out = {}
    for r in csv.DictReader(open(CSV8, encoding="utf-8")):
        if r.get("macro_f1","").strip() and r.get("evaluation_status","evaluated") == "evaluated":
            try:
                out[(r["run_id"], r["eval_set"])] = float(r["macro_f1"])
            except: pass
    return out

metrics = load_canonical()

def get_f1(run_id, dev_set="DEV960-GEN"):
    return metrics.get((run_id, dev_set))

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

# ── 3. domain_analysis.json: 8-set tablosundan yeniden üret ─────────
print("\n" + "=" * 60)
print("3. domain_analysis.json yeniden üretiliyor")
print("=" * 60)

A1_CONFIGS = {
    "TMED": {
        "planned": ["A1_M04_ELECTRA_TMED_LORA16_S42",
                    "A1_M04_ELECTRA_TMED_LORA16_S43",  # collapse
                    "A1_M04_ELECTRA_TMED_LORA16_S44"],
        "extra": ["A1_M04_ELECTRA_TMED_LORA16_S45"],
    },
    "TFIN": {"planned": ["A1_M04_ELECTRA_TFIN_LORA16_S42",
                          "A1_M04_ELECTRA_TFIN_LORA16_S43",
                          "A1_M04_ELECTRA_TFIN_LORA16_S44"], "extra": []},
    "TLEG": {"planned": ["A1_M04_ELECTRA_TLEG_LORA16_S42",
                          "A1_M04_ELECTRA_TLEG_LORA16_S43",
                          "A1_M04_ELECTRA_TLEG_LORA16_S44"], "extra": []},
    "TMIX": {"planned": ["A1_M04_ELECTRA_TMIX_LORA16_S42",
                          "A1_M04_ELECTRA_TMIX_LORA16_S43",
                          "A1_M04_ELECTRA_TMIX_LORA16_S44"], "extra": []},
    "TALL": {"planned": ["B1_M04_ELECTRA_ALLLINEAR_LORA16_S42",
                          "B2_M04_ELECTRA_ALLLINEAR_LORA16_S43",
                          "B2_M04_ELECTRA_ALLLINEAR_LORA16_S44"], "extra": []},
}

A3_CONFIGS = {
    "LODO-NoFin": {"planned": ["A3_M04_ELECTRA_LODO_NOFIN_S42",
                                "A3_M04_ELECTRA_LODO_NOFIN_S43",
                                "A3_M04_ELECTRA_LODO_NOFIN_S44"]},
    "LODO-NoLeg": {"planned": ["A3_M04_ELECTRA_LODO_NOLEG_S42",  # collapse
                                "A3_M04_ELECTRA_LODO_NOLEG_S43",
                                "A3_M04_ELECTRA_LODO_NOLEG_S44"]},
    "LODO-NoMed": {"planned": ["A3_M04_ELECTRA_LODO_NOMED_S42",
                                "A3_M04_ELECTRA_LODO_NOMED_S43",
                                "A3_M04_ELECTRA_LODO_NOMED_S44"]},
}

def build_config_stats(runs_planned, runs_extra=None, dev_sets=None):
    if dev_sets is None:
        dev_sets = list(DEV_SIZES.keys())
    successful = [r for r in runs_planned if qc_status(r) == "success"]
    failed = [r for r in runs_planned if qc_status(r) != "success"]
    result = {
        "planned_runs": runs_planned,
        "successful_runs": successful,
        "failed_seed_ids": failed,
        "success_rate": f"{len(successful)}/{len(runs_planned)}",
        "eval_sets": {}
    }
    for ds in dev_sets:
        vals = [get_f1(r, ds) for r in successful]
        m, s = mean_std(vals)
        all_vals = [get_f1(r, ds) for r in runs_planned]
        result["eval_sets"][ds] = {
            # Only successful (non-collapsed) runs contribute — this IS the conditional mean.
            # "unconditional_mean" was removed: it was identical to conditional_mean
            # because collapsed runs return None and were already excluded.
            "conditional_mean_successful_runs": round(m, 4) if m else None,
            "conditional_std": round(s, 4) if s else None,
            "n_successful": len(successful),
            "n_planned": len(runs_planned),
            "values_per_planned_run": [round(v, 4) if v else None for v in all_vals],
        }
    if runs_extra:
        extra_successful = [r for r in runs_extra if qc_status(r) == "success"]
        result["extra_runs"] = runs_extra
        result["extra_successful"] = extra_successful
    return result

domain_analysis = {}
for key, cfg in {**A1_CONFIGS, **A3_CONFIGS}.items():
    domain_analysis[key] = build_config_stats(
        cfg["planned"], cfg.get("extra"), list(DEV_SIZES.keys())
    )

json.dump(domain_analysis, open(STAT_DIR/"domain_analysis.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"  [OK] domain_analysis.json ({len(domain_analysis)} konfigürasyon)")

# ── 4. stress_vs_general.json yeniden üret ──────────────────────────
print("\n" + "=" * 60)
print("4. stress_vs_general.json yeniden üretiliyor")
print("=" * 60)

svg = {}
for key, cfg in {**A1_CONFIGS, **A3_CONFIGS}.items():
    successful = [r for r in cfg["planned"] if qc_status(r) == "success"]
    d960_vals = [get_f1(r, "DEV960-GEN") for r in successful]
    d600_vals = [get_f1(r, "DEV600-STR") for r in successful]
    gaps = []
    for r in successful:
        g = get_f1(r, "DEV960-GEN")
        s = get_f1(r, "DEV600-STR")
        if g and s:
            gaps.append(round(g - s, 4))
    m960, _ = mean_std(d960_vals)
    m600, _ = mean_std(d600_vals)
    svg[key] = {
        "planned_seeds": len(cfg["planned"]),
        "successful_seeds": len(successful),
        "dev960_gen_mean": round(m960, 4) if m960 else None,
        "dev600_str_mean": round(m600, 4) if m600 else None,
        "generalization_gap": round(m960 - m600, 4) if m960 and m600 else None,
        "per_seed": {r: {
            "dev960": get_f1(r, "DEV960-GEN"),
            "dev600": get_f1(r, "DEV600-STR"),
            "gap": round((get_f1(r,"DEV960-GEN") or 0) - (get_f1(r,"DEV600-STR") or 0), 4)
                    if get_f1(r,"DEV960-GEN") and get_f1(r,"DEV600-STR") else None
        } for r in cfg["planned"]}
    }

json.dump(svg, open(STAT_DIR/"stress_vs_general.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"  [OK] stress_vs_general.json")

# ── 5. Cross-domain matrix: 3-seed ortalama ─────────────────────────
print("\n" + "=" * 60)
print("5. Cross-domain matrix (3-seed ortalama) yeniden üretiliyor")
print("=" * 60)

DOMAIN_SUBSETS_960 = {"Medical": "DEV960-MED", "Finance": "DEV960-FIN", "Legal": "DEV960-LEG"}
DOMAIN_SUBSETS_600 = {"Medical": "DEV600-MED", "Finance": "DEV600-FIN", "Legal": "DEV600-LEG"}

cross_rows = []
for train_key, cfg in A1_CONFIGS.items():
    if train_key in ("TALL",): continue
    successful = [r for r in cfg["planned"] if qc_status(r) == "success"]
    row_960 = {"Train": train_key, "n_successful": f"{len(successful)}/{len(cfg['planned'])}"}
    row_600 = {"Train": train_key, "n_successful": f"{len(successful)}/{len(cfg['planned'])}"}
    for dom, ds in DOMAIN_SUBSETS_960.items():
        vals = [get_f1(r, ds) for r in successful]
        m, s = mean_std(vals)
        row_960[dom] = f"{m*100:.2f}±{s*100:.2f}" if m else "—"
    for dom, ds in DOMAIN_SUBSETS_600.items():
        vals = [get_f1(r, ds) for r in successful]
        m, s = mean_std(vals)
        row_600[dom] = f"{m*100:.2f}±{s*100:.2f}" if m else "—"
    cross_rows.append(("DEV960", row_960))
    cross_rows.append(("DEV600", row_600))

# TALL da ekle
tall_cfg = A1_CONFIGS["TALL"]
tall_succ = [r for r in tall_cfg["planned"] if qc_status(r) == "success"]
row_t960 = {"Train": "TALL", "n_successful": f"{len(tall_succ)}/{len(tall_cfg['planned'])}"}
row_t600 = {"Train": "TALL", "n_successful": f"{len(tall_succ)}/{len(tall_cfg['planned'])}"}
for dom, ds in DOMAIN_SUBSETS_960.items():
    vals = [get_f1(r, ds) for r in tall_succ]
    m, s = mean_std(vals)
    row_t960[dom] = f"{m*100:.2f}±{s*100:.2f}" if m else "—"
for dom, ds in DOMAIN_SUBSETS_600.items():
    vals = [get_f1(r, ds) for r in tall_succ]
    m, s = mean_std(vals)
    row_t600[dom] = f"{m*100:.2f}±{s*100:.2f}" if m else "—"
cross_rows.append(("DEV960", row_t960))
cross_rows.append(("DEV600", row_t600))

# CSV yaz — ikiye böl
headers_cross = ["Train", "Medical", "Finance", "Legal", "n_successful"]
for split, suffix in [("DEV960", "dev960"), ("DEV600", "dev600")]:
    split_rows = [r for s, r in cross_rows if s == split]
    out = TABLE_DIR / f"table_E_cross_domain_{suffix}_3seed.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers_cross, extrasaction="ignore")
        w.writeheader()
        w.writerows(split_rows)
    print(f"  [CSV] {out.name}")

# ── 6. A2 tablosunu düzelt ───────────────────────────────────────────
print("\n" + "=" * 60)
print("6. A2 tablosu güncelleniyor")
print("=" * 60)

a2_rows = []
for key, cfg in A1_CONFIGS.items():
    planned = cfg["planned"]
    successful = [r for r in planned if qc_status(r) == "success"]
    extra = cfg.get("extra", [])
    extra_succ = [r for r in extra if qc_status(r) == "success"]

    d960_vals = [get_f1(r, "DEV960-GEN") for r in successful]
    d600_vals = [get_f1(r, "DEV600-STR") for r in successful]
    m960, s960 = mean_std(d960_vals)
    m600, s600 = mean_std(d600_vals)

    a2_rows.append({
        "config": key,
        "planned_seeds": len(planned),
        "successful_seeds": len(successful),
        "success_rate": f"{len(successful)}/{len(planned)}",
        "extra_seeds": len(extra),
        "extra_successful": len(extra_succ),
        "train_size": 4800 if key == "TALL" else 1600,
        "dev960_conditional_mean": round(m960, 4) if m960 else None,
        "dev960_conditional_std": round(s960, 4) if s960 else None,
        "dev600_conditional_mean": round(m600, 4) if m600 else None,
        "dev600_conditional_std": round(s600, 4) if s600 else None,
    })

# a2_analysis.json güncelle
a2_data = json.load(open(STAT_DIR/"a2_analysis.json", encoding="utf-8"))

# best_single_domain düzelt: TMIX karma değil, TFIN gerçek en iyi tek-alan
single_domain_keys = [k for k in ["TMED","TFIN","TLEG"] if k in {r["config"] for r in a2_rows}]
single_means = {}
for r in a2_rows:
    if r["config"] in single_domain_keys and r["dev960_conditional_mean"]:
        single_means[r["config"]] = r["dev960_conditional_mean"]

best_single = max(single_means, key=lambda k: single_means[k]) if single_means else "TFIN"
a2_data["best_single_domain"] = best_single
a2_data["best_single_domain_f1"] = single_means.get(best_single)
a2_data["note_tmix"] = "TMIX mixed-domain control, not single-domain; excluded from single-domain comparison"

# TMIX vs single-domain average
tmix_row = next((r for r in a2_rows if r["config"] == "TMIX"), None)
single_avg = float(np.mean(list(single_means.values()))) if single_means else None
tall_row = next((r for r in a2_rows if r["config"] == "TALL"), None)

if tmix_row and single_avg:
    a2_data["domain_diversity_gain_vs_avg"] = round(
        (tmix_row["dev960_conditional_mean"] or 0) - single_avg, 4)
if tmix_row and single_means.get(best_single):
    a2_data["domain_diversity_gain_vs_best"] = round(
        (tmix_row["dev960_conditional_mean"] or 0) - single_means[best_single], 4)
if tall_row and tmix_row:
    a2_data["scale_gain_tall_vs_tmix"] = round(
        (tall_row["dev960_conditional_mean"] or 0) - (tmix_row["dev960_conditional_mean"] or 0), 4)

a2_data["per_config"] = {r["config"]: r for r in a2_rows}
json.dump(a2_data, open(STAT_DIR/"a2_analysis.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"  [OK] a2_analysis.json (best_single_domain={best_single})")

# A2 CSV
a2_headers = ["config","planned_seeds","successful_seeds","success_rate",
              "train_size","dev960_conditional_mean","dev960_conditional_std",
              "dev600_conditional_mean","dev600_conditional_std","extra_seeds"]
with open(TABLE_DIR/"table_F_a2_data_effect.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=a2_headers, extrasaction="ignore")
    w.writeheader()
    w.writerows(a2_rows)
print(f"  [CSV] table_F_a2_data_effect.csv")

# ── 7. Key findings ownership ───────────────────────────────────────
# key_findings.md is written only by scripts/16_generate_statistics.py.
# This script intentionally does not overwrite it.
print("\n[SKIP] key_findings.md yalnız scripts/16_generate_statistics.py tarafından üretilir.")

# ── Cross-domain figürü yeniden oluştur ─────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pathlib as _pl

    FIG_DIR_EN = FIG_DIR.parent / "figures" / "en"
    FIG_DIR_TR = FIG_DIR.parent / "figures" / "tr"
    FIG_DIR_EN.mkdir(parents=True, exist_ok=True)
    FIG_DIR_TR.mkdir(parents=True, exist_ok=True)

    # 3-seed heatmap (DEV960)
    train_labels = ["TMED","TFIN","TLEG","TMIX","TALL"]
    dom_labels_en = ["Medical","Finance","Legal"]
    dom_labels_tr = ["Tıp","Finans","Hukuk"]
    dom_sets_960 = ["DEV960-MED","DEV960-FIN","DEV960-LEG"]

    mat = []
    annots = []
    for tl in train_labels:
        cfg = A1_CONFIGS[tl]
        succ = [r for r in cfg["planned"] if qc_status(r) == "success"]
        row = []
        anr = []
        for ds in dom_sets_960:
            vals = [get_f1(r, ds) for r in succ]
            m, s = mean_std(vals)
            row.append(m if m else 0)
            anr.append(f"{m*100:.1f}\n({len(vals)}/{len(cfg['planned'])})" if m else "—")
        mat.append(row)
        annots.append(anr)

    for lang, dom_labels, title, ylabel, out_dir in [
        ("en", dom_labels_en,
         "Cross-Domain Evaluation — DEV960-MED/FIN/LEG\n(3-seed conditional average, successful/planned)",
         "Training Configuration", FIG_DIR_EN),
        ("tr", dom_labels_tr,
         "Çapraz Alan Değerlendirmesi — DEV960-MED/FIN/LEG\n(3-seed koşullu ortalama, başarılı seed/planlanan)",
         "Eğitim Konfigürasyonu", FIG_DIR_TR),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(mat, annot=annots, fmt="", cmap="Blues", vmin=0.65, vmax=0.95,
                    xticklabels=dom_labels, yticklabels=train_labels, ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Macro F1"})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Eval Domain (n=320)")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        out_path = out_dir / "fig_07_cross_domain_heatmap.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PNG] {lang}/fig_07_cross_domain_heatmap.png (3-seed ortalama)")


except Exception as e:
    print(f"  [WARN] Figür üretilemedi: {e}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] 28_canonical_report.py")
print("=" * 60)