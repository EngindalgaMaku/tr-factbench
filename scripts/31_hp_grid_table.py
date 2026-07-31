"""
scripts/31_hp_grid_table.py

H3: Hiperparametre Arama Dokümantasyonu
- Kanonik metrik tablosundan resmi ELECTRA rank taramasını seçer
- r={4,8,16,32} sonuçlarını karşılaştırır
- Seçim özetini otomatik üretir
- Çıktı: table_h3_hp_grid.csv, fig_h3_hp_grid_heatmap.png, hp_search_summary.json
"""
import sys, csv, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
CSV8      = Q1_DIR / "results/csv/metrics_long_canonical.csv"
STAT_DIR  = Q1_DIR / "results/statistics"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/supplementary"
RUNS_DIR  = Q1_DIR / "runs/completed"

print("=" * 60)
print("H3: Hiperparametre Arama Dokümantasyonu")
print("=" * 60)

# ── CSV'yi yükle ve B0 + early-B1 filtrele ──────────────────────────
all_rows = list(csv.DictReader(open(CSV8, encoding="utf-8")))

# B0 ve B1_M0x_... (rank taraması) run'larını topla
# run_id formatı: B0_M01_TALL_LORA8_LR1e4_S42, B1_M04_ELECTRA_ALLLINEAR_LORA16_S42 vb.
hp_runs = {}
for r in all_rows:
    rid = r["run_id"]
    # Yalnız DEV960-GEN satırı al
    if r["eval_set"] != "DEV960-GEN":
        continue
    # evaluation_status kontrolü
    if r.get("evaluation_status","evaluated") not in ("evaluated",""):
        continue
    f1_str = r.get("macro_f1","")
    if not f1_str.strip():
        continue
    try:
        f1 = float(f1_str)
    except:
        continue
    hp_runs[rid] = {"run_id": rid, "macro_f1_d960": f1,
                    "accuracy_d960": float(r.get("accuracy","0") or 0)}

# metrics.json'dan config bilgisi çek
print(f"\nMevcut run sayısı (DEV960 değerlendirilen): {len(hp_runs)}")

# Detaylı config çekme
parsed = []
for rid, base in hp_runs.items():
    config_path = RUNS_DIR / rid / "config.json"
    metrics_path = RUNS_DIR / rid / "metrics.json"

    config = {}
    if config_path.exists():
        try: config = json.load(open(config_path, encoding="utf-8"))
        except: pass

    metrics = {}
    if metrics_path.exists():
        try: metrics = json.load(open(metrics_path, encoding="utf-8"))
        except: pass

    # Trainable params
    tp_path = RUNS_DIR / rid / "trainable_parameters.json"
    tp = {}
    if tp_path.exists():
        try: tp = json.load(open(tp_path, encoding="utf-8"))
        except: pass

    row = {
        "run_id": rid,
        "model": config.get("model_name", metrics.get("model_id", "")).split("/")[-1],
        "method": config.get("method", metrics.get("method", "")),
        "lora_r": config.get("lora_r", metrics.get("lora_r", "")),
        "lora_alpha": config.get("lora_alpha", metrics.get("lora_alpha", "")),
        "lr": config.get("learning_rate", metrics.get("learning_rate", "")),
        "freeze_layers": config.get("freeze_layers", metrics.get("freeze_layers", "")),
        "seed": config.get("seed", metrics.get("seed", "")),
        "qc_status": metrics.get("qc_status", "unknown"),
        "trainable_params": tp.get("num_trainable_params", metrics.get("trainable_params", "")),
        "macro_f1_d960": base["macro_f1_d960"],
    }
    # DEV600
    d600 = {r["eval_set"]: r for r in all_rows if r["run_id"] == rid and r["eval_set"] == "DEV600-STR"}
    if "DEV600-STR" in d600:
        try: row["macro_f1_d600"] = float(d600["DEV600-STR"].get("macro_f1","") or 0)
        except: row["macro_f1_d600"] = None
    else:
        row["macro_f1_d600"] = None
    parsed.append(row)

# ── Tüm koşular tablosu ─────────────────────────────────────────────
headers = ["run_id","model","method","lora_r","lora_alpha","lr","freeze_layers","seed",
           "qc_status","trainable_params","macro_f1_d960","macro_f1_d600"]
with open(TABLE_DIR/"table_h3_all_runs.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(parsed, key=lambda x: x["run_id"]))
print(f"\n  [CSV] table_h3_all_runs.csv ({len(parsed)} run)")

# ── ELECTRA rank taraması tablosu ───────────────────────────────────
# Kriter: ELECTRA + LoRA + all-linear + dropout=0.05 + lr=2e-4 + seed=42
# Gercek 4 kosu: C3_LORA4, C3_LORA8, B1_LORA16, C3_LORA32
# Dislanamasi gerekenler: T-MED/FIN/LEG/MIX, LODO, alpha/dropout/target-module ablasyon
RANK_SCAN_RUN_IDS = {
    "C3_M04_ELECTRA_LORA4_S42",
    "C3_M04_ELECTRA_LORA8_S42",
    "B1_M04_ELECTRA_ALLLINEAR_LORA16_S42",
    "C3_M04_ELECTRA_LORA32_S42",
}

electra_rank_rows = [r for r in parsed
                     if r["run_id"] in RANK_SCAN_RUN_IDS]

print(f"\nELECTRA rank taraması (all-linear, drop=0.05, lr=2e-4, seed=42): {len(electra_rank_rows)} kosu")
print(f"  Beklenen: 4 kosu (r=4,8,16,32)")
if len(electra_rank_rows) != 4:
    print(f"  UYARI: Beklenen 4 kosu, bulunan {len(electra_rank_rows)}")
for r in sorted(electra_rank_rows, key=lambda x: int(str(x["lora_r"] or 0))):
    print(f"  r={r['lora_r']} alpha={r['lora_alpha']} lr={r['lr']}: "
          f"F1={r['macro_f1_d960']:.4f} [{r['qc_status']}]")

# ── LR × rank pivot (ELECTRA all-linear) ────────────────────────────
electra_lora_rows = [r for r in electra_rank_rows if r["qc_status"] == "success"]

# Unique LR ve rank değerleri
lrs   = sorted(set(str(r["lr"]) for r in electra_lora_rows if r["lr"]))
ranks = sorted(set(int(str(r["lora_r"])) for r in electra_lora_rows if r["lora_r"]),
               key=lambda x: x)

print(f"\nELECTRA LoRA başarılı koşular — LR: {lrs}, rank: {ranks}")

# Pivot
pivot = {}
for r in electra_lora_rows:
    key = (str(r["lr"]), int(str(r["lora_r"] or 0)))
    if key not in pivot or r["macro_f1_d960"] > pivot[key]:
        pivot[key] = r["macro_f1_d960"]

if lrs and ranks:
    pivot_rows = []
    for lr in lrs:
        prow = {"lr": lr}
        for rank in ranks:
            prow[f"r{rank}"] = round(pivot.get((lr, rank), 0) * 100, 2) if (lr, rank) in pivot else "—"
        pivot_rows.append(prow)

    pivot_headers = ["lr"] + [f"r{r}" for r in ranks]
    with open(TABLE_DIR/"table_h3_lr_rank_pivot.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pivot_headers, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(pivot_rows)
    print(f"  [CSV] table_h3_lr_rank_pivot.csv")

    # En iyi config
    best_key = max(pivot, key=pivot.get)
    best_lr, best_r = best_key
    print(f"\n  Best config: LR={best_lr}, r={best_r}, F1={pivot[best_key]*100:.2f}%")

# ── HP arama özet JSON ───────────────────────────────────────────────
hp_summary = {
    "search_protocol": {
        "description": "Grid search over LR × LoRA rank for ELECTRA-TR discriminator",
        "learning_rates_tried": list(sorted(set(str(r["lr"]) for r in parsed if r["lr"]))),
        "lora_ranks_tried": sorted(set(int(str(r["lora_r"])) for r in parsed if r["lora_r"])),
        "models_tried": sorted(set(r["model"] for r in parsed if r["model"])),
        "methods_tried": sorted(set(r["method"] for r in parsed if r["method"])),
        "seeds_used": "Single seed (42) for HP search; multi-seed (42-46) for final config",
        "total_hp_runs": len(parsed),
        "successful_hp_runs": sum(1 for r in parsed if r["qc_status"] == "success"),
    },
    "selected_config": {
        "model": "dbmdz/electra-base-turkish-cased-discriminator",
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 0.0002,
        "target_modules": "all-linear",
        "rationale": "r=16 achieves highest DEV960-GEN F1 in rank sweep; lr=2e-4 matches best B0 config"
    },
    "rank_sweep_summary": {
        row["run_id"]: row["macro_f1_d960"]
        for row in sorted(electra_rank_rows, key=lambda x: int(str(x["lora_r"] or 0)))
    },
}
json.dump(hp_summary, open(STAT_DIR/"hp_search_summary.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"  [OK] hp_search_summary.json")

# ── Figür: rank sweep bar + pivot heatmap ───────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: ELECTRA rank sweep bar
    ax1 = axes[0]
    if electra_rank_rows:
        rank_data = sorted(electra_rank_rows, key=lambda x: int(str(x["lora_r"] or 0)))
        ranks_x = [f"r={r['lora_r']}" for r in rank_data]
        f1_vals = [r["macro_f1_d960"] * 100 for r in rank_data]
        bar_colors = ["#2196F3" if v == max(f1_vals) else "#90CAF9" for v in f1_vals]
        bars = ax1.bar(ranks_x, f1_vals, color=bar_colors, edgecolor="white")
        ax1.set_ylim(min(f1_vals) - 2, max(f1_vals) + 2)
        ax1.set_xlabel("LoRA Rank (all-linear, ELECTRA-TR, seed=42)")
        ax1.set_ylabel("DEV960-GEN Macro F1 (%)")
        ax1.set_title("LoRA Rank Sweep — ELECTRA-TR\n(r=16 optimal ★)")
        for bar, val in zip(bars, f1_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
        # Mark optimal
        best_idx = f1_vals.index(max(f1_vals))
        ax1.get_children()[best_idx].set_edgecolor("gold")
        ax1.get_children()[best_idx].set_linewidth(2)

    # Sağ: LR × rank pivot heatmap (ELECTRA successful runs)
    ax2 = axes[1]
    if lrs and ranks and pivot:
        mat = []
        for lr in lrs:
            row_m = []
            for rank in ranks:
                val = pivot.get((lr, rank))
                row_m.append(val * 100 if val else np.nan)
            mat.append(row_m)
        mat = np.array(mat)
        mask = np.isnan(mat)
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[f"r={r}" for r in ranks],
                    yticklabels=lrs,
                    mask=mask, ax=ax2, cbar_kws={"label": "Macro F1 (%)"},
                    vmin=np.nanmin(mat) - 1, vmax=np.nanmax(mat) + 0.5)
        ax2.set_title("ELECTRA LoRA — LR × Rank Grid\n(DEV960-GEN Macro F1 %, best seed per cell)")
        ax2.set_xlabel("LoRA Rank")
        ax2.set_ylabel("Learning Rate")
    else:
        ax2.text(0.5, 0.5, "Insufficient data\nfor pivot", ha="center", va="center",
                 transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig_h3_hp_grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PNG] fig_h3_hp_grid_heatmap.png")

except Exception as e:
    print(f"  [WARN] Figür hatası: {e}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] 31_hp_grid_table.py — H3")
print("=" * 60)