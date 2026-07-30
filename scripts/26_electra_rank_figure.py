"""
scripts/26_electra_rank_figure.py

ELECTRA rank ablasyonu sonuçlarını fig_02b ve electra_rank_analysis.json olarak yazar.
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
FIG_DIR  = Q1_DIR / "results/figures/supplementary"
STAT_DIR = Q1_DIR / "results/statistics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

ranks = [4,     8,     16,    32]
d960  = [88.25, 90.31, 90.75, 88.66]
d600  = [85.30, 83.07, 85.78, 85.25]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ranks, d960, "o-", color="#1f77b4", linewidth=2, markersize=8, label="DEV960-GEN")
ax.plot(ranks, d600, "s--", color="#ff7f0e", linewidth=2, markersize=8, label="DEV600-STR")
ax.axvline(x=16, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax.annotate("r=16 optimal",
            xy=(16, 90.75), xytext=(18.5, 90.0),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=9)
ax.set_xlabel("LoRA Rank (r)", fontsize=11)
ax.set_ylabel("DEV Macro F1 (%)", fontsize=11)
ax.set_title("ELECTRA-TR LoRA Rank Scaling\n(all-linear, alpha=2r, seed=42)", fontsize=11)
ax.set_xticks(ranks)
ax.set_ylim(82, 93)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = FIG_DIR / "fig_02b_electra_rank_scaling.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"  [PNG] {out.name}")

# JSON
rank_data = {
    "model": "dbmdz/electra-base-turkish-cased-discriminator",
    "method": "LoRA all-linear, alpha=2r, seed=42",
    "results": [
        {"rank": 4,  "alpha": 8,  "dev960": 88.25, "dev600": 85.30},
        {"rank": 8,  "alpha": 16, "dev960": 90.31, "dev600": 83.07},
        {"rank": 16, "alpha": 32, "dev960": 90.75, "dev600": 85.78, "optimal": True},
        {"rank": 32, "alpha": 64, "dev960": 88.66, "dev600": 85.25},
    ],
    "optimal_rank": 16,
    "finding": (
        "r=16 iki kumede de en yuksek Macro F1 (%90.75 DEV960, %85.78 DEV600). "
        "r=4->r=8 +2.06pp kazanc (kapasite yetersizligi). "
        "r=8->r=16 +0.44pp ek kazanc (optimal nokta). "
        "r=16->r=32 -2.09pp dusus: ELECTRA mimarisi icin r=16 optimal rank."
    ),
}
stat_out = STAT_DIR / "electra_rank_analysis.json"
with open(stat_out, "w", encoding="utf-8") as f:
    json.dump(rank_data, f, indent=2, ensure_ascii=False)
print(f"  [OK]  {stat_out.name}")

print("\n[TAMAMLANDI]")
