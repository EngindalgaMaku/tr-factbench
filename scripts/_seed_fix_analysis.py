import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("runs/completed")
seeds = [42, 43, 44, 45, 46]

print(f"{'Seed':<5} {'H1_D960':>8} {'H1_D600':>8} {'H1_AVG':>8}  |  {'G1_D960':>8} {'G1_D600':>8} {'G1_AVG':>8}  |  {'delta_960':>10} {'delta_600':>10}")
print("-" * 100)
h_avgs_960, h_avgs_600 = [], []
for s in seeds:
    h = json.loads((ROOT / f"H1_DROP10_S{s}/metrics.json").read_text(encoding="utf-8"))
    g = json.loads((ROOT / f"G1_C6_M04_ELECTRA_LORA16_DROP10_S{s}/metrics.json").read_text(encoding="utf-8"))
    h960, h600 = h["d960_macro_f1"], h["d600_macro_f1"]
    g960, g600 = g["d960_macro_f1"], g["d600_macro_f1"]
    havg = (h960 + h600) / 2
    gavg = (g960 + g600) / 2
    h_avgs_960.append(h960); h_avgs_600.append(h600)
    print(f"S{s:<4} {h960:>8.4f} {h600:>8.4f} {havg:>8.4f}  |  {g960:>8.4f} {g600:>8.4f} {gavg:>8.4f}  |  {h960-g960:>+10.4f} {h600-g600:>+10.4f}")

print("-" * 100)
mean960 = sum(h_avgs_960) / 5
mean600 = sum(h_avgs_600) / 5
print(f"Mean:                                      H1 DEV960={mean960:.4f}  DEV600={mean600:.4f}")

best_s = max(seeds, key=lambda s: (
    json.loads((ROOT / f"H1_DROP10_S{s}/metrics.json").read_text(encoding="utf-8"))["d960_macro_f1"] +
    json.loads((ROOT / f"H1_DROP10_S{s}/metrics.json").read_text(encoding="utf-8"))["d600_macro_f1"]
))
best = json.loads((ROOT / f"H1_DROP10_S{best_s}/metrics.json").read_text(encoding="utf-8"))
print(f"Best seed: {best_s}  DEV960={best['d960_macro_f1']:.4f}  DEV600={best['d600_macro_f1']:.4f}  AVG={(best['d960_macro_f1']+best['d600_macro_f1'])/2:.4f}")