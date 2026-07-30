import pathlib, json, sys
sys.stdout.reconfigure(encoding="utf-8")
runs_dir = pathlib.Path("runs/completed")

print(f"{'Dropout':<10} {'DEV960 mean':<14} {'+-std':<8} {'DEV600 mean':<14} Seeds")
print("-" * 65)
for dropout, tag in [(0.0,"DROP0"),(0.05,"DROP5"),(0.10,"DROP10")]:
    d960s, d600s = [], []
    for seed in [42, 43, 44]:
        name = f"G1_C6_M04_ELECTRA_LORA16_{tag}_S{seed}"
        m = json.loads((runs_dir / name / "metrics.json").read_text(encoding="utf-8"))
        d960s.append(m["d960_macro_f1"])
        d600s.append(m["d600_macro_f1"])
    mean960 = sum(d960s) / 3
    std960  = (sum((v - mean960)**2 for v in d960s) / 3) ** 0.5
    mean600 = sum(d600s) / 3
    print(f"{dropout:<10.2f} {mean960:<14.4f} {std960:<8.4f} {mean600:<14.4f} {[round(v,4) for v in d960s]}")