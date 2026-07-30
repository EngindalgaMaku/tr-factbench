import csv, sys, statistics
sys.stdout.reconfigure(encoding="utf-8")

rows = list(csv.DictReader(open("results/csv/metrics_long_canonical.csv", encoding="utf-8")))

drop10 = [r for r in rows if r["run_id"].startswith("G1_C6_M04_ELECTRA_LORA16_DROP10_")]
by_seed = {}
for r in drop10:
    sid = r["run_id"]
    if sid not in by_seed:
        by_seed[sid] = {}
    by_seed[sid][r["eval_set"]] = float(r["macro_f1"])

print("=== DROP10 — 5 seed sonuclari ===")
avgs = []
for rid, evals in sorted(by_seed.items()):
    d9 = evals.get("DEV960-GEN", None)
    d6 = evals.get("DEV600-STR", None)
    if d9 is not None and d6 is not None:
        avg = (d9 + d6) / 2
        avgs.append((rid, d9, d6, avg))
    print(f"  {rid}  dev960={d9}  dev600_stress={d6}")

print()
print("AVG siralamasi (dev960+dev600_stress / 2):")
for rid, d9, d6, avg in sorted(avgs, key=lambda x: -x[3]):
    print(f"  {rid}  AVG={avg:.4f}  (dev960={d9}  dev600={d6})")

if avgs:
    best = max(avgs, key=lambda x: x[3])
    print(f"\nNIHAI MODEL: {best[0]}")
    print(f"  dev960-GEN: {best[1]:.4f}")
    print(f"  dev600-STR: {best[2]:.4f}")
    print(f"  AVG:        {best[3]:.4f}")

# DROP0 durumu
print()
print("=== DROP0 — collapse kontrolu ===")
drop0 = [r for r in rows if r["run_id"].startswith("G1_C6_M04_ELECTRA_LORA16_DROP0_")]
by_seed0 = {}
for r in drop0:
    sid = r["run_id"]
    if sid not in by_seed0:
        by_seed0[sid] = {}
    by_seed0[sid][r["eval_set"]] = float(r["macro_f1"])
for rid, evals in sorted(by_seed0.items()):
    d9 = evals.get("DEV960-GEN", "?")
    d6 = evals.get("DEV600-STR", "?")
    collapse = "COLLAPSE" if (isinstance(d9, float) and d9 < 0.5) or (isinstance(d6, float) and d6 < 0.5) else ""
    print(f"  {rid}  dev960={d9}  dev600_stress={d6}  {collapse}")