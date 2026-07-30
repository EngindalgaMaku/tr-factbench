import csv, sys, collections, statistics
sys.stdout.reconfigure(encoding="utf-8")

rows = list(csv.DictReader(open("results/csv/metrics_long_canonical.csv", encoding="utf-8")))
print("eval_set ornekleri:", sorted(set(r["eval_set"] for r in rows))[:10])

targets = [r for r in rows if r["run_id"].startswith("G1_C6_")]
by_run = collections.defaultdict(dict)
for r in targets:
    by_run[r["run_id"]][r["eval_set"]] = float(r["macro_f1"])

groups = {"DROP0": {}, "DROP5": {}, "DROP10": {}}
for rid, evals in sorted(by_run.items()):
    for key in groups:
        if key in rid:
            groups[key][rid] = evals

for drop, runs in groups.items():
    print(f"\n{drop}: {len(runs)} kosus")
    for rid, evals in sorted(runs.items()):
        seed = rid.split("_S")[-1]
        print(f"  S{seed}: {dict(sorted(evals.items()))}")
    vals_960 = [v for rid, e in runs.items() for k, v in e.items() if "960" in k and "GEN" in k]
    vals_600 = [v for rid, e in runs.items() for k, v in e.items() if "600" in k and "STR" in k]
    if vals_960:
        print(f"  AVG dev960: {statistics.mean(vals_960):.4f}  STD: {statistics.stdev(vals_960):.4f}" if len(vals_960)>1 else f"  dev960: {vals_960[0]:.4f}")
    if vals_600:
        print(f"  AVG dev600_stress: {statistics.mean(vals_600):.4f}  STD: {statistics.stdev(vals_600):.4f}" if len(vals_600)>1 else f"  dev600_stress: {vals_600[0]:.4f}")