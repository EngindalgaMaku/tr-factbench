"""
scripts/98e_v12_diagnosis.py  (diagnostic, 2026-11-16)

All v1.1.0 runs leave the initial loss plateau (ln 4 = 1.386) after about 30-40 steps; both
v1.2.0 runs stayed on it and collapsed to one class. The v1.2.0 file passes every structural
check (same rows, ids, labels, order, no duplicates, no conflicting labels), so this script
localizes the effect by bisection over the 47 masked context groups, all with seed 42:

  DIAG_HALF1_S42  v1.1.0 text, masked text only in the first half of the 47 groups
  DIAG_HALF2_S42  v1.1.0 text, masked text only in the second half
  DIAG_V12_S44    full v1.2.0, seed 44 (third v1.2.0 seed, for the collapse rate)

Each run takes about 13 minutes. Output: runs/diag_v12/<run_id>/eval.json and a summary
results/statistics/V12_DIAGNOSIS_v1.json with the first-epoch loss curve of every run.
"""
import glob
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pr98", ROOT / "scripts" / "98_post_review_runs.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

V11 = ROOT.parent / "_private_masking_v1.2" / "v1.1_originals" / "data" / "primary" / "train.jsonl"
V12 = ROOT / "data" / "primary" / "train.jsonl"
OUT = ROOT / "runs" / "diag_v12"
SUMMARY = ROOT / "results" / "statistics" / "V12_DIAGNOSIS_v1.json"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


a, b = load(V11), load(V12)
assert [r["example_id"] for r in a] == [r["example_id"] for r in b]
changed_groups = []
for x, y in zip(a, b):
    if x != y and x["context_id"] not in changed_groups:
        changed_groups.append(x["context_id"])
half = len(changed_groups) // 2
SETS = {"DIAG_HALF1_S42": (set(changed_groups[:half]), 42),
        "DIAG_HALF2_S42": (set(changed_groups[half:]), 42)}


def build(groups, path):
    rows = [y if x["context_id"] in groups else x for x, y in zip(a, b)]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def loss_curve(run_dir):
    ps = sorted(glob.glob(str(run_dir / "ckpt" / "**" / "trainer_state.json"), recursive=True))
    if not ps:
        return None
    h = json.load(open(ps[-1]))["log_history"]
    return {"loss_every_8_steps": [round(x["loss"], 3) for x in h if "loss" in x][:80:8],
            "eval_macro_f1": [round(x["eval_macro_f1"], 3) for x in h if "eval_macro_f1" in x]}


def run(run_id, train_path, seed):
    run_dir = OUT / run_id
    if (run_dir / "eval.json").exists():
        print(f"skip {run_id}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    pr.OUT_DIR = OUT
    pr.TRAIN = train_path
    try:
        pr.best_model_dir(run_dir)
    except FileNotFoundError:
        pr.train(run_id, "full_input", seed)
    ev = pr.evaluate(run_id, "full_input")
    (run_dir / "eval.json").write_text(json.dumps({"run_id": run_id, "train_file": str(train_path), "seed": seed,
                                                   "eval": ev}, ensure_ascii=False, indent=1), encoding="utf-8")


OUT.mkdir(parents=True, exist_ok=True)
for run_id, (groups, seed) in SETS.items():
    p = V11.parents[3] / "diag" / f"{run_id}_train.jsonl"   # outside the repository: contains unmasked text
    p.parent.mkdir(parents=True, exist_ok=True)
    build(groups, p)
    run(run_id, p, seed)
run("DIAG_V12_S44", V12, 44)

summary = {"changed_groups": len(changed_groups), "half1": changed_groups[:half], "half2": changed_groups[half:], "runs": {}}
for rid in list(SETS) + ["DIAG_V12_S44"]:
    e = OUT / rid / "eval.json"
    summary["runs"][rid] = {"dev_general_macro_f1": json.loads(e.read_text())["eval"]["dev_general"]["macro_f1"] if e.exists() else None,
                            "curve": loss_curve(OUT / rid)}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary["runs"], indent=1))
