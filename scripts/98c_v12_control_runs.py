"""
scripts/98c_v12_control_runs.py  (v1.2.0 control runs, 2026-11-16)

Retrains the final encoder configuration on the v1.2.0 training split (entity-masked) and
evaluates each run once on the development sets and Gold-480, as scripts/98_post_review_runs.py
does for v1.1.0. The reported models are unchanged; the runs only check that masking does not
move the scores outside the v1.1.0 seed range.

Collapse rule (applied to v1.1.0 and v1.2.0 alike): a run whose predictions on the general
development set fall in a single class for more than 95% of the examples is counted as
collapsed and replaced by the next seed (47, 48, ...) until five runs have converged.
Collapsed runs are kept and reported separately. None of the five v1.1.0 runs collapsed.

Outputs: runs/post_review_v12/<run_id>/..., results/statistics/POST_REVIEW_V12_CONTROL_v1.json
Usage:   python scripts/98c_v12_control_runs.py      (resumable)
"""
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pr98", ROOT / "scripts" / "98_post_review_runs.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

pr.OUT_DIR = ROOT / "runs" / "post_review_v12"
pr.TRAIN = ROOT / "data" / "primary" / "train.jsonl"          # v1.2.0 (masked)
SUMMARY = ROOT / "results" / "statistics" / "POST_REVIEW_V12_CONTROL_v1.json"
V11_RUNS = ROOT / "runs" / "post_review"
TARGET, MAX_SEED, COLLAPSE_SHARE = 5, 60, 0.95


def collapse_share(run_dir):
    preds = [json.loads(l)["predicted_label"] for l in open(run_dir / "predictions_dev_general.jsonl", encoding="utf-8") if l.strip()]
    return max(Counter(preds).values()) / len(preds)


def agg(rows):
    out = {"n_runs": len(rows)}
    for split in pr.EVAL:
        vals = [v["eval"][split]["macro_f1"] for v in rows]
        if vals:
            out[split] = {"mean": round(statistics.mean(vals), 4),
                          "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                          "min": min(vals), "max": max(vals)}
    g = [v["eval"]["heldout"]["groups_all_correct"] for v in rows]
    if g:
        out["heldout_groups_all_correct"] = {"mean": round(statistics.mean(g), 1), "min": min(g), "max": max(g)}
    return out


def summarize():
    runs = {}
    for d in sorted(pr.OUT_DIR.glob("V12_FULL_S*")):
        if (d / "eval.json").exists():
            r = json.loads((d / "eval.json").read_text(encoding="utf-8"))
            r["dominant_class_share_dev_general"] = round(collapse_share(d), 3)
            r["collapsed"] = r["dominant_class_share_dev_general"] > COLLAPSE_SHARE
            runs[d.name] = r
    conv = [r for r in runs.values() if not r["collapsed"]]
    v11 = []
    for d in sorted(V11_RUNS.glob("PR_FULL_S*")):
        r = json.loads((d / "eval.json").read_text(encoding="utf-8"))
        r["dominant_class_share_dev_general"] = round(collapse_share(d), 3)
        v11.append(r)
    out = {"train_file": "data/primary/train.jsonl (v1.2.0, entity-masked)",
           "collapse_rule": f"dominant class share on the general development set > {COLLAPSE_SHARE}; replaced by the next seed",
           "v12_converged": agg(conv), "v12_seeds_converged": [r["seed"] for r in conv],
           "v12_seeds_collapsed": [r["seed"] for r in runs.values() if r["collapsed"]],
           "v11_converged": agg([r for r in v11 if r["dominant_class_share_dev_general"] <= COLLAPSE_SHARE]),
           "v11_seeds_collapsed": [r["seed"] for r in v11 if r["dominant_class_share_dev_general"] > COLLAPSE_SHARE],
           "runs": runs}
    SUMMARY.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "runs"}, ensure_ascii=False, indent=1))
    return len(conv)


def main():
    seed = 42
    while summarize() < TARGET and seed <= MAX_SEED:
        run_id = f"V12_FULL_S{seed}"
        run_dir = pr.OUT_DIR / run_id
        if not (run_dir / "eval.json").exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                pr.best_model_dir(run_dir)
                secs = None
            except FileNotFoundError:
                secs = pr.train(run_id, "full_input", seed)
            ev = pr.evaluate(run_id, "full_input")
            (run_dir / "eval.json").write_text(json.dumps(
                {"run_id": run_id, "view": "full_input", "seed": seed, "train_seconds": secs, "eval": ev},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{run_id}: dev_general {ev['dev_general']['macro_f1']}  heldout {ev['heldout']['macro_f1']}", flush=True)
        seed += 1


if __name__ == "__main__":
    main()
