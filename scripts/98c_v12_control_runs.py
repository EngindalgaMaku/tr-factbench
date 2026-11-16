"""
scripts/98c_v12_control_runs.py  (v1.2.0 control runs, 2026-11-16)

Retrains the final encoder configuration on the v1.2.0 training split (entity-masked) with
seeds 42-46 and evaluates each run once on the development sets and Gold-480, exactly as
scripts/98_post_review_runs.py does for v1.1.0. The reported models are unchanged; this run
only checks that masking does not move the scores outside the v1.1.0 seed range.

Outputs: runs/post_review_v12/<run_id>/..., results/statistics/POST_REVIEW_V12_CONTROL_v1.json
Usage:   python scripts/98c_v12_control_runs.py      (resumable)
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pr98", ROOT / "scripts" / "98_post_review_runs.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

pr.OUT_DIR = ROOT / "runs" / "post_review_v12"
pr.TRAIN = ROOT / "data" / "primary" / "train.jsonl"          # v1.2.0 (masked)
pr.RUNS = [(f"V12_FULL_S{s}", "full_input", s) for s in (42, 43, 44, 45, 46)]
SUMMARY = ROOT / "results" / "statistics" / "POST_REVIEW_V12_CONTROL_v1.json"
V11 = ROOT / "results" / "statistics" / "POST_REVIEW_ENCODER_RUNS_v1.json"


def summarize():
    rows = {}
    for run_id, _, _ in pr.RUNS:
        p = pr.OUT_DIR / run_id / "eval.json"
        if p.exists():
            rows[run_id] = json.loads(p.read_text(encoding="utf-8"))
    agg = {"n_runs": len(rows)}
    for split in pr.EVAL:
        vals = [v["eval"][split]["macro_f1"] for v in rows.values()]
        if vals:
            agg[split] = {"mean": round(statistics.mean(vals), 4),
                          "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                          "min": min(vals), "max": max(vals)}
    g = [v["eval"]["heldout"]["groups_all_correct"] for v in rows.values()]
    if g:
        agg["heldout_groups_all_correct"] = {"mean": round(statistics.mean(g), 1), "min": min(g), "max": max(g)}
    out = {"train_file": "data/primary/train.jsonl (v1.2.0, entity-masked)", "runs": rows, "aggregate_v12": agg}
    if V11.exists():
        out["aggregate_v11"] = json.loads(V11.read_text(encoding="utf-8"))["aggregate"].get("PR_FULL")
    SUMMARY.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("aggregate_v12", "aggregate_v11") if k in out}, ensure_ascii=False, indent=1))


def main():
    for run_id, view, seed in pr.RUNS:
        run_dir = pr.OUT_DIR / run_id
        if (run_dir / "eval.json").exists():
            print(f"skip {run_id} (done)")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            pr.best_model_dir(run_dir)
            secs = None
        except FileNotFoundError:
            secs = pr.train(run_id, view, seed)
        ev = pr.evaluate(run_id, view)
        (run_dir / "eval.json").write_text(json.dumps(
            {"run_id": run_id, "view": view, "seed": seed, "train_seconds": secs, "eval": ev},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{run_id}: heldout Macro-F1 {ev['heldout']['macro_f1']}", flush=True)
        summarize()
    summarize()


if __name__ == "__main__":
    main()
