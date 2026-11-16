"""
scripts/98d_recheck_v11_s42.py  (diagnostic, 2026-11-16)

Two v1.2.0 control runs collapsed to the majority class. This run separates the two possible
causes: it retrains seed 42 on the *v1.1.0* training split (unmasked copy kept outside the
repository) with the identical configuration. If it learns normally, the masked data is the
cause; if it also collapses, something in the environment changed since the earlier runs.

Output: runs/recheck_v11/RECHECK_V11_S42/eval.json
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pr98", ROOT / "scripts" / "98_post_review_runs.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

pr.OUT_DIR = ROOT / "runs" / "recheck_v11"
pr.TRAIN = ROOT.parent / "_private_masking_v1.2" / "v1.1_originals" / "data" / "primary" / "train.jsonl"
assert pr.TRAIN.exists(), pr.TRAIN
run_id, view, seed = "RECHECK_V11_S42", "full_input", 42
run_dir = pr.OUT_DIR / run_id
run_dir.mkdir(parents=True, exist_ok=True)
try:
    pr.best_model_dir(run_dir)
    secs = None
except FileNotFoundError:
    secs = pr.train(run_id, view, seed)
ev = pr.evaluate(run_id, view)
(run_dir / "eval.json").write_text(json.dumps({"run_id": run_id, "train_file": str(pr.TRAIN), "seed": seed,
                                               "train_seconds": secs, "eval": ev}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{run_id}: dev_general {ev['dev_general']['macro_f1']}  heldout {ev['heldout']['macro_f1']}")
