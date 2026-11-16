"""
scripts/98_post_review_runs.py  (post-review analyses for the LRE manuscript, 2026-11-16)

Trains and evaluates encoder runs requested by the reviewer, with the exact configuration
of the selected encoder (ELECTRA-TR, LoRA r=16, alpha=32, dropout 0.10, all-linear, lr 2e-4,
batch 8, 5 epochs, max length 512; checkpoint chosen on the pre-review general development
set, as in the original runs). Training goes through src/train.py, like scripts/10.

Runs
  PR_FULL_S42 ... PR_FULL_S46   full input, five seeds        (reviewer M3b: seed variance)
  PR_CLAIM_S42/S43/S44          claim only                     (reviewer M4a: claim-only encoder)
  PR_QCLAIM_S42/S43/S44         question + claim               (reviewer M4a)

Each run is evaluated on the reviewed development sets and on the held-out set (Gold-480),
batch size 1 as in the frozen held-out evaluation. Held-out labels are only read.
Outputs: runs/post_review/<run_id>/{ckpt/, eval.json, predictions_<set>.jsonl}
         results/statistics/POST_REVIEW_ENCODER_RUNS_v1.json  (summary)

The script is resumable: a run whose eval.json exists is skipped.
Usage: python scripts/98_post_review_runs.py            (all runs)
       python scripts/98_post_review_runs.py PR_FULL_S42 (selected runs)
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "src" / "train.py"
OUT_DIR = ROOT / "runs" / "post_review"
SUMMARY = ROOT / "results" / "statistics" / "POST_REVIEW_ENCODER_RUNS_v1.json"
MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
TRAIN = ROOT / "data/primary/train.jsonl"
SELECT_DEV = ROOT / "data/evaluation/legacy_experiment_version/dev_general.jsonl"
EVAL = {
    "dev_general": ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl",
    "dev_stress": ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl",
    "heldout": ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl",
}
RUNS = [(f"PR_FULL_S{s}", "full_input", s) for s in (42, 43, 44, 45, 46)]
RUNS += [(f"PR_CLAIM_S{s}", "claim_only", s) for s in (42, 43, 44)]
RUNS += [(f"PR_QCLAIM_S{s}", "question_claim", s) for s in (42, 43, 44)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def view_text(r, view):
    """Same formatting as src/train.py for each input view."""
    ctx, q, cl = r.get("context", ""), r.get("question", ""), r.get("claim", "")
    if view == "full_input":
        return f"[CONTEXT] {ctx}", f"[QUESTION] {q} [CLAIM] {cl}"
    if view == "claim_only":
        return f"[CLAIM] {cl}", None
    if view == "question_claim":
        return f"[QUESTION] {q} [CLAIM] {cl}", None
    raise ValueError(view)


def train(run_id, view, seed):
    run_dir = OUT_DIR / run_id
    cmd = [sys.executable, str(TRAIN_PY), "--domain", "multidomain", "--model_name", MODEL,
           "--seed", str(seed), "--epochs", "5.0", "--batch_size", "8", "--learning_rate", "0.0002",
           "--peft_type", "lora", "--freeze_layers", "0",
           "--train_file", str(TRAIN), "--dev_file", str(SELECT_DEV), "--test_file", str(SELECT_DEV),
           "--output_dir", str(run_dir / "ckpt"), "--label_set", "4class",
           "--lora_r", "16", "--lora_alpha", "32", "--lora_target_modules", "all-linear",
           "--lora_dropout", "0.1", "--input_view", view]
    print(f"\n=== TRAIN {run_id} ({view}, seed {seed}) ===", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"training failed for {run_id}")
    return round(time.time() - t0, 1)


def best_model_dir(run_dir):
    hits = sorted((run_dir / "ckpt").rglob("best_model"))
    if not hits:
        raise FileNotFoundError(f"no best_model under {run_dir}")
    return hits[-1]


def evaluate(run_id, view):
    run_dir = OUT_DIR / run_id
    adapter = best_model_dir(run_dir)
    tok = AutoTokenizer.from_pretrained(str(adapter))
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, str(adapter)).eval().to(DEVICE)
    out = {}
    for name, path in EVAL.items():
        rows = load_jsonl(path)
        preds = []
        with torch.no_grad():
            for r in rows:
                a, b = view_text(r, view)
                enc = tok(a, b, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
                preds.append(LABELS[int(model(**enc).logits.argmax(-1))])
        gold = [r["gold_label"] for r in rows]
        ids = [r.get("release_example_id") or r["example_id"] for r in rows]
        res = {"n": len(rows),
               "macro_f1": round(f1_score(gold, preds, labels=LABELS, average="macro"), 4),
               "accuracy": round(sum(g == p for g, p in zip(gold, preds)) / len(gold), 4),
               "per_class_f1": dict(zip(LABELS, [round(x, 4) for x in f1_score(gold, preds, labels=LABELS, average=None)]))}
        if name == "heldout":
            groups = {}
            for r, g, p in zip(rows, gold, preds):
                groups.setdefault(r["context"], []).append(g == p)
            res["groups_all_correct"] = sum(all(v) for v in groups.values())
            res["n_groups"] = len(groups)
        out[name] = res
        with open(run_dir / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for i, g, p in zip(ids, gold, preds):
                f.write(json.dumps({"example_id": i, "gold_label": g, "predicted_label": p}, ensure_ascii=False) + "\n")
    out["adapter"] = str(adapter.relative_to(ROOT))
    return out


def summarize():
    rows = {}
    for run_id, view, seed in RUNS:
        p = OUT_DIR / run_id / "eval.json"
        if p.exists():
            rows[run_id] = json.loads(p.read_text(encoding="utf-8"))
    summary = {"runs": rows, "aggregate": {}}
    for prefix in ("PR_FULL", "PR_CLAIM", "PR_QCLAIM"):
        sel = [v for k, v in rows.items() if k.startswith(prefix + "_S")]
        if not sel:
            continue
        agg = {"n_runs": len(sel)}
        for split in EVAL:
            vals = [v["eval"][split]["macro_f1"] for v in sel]
            agg[split] = {"mean": round(statistics.mean(vals), 4),
                          "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                          "min": min(vals), "max": max(vals)}
        g = [v["eval"]["heldout"]["groups_all_correct"] for v in sel]
        agg["heldout_groups_all_correct"] = {"mean": round(statistics.mean(g), 1), "min": min(g), "max": max(g)}
        summary["aggregate"][prefix] = agg
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=1))


def main():
    wanted = set(sys.argv[1:])
    print(f"device: {DEVICE}")
    if DEVICE != "cuda":
        print("WARNING: no CUDA device; training on CPU will be very slow.")
    for run_id, view, seed in RUNS:
        if wanted and run_id not in wanted:
            continue
        run_dir = OUT_DIR / run_id
        if (run_dir / "eval.json").exists():
            print(f"skip {run_id} (done)")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            best_model_dir(run_dir)
            secs = None
        except FileNotFoundError:
            secs = train(run_id, view, seed)
        ev = evaluate(run_id, view)
        (run_dir / "eval.json").write_text(json.dumps(
            {"run_id": run_id, "view": view, "seed": seed, "train_seconds": secs, "eval": ev},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{run_id}: heldout Macro-F1 {ev['heldout']['macro_f1']}  groups {ev['heldout']['groups_all_correct']}/120", flush=True)
        summarize()
    summarize()


if __name__ == "__main__":
    main()
