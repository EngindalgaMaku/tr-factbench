"""
scripts/91_gold480_context_ablation.py

Frozen K1 verifier (H1_DROP10_S45, VERIFIER_CONFIG_v1.0) on Gold-480 under three input
conditions, inference only: full input, no context, shuffled context (seed 0, no self-match).
Reuses the model loading and inference code of scripts/44_context_ablation_eval.py so the
conditions are identical to the development diagnostic. Batch size 1, as in the frozen
gold evaluation. The full-input condition must reproduce verifier_hold/PREDICTIONS_v1.0.jsonl.

Gold labels, texts and the frozen predictions are only read, never written.
Output: runs/ablations/GOLD480_CONTEXT_ABLATION_v1/{predictions_<mode>.jsonl, metrics.json}

Usage: python scripts/91_gold480_context_ablation.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ctx44", ROOT / "scripts" / "44_context_ablation_eval.py")
ctx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctx)
ctx.BATCH_SIZE = 1

GOLD = ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl"
FROZEN = ROOT / "data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl"
OUT = ROOT / "runs/ablations/GOLD480_CONTEXT_ABLATION_v1"


def main() -> None:
    records = ctx.load_jsonl(GOLD)
    for r in records:
        r["example_id"] = r["release_example_id"]
    frozen = {}
    for r in ctx.load_jsonl(FROZEN):
        frozen[r.get("release_example_id") or r["example_id"]] = r["predicted_label"]

    model, tok = ctx.load_model(ROOT / "runs/ablations/H1_DROP10_S45_ABLATION_run")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for mode in ["full", "claim_only", "shuffled"]:
        preds, labels, probs = ctx.run_inference(model, tok, records, mode, 0)
        res = ctx.evaluate(preds, labels, probs, records, mode, "GOLD480")
        ctx.write_jsonl(OUT / f"predictions_{mode}.jsonl", res["predictions"])
        dist = {}
        for p in res["predictions"]:
            dist[p["predicted_label"]] = dist.get(p["predicted_label"], 0) + 1
        metrics[mode] = {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"],
                         "per_class_f1": res["per_class"], "prediction_distribution": dist}
        if mode == "full":
            same = sum(frozen[p["example_id"]] == p["predicted_label"] for p in res["predictions"])
            metrics[mode]["matches_frozen_predictions"] = f"{same}/{len(records)}"
        print(mode, json.dumps(metrics[mode], ensure_ascii=False))
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
