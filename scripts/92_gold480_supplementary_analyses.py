"""
scripts/92_gold480_supplementary_analyses.py

Supplementary Gold-480 analyses of the frozen held-out predictions (no inference):

1. per-class F1 of the encoder (H1_DROP10_S45) and the eight prompted LLM conditions
2. annotator robustness: Macro-F1 against the adjudicated gold labels (= Annotator A),
   against Annotator B's blind labels, and on the 432 examples where A and B agree
3. error overlap between the encoder and each LLM condition, and the examples that
   the encoder and the four strongest prompted conditions all get wrong

4. human reference: Annotator B's blind labels scored against the adjudicated labels, with a
   95% CI and paired differences (system minus B) on the same context-group bootstrap draws as
   scripts/46_gold_group_bootstrap.py (120 groups, 20,000 draws, seed 42)

Gold labels, annotator files and predictions are only read, never written.
Output: results/llm_baselines/gold_v1.0/supplementary/GOLD480_SUPPLEMENTARY_v1.json

Usage: python scripts/92_gold480_supplementary_analyses.py
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data/evaluation/gold_v1.0"
LLM_ROOT = ROOT / "results/llm_baselines/gold_v1.0"
OUT = LLM_ROOT / "supplementary" / "GOLD480_SUPPLEMENTARY_v1.json"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]

SYSTEMS = {"encoder": GOLD_DIR / "verifier_hold/PREDICTIONS_v1.0.jsonl"}
for alias, short in [("openai_gpt-4.1-mini", "gpt"), ("google_gemma-4-26b-a4b-it", "gemma"),
                     ("meta-llama_llama-3.3-70b-instruct", "llama"), ("qwen_qwen-2.5-72b-instruct", "qwen")]:
    for mode, tag in [("zero_shot", "0s"), ("few_shot_8", "8s")]:
        SYSTEMS[f"{short}_{tag}"] = LLM_ROOT / alias / mode / "predictions.jsonl"
STRONGEST = ["gpt_8s", "llama_8s", "gemma_0s", "gemma_8s"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_class_f1(gold, pred):
    out = {}
    for lab in LABELS:
        tp = sum(gold[i] == lab and pred.get(i) == lab for i in gold)
        fp = sum(gold[i] != lab and pred.get(i) == lab for i in gold)
        fn = sum(gold[i] == lab and pred.get(i) != lab for i in gold)
        out[lab] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def macro_f1(gold, pred, ids=None):
    ref = gold if ids is None else {i: gold[i] for i in ids}
    return sum(per_class_f1(ref, pred).values()) / len(LABELS)


def human_reference(gold, ann_b, preds):
    import importlib.util
    from collections import defaultdict
    import numpy as np
    spec = importlib.util.spec_from_file_location("boot46", ROOT / "scripts" / "46_gold_group_bootstrap.py")
    b46 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b46)

    rows = load_jsonl(GOLD_DIR / "gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl")
    ids = [r["release_example_id"] for r in rows]
    grouped = defaultdict(list)
    for idx, r in enumerate(rows):
        grouped[r["context"]].append(idx)
    order = sorted(grouped, key=lambda c: ids[grouped[c][0]])
    group_idx = np.array([grouped[c] for c in order], dtype=np.int64)
    rng = np.random.default_rng(b46.SEED)
    draws = rng.integers(0, b46.EXPECTED_GROUPS, size=(b46.N_BOOTSTRAP, b46.EXPECTED_GROUPS), dtype=np.int64)
    ex = group_idx[draws].reshape(b46.N_BOOTSTRAP, b46.EXPECTED_EXAMPLES)
    enc = lambda labels: np.array([b46.LABEL_TO_I[labels[i]] for i in ids], dtype=np.int64)
    y = enc(gold)
    systems = {"annotator_B": enc(ann_b), **{k: enc(v) for k, v in preds.items()}}
    boot = {k: np.array([b46.macro_f1(y[e], p[e]) for e in ex]) for k, p in systems.items()}
    point = {k: b46.macro_f1(y, p) for k, p in systems.items()}
    out = {"macro_f1": round(point["annotator_B"], 6),
           "macro_f1_ci95": [round(x, 6) for x in b46.percentile_ci(boot["annotator_B"])],
           "paired_delta_system_minus_B": {}}
    for k in preds:
        d = boot[k] - boot["annotator_B"]
        out["paired_delta_system_minus_B"][k] = {
            "delta": round(point[k] - point["annotator_B"], 6),
            "ci95": [round(x, 6) for x in b46.percentile_ci(d)]}
    return out


def main() -> None:
    gold = {r["release_example_id"]: r["gold_label"] if "gold_label" in r else r["label"]
            for r in load_jsonl(GOLD_DIR / "gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl")}
    ann_a, ann_b = {}, {}
    with open(GOLD_DIR / "agreement/AB_ALIGNED_480_v1.0.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ann_a[row["release_example_id"]] = row["A_label"]
            ann_b[row["release_example_id"]] = row["B_label"]
    assert len(gold) == len(ann_a) == len(ann_b) == 480
    assert all(gold[i] == ann_a[i] for i in gold)  # all 48 adjudicated labels matched A
    agreed = [i for i in gold if ann_a[i] == ann_b[i]]
    assert len(agreed) == 432

    preds = {}
    for name, path in SYSTEMS.items():
        preds[name] = {(r.get("release_example_id") or r["example_id"]): r["predicted_label"]
                       for r in load_jsonl(path)}
        assert set(preds[name]) == set(gold), name

    result = {"per_class_f1": {}, "annotator_robustness": {}, "error_overlap": {}}
    for name, pred in preds.items():
        result["per_class_f1"][name] = {k: round(v, 6) for k, v in per_class_f1(gold, pred).items()}
        result["annotator_robustness"][name] = {
            "macro_f1_vs_gold": round(macro_f1(gold, pred), 6),
            "macro_f1_vs_annotator_B": round(macro_f1(ann_b, pred), 6),
            "macro_f1_on_agreed_432": round(macro_f1(gold, pred, agreed), 6),
        }

    enc_err = {i for i in gold if preds["encoder"][i] != gold[i]}
    for name, pred in preds.items():
        if name == "encoder":
            continue
        fixed = [i for i in enc_err if pred[i] == gold[i]]
        result["error_overlap"][name] = {
            "encoder_errors": len(enc_err),
            "llm_correct_on_encoder_errors": len(fixed),
            "both_wrong": len(enc_err) - len(fixed),
            "llm_correct_on_encoder_errors_by_gold_label": {
                lab: sum(gold[i] == lab for i in fixed) for lab in LABELS},
            "encoder_errors_by_gold_label": {lab: sum(gold[i] == lab for i in enc_err) for lab in LABELS},
        }
    all_wrong = sorted(i for i in enc_err if all(preds[s][i] != gold[i] for s in STRONGEST))
    result["wrong_for_encoder_and_strongest"] = {"conditions": STRONGEST, "n": len(all_wrong), "ids": all_wrong}

    # Human reference: Annotator B's blind labels scored against the adjudicated labels, with the
    # same context-group bootstrap as scripts/46 (120 groups, 20,000 draws, seed 42, same draws).
    result["human_reference_annotator_B"] = human_reference(gold, ann_b, preds)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in preds:
        r = result["annotator_robustness"][name]
        pc = result["per_class_f1"][name]
        print(f"{name:9s} gold {r['macro_f1_vs_gold']:.3f}  B {r['macro_f1_vs_annotator_B']:.3f}  "
              f"agreed {r['macro_f1_on_agreed_432']:.3f}  | " + " ".join(f"{pc[l]:.3f}" for l in LABELS))
    g = result["error_overlap"]["gpt_8s"]
    print("gpt_8s on encoder errors:", g["llm_correct_on_encoder_errors"], "/", g["encoder_errors"],
          "| unverifiable:", g["llm_correct_on_encoder_errors_by_gold_label"]["unverifiable"], "/",
          g["encoder_errors_by_gold_label"]["unverifiable"])
    print("wrong for encoder and strongest four:", len(all_wrong))
    h = result["human_reference_annotator_B"]
    print("annotator B vs adjudicated: %.3f [%.3f, %.3f]" % (h["macro_f1"], *h["macro_f1_ci95"]))
    for k, v in h["paired_delta_system_minus_B"].items():
        print(f"  {k:9s} minus B {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
