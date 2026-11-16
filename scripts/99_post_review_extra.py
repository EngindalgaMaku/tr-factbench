"""
scripts/99_post_review_extra.py  (post-review analyses for the LRE manuscript, 2026-11-16)

Reviewer M4c: held-out scores on examples that the partial-input encoders get right ("easy")
              and wrong ("hard"), following Gururangan et al. (2018).
Reviewer M8.5: paired group bootstrap (Macro-F1) of every prompted condition against the encoder,
              under the adjudicated labels, Annotator B's labels, and the A/B/C majority labels.
Minor 3:     how many context groups lost the one-claim-per-label structure after review/adjudication.
Minor 5:     error-overlap counts quoted in the text, as a table.
Usage: python scripts/99_post_review_extra.py   ->  results/statistics/POST_REVIEW_EXTRA_v1.json
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = ROOT / "results/statistics/POST_REVIEW_EXTRA_v1.json"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
L = lambda p: [json.loads(x) for x in open(p, encoding="utf-8-sig") if x.strip()]
eid = lambda r: r.get("release_example_id") or r["example_id"]
# majority of A, B, C (scripts/96): the two disputed items on which C sided with B
MAJ_CHANGES = {"tfb_ex_0294": "contradicted", "tfb_ex_0068": "partially_supported"}
LLM = {"GPT-4.1 mini zero-shot": "openai_gpt-4.1-mini/zero_shot", "GPT-4.1 mini 8-shot": "openai_gpt-4.1-mini/few_shot_8",
       "Gemma 4 zero-shot": "google_gemma-4-26b-a4b-it/zero_shot", "Gemma 4 8-shot": "google_gemma-4-26b-a4b-it/few_shot_8",
       "Llama 3.3 zero-shot": "meta-llama_llama-3.3-70b-instruct/zero_shot", "Llama 3.3 8-shot": "meta-llama_llama-3.3-70b-instruct/few_shot_8",
       "Qwen2.5 zero-shot": "qwen_qwen-2.5-72b-instruct/zero_shot", "Qwen2.5 8-shot": "qwen_qwen-2.5-72b-instruct/few_shot_8"}
STRONG = ["GPT-4.1 mini 8-shot", "Gemma 4 zero-shot", "Gemma 4 8-shot", "Llama 3.3 8-shot"]


def mf1(ref, pred, ids):
    f = []
    for lab in LABELS:
        tp = sum(ref[i] == lab and pred[i] == lab for i in ids)
        fp = sum(ref[i] != lab and pred[i] == lab for i in ids)
        fn = sum(ref[i] == lab and pred[i] != lab for i in ids)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f) / 4


def acc(ref, pred, ids):
    return sum(ref[i] == pred[i] for i in ids) / len(ids)


def main():
    gold_rows = L(ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl")
    gold = {eid(r): r["gold_label"] for r in gold_rows}
    group = {eid(r): r["context"] for r in gold_rows}
    ab = list(csv.DictReader(open(ROOT / "data/evaluation/gold_v1.0/agreement/AB_ALIGNED_480_v1.0.csv", encoding="utf-8")))
    refB = {r["release_example_id"]: r["B_label"] for r in ab}
    refM = {**gold, **MAJ_CHANGES}
    sysp = {"Encoder": {eid(r): r["predicted_label"] for r in L(ROOT / "data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl")}}
    for name, d in LLM.items():
        sysp[name] = {eid(r): r["predicted_label"] for r in L(ROOT / f"results/llm_baselines/gold_v1.0/{d}/predictions.jsonl")}
    sysp["Annotator B"] = refB
    res = {}

    # --- M4c: easy / hard by partial-input encoders (majority vote of three seeds) ---
    for view, prefix in (("claim_only", "PR_CLAIM"), ("question_claim", "PR_QCLAIM")):
        runs = [{r["example_id"]: r["predicted_label"] for r in L(ROOT / f"runs/post_review/{prefix}_S{s}/predictions_heldout.jsonl")} for s in (42, 43, 44)]
        right = {i for i in gold if sum(run[i] == gold[i] for run in runs) >= 2}
        easy, hard = sorted(right), sorted(set(gold) - right)
        res[f"easy_hard_{view}"] = {
            "definition": "easy = at least 2 of 3 seeds of the partial-input encoder predict the gold label",
            "n_easy": len(easy), "n_hard": len(hard),
            "easy_by_label": Counter(gold[i] for i in easy),
            "systems": {k: {"easy_macro_f1": round(mf1(gold, p, easy), 3), "hard_macro_f1": round(mf1(gold, p, hard), 3),
                            "easy_acc": round(acc(gold, p, easy), 3), "hard_acc": round(acc(gold, p, hard), 3)}
                        for k, p in sysp.items() if k != "Annotator B"} |
                       {"Annotator B": {"easy_macro_f1": round(mf1(gold, refB, easy), 3), "hard_macro_f1": round(mf1(gold, refB, hard), 3),
                                        "easy_acc": round(acc(gold, refB, easy), 3), "hard_acc": round(acc(gold, refB, hard), 3)}},
            "hard_ids": hard}

    # --- M8.5: paired group bootstrap against the encoder, three references ---
    groups = defaultdict(list)
    for i in gold:
        groups[group[i]].append(i)
    keys = list(groups)
    rng = random.Random(42)
    draws = [[rng.choice(keys) for _ in keys] for _ in range(20000)]
    boot = {}
    for refname, ref in (("adjudicated", gold), ("majority_ABC", refM), ("annotator_B", refB)):
        out = {}
        others = [k for k in sysp if k not in ("Encoder",) and not (refname == "annotator_B" and k == "Annotator B")]
        point = {k: mf1(ref, sysp[k], list(gold)) for k in ["Encoder"] + others}
        dist = {k: [] for k in others}
        for d in draws:
            ids = [i for g in d for i in groups[g]]
            e = mf1(ref, sysp["Encoder"], ids)
            for k in others:
                dist[k].append(mf1(ref, sysp[k], ids) - e)
        for k in others:
            v = sorted(dist[k])
            lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
            out[k] = {"delta": round(point[k] - point["Encoder"], 3), "ci": [round(lo, 3), round(hi, 3)], "excludes_0": lo > 0 or hi < 0}
        out["Encoder_macro_f1"] = round(point["Encoder"], 3)
        boot[refname] = out
    res["paired_bootstrap_vs_encoder"] = boot

    # --- Minor 3: groups that lost one claim per label ---
    def broken(rows, lab="gold_label", key="context"):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r[lab])
        return sum(sorted(v) != sorted(LABELS) for v in g.values()), len(g)
    res["groups_not_one_per_label"] = {
        "dev_general": broken(L(ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl")),
        "dev_stress": broken(L(ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl")),
        "heldout": broken(gold_rows), "train": broken(L(ROOT / "data/primary/train.jsonl"))}

    # --- Minor 5: error overlap ---
    enc = sysp["Encoder"]
    errs = {i for i in gold if enc[i] != gold[i]}
    g8 = sysp["GPT-4.1 mini 8-shot"]
    all_wrong = sorted(i for i in errs if all(sysp[k][i] != gold[i] for k in STRONG))
    res["error_overlap"] = {
        "encoder_errors": len(errs),
        "encoder_errors_by_gold": Counter(gold[i] for i in errs),
        "right_by_system_on_encoder_errors": {k: sum(sysp[k][i] == gold[i] for i in errs) for k in LLM},
        "right_by_system_on_encoder_errors_unverifiable": {k: sum(sysp[k][i] == gold[i] for i in errs if gold[i] == "unverifiable") for k in LLM},
        "wrong_for_encoder_and_all_four_strongest": all_wrong,
        "their_gold": [gold[i] for i in all_wrong],
        "errors_by_system": {k: sum(p[i] != gold[i] for i in gold) for k, p in sysp.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if "easy_hard" not in k else {a: b for a, b in v.items() if a != "hard_ids"}) for k, v in res.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
