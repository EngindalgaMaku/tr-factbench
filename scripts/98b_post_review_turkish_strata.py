"""
scripts/98b_post_review_turkish_strata.py  (post-review analyses for the LRE manuscript, 2026-11-16)

Reviewer M5: (a) encoder error rate by Turkish marker *within* each gold label, so that the
-Abil- comparison is not confounded by the label distribution; (b) evidential and epistemic
markers: -mIş (zeyrek tag Narr) and the copula -DIr (zeyrek tag Cop), in claims and contexts.

Uses the marker rules of scripts/94_turkish_phenomena.py (a marker counts only when every
zeyrek analysis of a token contains it) and the same PYTHONHASHSEED=0 restart.
Output: results/statistics/POST_REVIEW_TURKISH_STRATA_v1.json
Usage: python scripts/98b_post_review_turkish_strata.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env={**os.environ, "PYTHONHASHSEED": "0"}))

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tp94", ROOT / "scripts" / "94_turkish_phenomena.py")
tp = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
spec.loader.exec_module(tp)   # defines helpers; main() is not called on import
sys.argv = _argv

OUT = ROOT / "results/statistics/POST_REVIEW_TURKISH_STRATA_v1.json"
LABELS = tp.LABELS


def evid(text):
    """Return {'narr': bool, 'cop': bool} with the all-analyses rule of script 94."""
    narr = cop = False
    for w in tp.TOKEN.findall(tp.tr_lower(text)):
        parses = [a.formatted for a in tp.analyses(w)]
        if parses and all(":Narr" in s or "+Narr" in s for s in parses):
            narr = True
        if parses and all(":Cop" in s or "+Cop" in s for s in parses):
            cop = True
    return {"narr": narr, "cop": cop}


def main():
    res = {}
    gold = tp.load_jsonl(ROOT / tp.SPLITS["gold480"])
    pred = {r.get("release_example_id") or r["example_id"]: r["predicted_label"] for r in tp.load_jsonl(tp.ENCODER)}

    # (a) encoder error rate by label x marker on the held-out set
    strata = defaultdict(lambda: [0, 0])  # (label, group) -> [errors, n]
    for r in gold:
        m = tp.markers(r["claim"])
        lab = r["gold_label"]
        err = pred[r["release_example_id"]] != lab
        groups = ["any"]
        groups.append("able" if "able" in m else "no_able")
        if not any(k in m for k in tp.MARKERS):
            groups.append("none_of_three")
        if "neg" in m:
            groups.append("neg")
        for g in groups:
            strata[(lab, g)][0] += err
            strata[(lab, g)][1] += 1
    res["heldout_encoder_error_by_label_and_marker"] = {
        lab: {g: {"errors": strata[(lab, g)][0], "n": strata[(lab, g)][1],
                  "rate": round(strata[(lab, g)][0] / strata[(lab, g)][1], 3) if strata[(lab, g)][1] else None}
              for g in ("any", "able", "no_able", "none_of_three", "neg")}
        for lab in LABELS}

    # (b) evidential / copula markers in claims and contexts, by split and label
    ev = {}
    for split, path in tp.SPLITS.items():
        rows = tp.load_jsonl(ROOT / path)
        ctx_cache = {}
        by = defaultdict(lambda: defaultdict(int))
        for r in rows:
            lab = r["gold_label"]
            c = evid(r["claim"])
            if r["context"] not in ctx_cache:
                ctx_cache[r["context"]] = evid(r["context"])
            x = ctx_cache[r["context"]]
            by[lab]["n"] += 1
            by[lab]["claim_narr"] += c["narr"]
            by[lab]["claim_cop"] += c["cop"]
            by[lab]["claim_narr_context_without"] += c["narr"] and not x["narr"]
        ev[split] = {lab: {k: (round(v / by[lab]["n"], 3) if k != "n" else v) for k, v in by[lab].items()} for lab in LABELS}
    res["evidential_markers"] = ev

    # encoder error rate on held-out claims with / without -mIs
    e = defaultdict(lambda: [0, 0])
    for r in gold:
        k = "narr" if evid(r["claim"])["narr"] else "no_narr"
        e[k][0] += pred[r["release_example_id"]] != r["gold_label"]
        e[k][1] += 1
    res["heldout_encoder_error_by_narr"] = {k: {"errors": v[0], "n": v[1], "rate": round(v[0] / v[1], 3)} for k, v in e.items()}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
