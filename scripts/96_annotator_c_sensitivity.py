"""
scripts/96_annotator_c_sensitivity.py

Post-release third-annotator (C) sensitivity analysis, exactly as pre-registered in
results/annotation/gold480_annotator_c/PREREGISTRATION_TR.md (section 3):

1. care check: C on the 32 filler items vs the shared A/B label (agreement, Cohen kappa; < 75% -> low reliability)
2. primary: on the 48 A/B disagreements, how often C agrees with A, with B, or with neither; by (A, B) pair type
3. sensitivity: majority label of A, B, C on the 48 items (gold kept when all three differ); the main Gold-480
   results (K1, K2, hybrid P0, eight prompted LLM conditions; accuracy, Macro-F1, exact McNemar) are recomputed
   and reported next to the published ones
Gold-480 v1.0 labels are only read, never written. C's confidence column is not used (pre-registration).
K2 and hybrid P0 predictions are read from the sibling k2_nli repository when it is present.

Output: results/annotation/gold480_annotator_c/analysis/C_SENSITIVITY_v1.{json,md}
Usage: python scripts/96_annotator_c_sensitivity.py
"""
import csv
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "results/annotation/gold480_annotator_c"
C_FILE = C_DIR / "returned/ANNOTATION_C_final_2026-10-23.xlsx"
KEY = C_DIR / "private/blind_key.csv"
GOLD = ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl"
K1 = ROOT / "data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl"
LLM_ROOT = ROOT / "results/llm_baselines/gold_v1.0"
K2NLI = ROOT.parent / "k2_nli/reports/experiments"
OUT = C_DIR / "analysis"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def eid(r):
    return r.get("release_example_id") or r["example_id"]


def macro_f1(ref, pred):
    f = []
    for lab in LABELS:
        tp = sum(ref[i] == lab and pred[i] == lab for i in ref)
        fp = sum(ref[i] != lab and pred[i] == lab for i in ref)
        fn = sum(ref[i] == lab and pred[i] != lab for i in ref)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(LABELS)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABELS) / n ** 2
    return (po - pe) / (1 - pe)


def mcnemar(ref, p, q):
    b = sum(p[i] == ref[i] and q[i] != ref[i] for i in ref)
    c = sum(q[i] == ref[i] and p[i] != ref[i] for i in ref)
    n, k = b + c, min(b, c)
    return b, c, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)


def main():
    ws = openpyxl.load_workbook(C_FILE, read_only=True)["Anotasyon"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    c_rows = [dict(zip(hdr, r)) for r in rows[1:] if any(r)]
    key = list(csv.DictReader(open(KEY, encoding="utf-8")))
    assert len(c_rows) == len(key) == 80
    items = []
    for c, k in zip(c_rows, key):
        assert str(c["Kimlik"]) == k["kor_kimlik"] and c["Etiket"] in LABELS
        items.append({**k, "C": c["Etiket"]})
    fill = [x for x in items if x["madde_turu"] != "uyusmazlik"]
    dis = [x for x in items if x["madde_turu"] == "uyusmazlik"]
    assert len(fill) == 32 and len(dis) == 48 and all(x["A"] == x["B"] for x in fill)

    rep = {"inputs": {"C_file": C_FILE.name}}
    agree = sum(x["C"] == x["gold"] for x in fill)
    rep["care_check"] = {"n": 32, "agree": agree, "rate": round(agree / 32, 3),
                         "kappa": round(kappa([x["C"] for x in fill], [x["gold"] for x in fill]), 3),
                         "low_reliability": agree / 32 < 0.75,
                         "confusion": Counter(f"{x['gold']}->{x['C']}" for x in fill if x["C"] != x["gold"])}

    side = lambda x: "A" if x["C"] == x["A"] else "B" if x["C"] == x["B"] else "neither"
    by_type = {}
    for x in dis:
        by_type.setdefault(f"A={x['A']} / B={x['B']}", Counter())[side(x)] += 1
    rep["primary"] = {"n": 48, "counts": Counter(side(x) for x in dis),
                      "by_pair_type": dict(sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))),
                      "neither_labels": Counter(x["C"] for x in dis if side(x) == "neither")}

    gold = {eid(r): r["gold_label"] for r in load_jsonl(GOLD)}
    maj = dict(gold)
    changed = []
    for x in dis:
        votes = Counter([x["A"], x["B"], x["C"]]).most_common(1)[0]
        new = votes[0] if votes[1] >= 2 else x["gold"]
        if new != gold[x["release_example_id"]]:
            maj[x["release_example_id"]] = new
            changed.append({"id": x["release_example_id"], "gold": gold[x["release_example_id"]], "majority": new})
    rep["majority_labels"] = {"changed": len(changed), "transitions": Counter(f"{c['gold']}->{c['majority']}" for c in changed)}

    systems = {"K1 (ELECTRA-TR)": {eid(r): r["predicted_label"] for r in load_jsonl(K1)}}
    agg = K2NLI / "K2-AGG-ABLATION-GOLD480-v1/artifacts/predictions__K2-PIPE-PRED-GOLD480-v1__mdeberta_base_2mil7__soft_prob_avg.jsonl"
    v4f = K2NLI / "K3-PREDICATE-AWARE-DEBIASED-LLAMA70B-v1/arbitration_predictions_123.jsonl"
    if agg.exists() and v4f.exists():
        k1, k2 = systems["K1 (ELECTRA-TR)"], {eid(r): r["pred_label"] for r in load_jsonl(agg)}
        v4 = {r["example_id"]: r["judge_decision"] for r in load_jsonl(v4f)}
        systems["K2 soft-prob"] = k2
        systems["Hybrid P0 (V4 judge)"] = {i: k1[i] if k1[i] == k2[i] else v4[i] for i in gold}
    for alias, short in [("openai_gpt-4.1-mini", "GPT-4.1-mini"), ("google_gemma-4-26b-a4b-it", "Gemma-4-26B"),
                         ("meta-llama_llama-3.3-70b-instruct", "Llama-3.3-70B"), ("qwen_qwen-2.5-72b-instruct", "Qwen-2.5-72B")]:
        for mode, tag in [("zero_shot", "0-shot"), ("few_shot_8", "8-shot")]:
            systems[f"{short} {tag}"] = {eid(r): r["predicted_label"] for r in load_jsonl(LLM_ROOT / alias / mode / "predictions.jsonl")}

    res = {}
    for name, p in systems.items():
        assert set(p) >= set(gold), name
        res[name] = {ref_name: {"acc": round(sum(p[i] == ref[i] for i in ref) / len(ref), 4), "macro_f1": round(macro_f1(ref, p), 4)}
                     for ref_name, ref in [("gold_v1.0", gold), ("majority_ABC", maj)]}
    rep["systems"] = res
    if "Hybrid P0 (V4 judge)" in systems:
        p0 = systems["Hybrid P0 (V4 judge)"]
        rep["mcnemar_P0_vs"] = {name: {ref_name: dict(zip(["P0_only", "other_only", "p"], [round(v, 4) if isinstance(v, float) else v for v in mcnemar(ref, p0, q)]))
                                       for ref_name, ref in [("gold_v1.0", gold), ("majority_ABC", maj)]}
                                for name, q in systems.items() if name != "Hybrid P0 (V4 judge)"}

    OUT.mkdir(exist_ok=True)
    (OUT / "C_SENSITIVITY_v1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    cc, pr = rep["care_check"], rep["primary"]
    md = ["# Gold-480 annotator C sensitivity analysis (post-release, pre-registered)", "",
          f"Care check (32 fillers): {cc['agree']}/32 = {cc['rate']}, kappa {cc['kappa']}, low reliability: {cc['low_reliability']}", "",
          f"Primary (48 disagreements): C = A {pr['counts']['A']}, C = B {pr['counts']['B']}, neither {pr['counts']['neither']}", "",
          "| A / B pair type | C=A | C=B | neither |", "|---|---:|---:|---:|"]
    md += [f"| {t} | {c['A']} | {c['B']} | {c['neither']} |" for t, c in pr["by_pair_type"].items()]
    md += ["", f"Majority labels: {rep['majority_labels']['changed']} of 480 gold labels change; {dict(rep['majority_labels']['transitions'])}", "",
           "| System | Acc gold | Acc majority | Macro-F1 gold | Macro-F1 majority |", "|---|---:|---:|---:|---:|"]
    md += [f"| {n} | {r['gold_v1.0']['acc']:.3f} | {r['majority_ABC']['acc']:.3f} | {r['gold_v1.0']['macro_f1']:.3f} | {r['majority_ABC']['macro_f1']:.3f} |" for n, r in res.items()]
    if "mcnemar_P0_vs" in rep:
        md += ["", "| P0 vs | gold: P0 only / other only / p | majority: P0 only / other only / p |", "|---|---|---|"]
        md += [f"| {n} | {m['gold_v1.0']['P0_only']} / {m['gold_v1.0']['other_only']} / {m['gold_v1.0']['p']} | {m['majority_ABC']['P0_only']} / {m['majority_ABC']['other_only']} / {m['majority_ABC']['p']} |" for n, m in rep["mcnemar_P0_vs"].items()]
    (OUT / "C_SENSITIVITY_v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
