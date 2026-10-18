"""
scripts/94_turkish_phenomena.py

Turkish morphological markers and lemma-level overlap by label, for all four splits.

- negation: verbal -mA- (zeyrek analysis contains Neg) or the words değil / yok / hiçbir
- possibility/ability: -Abil- (zeyrek analysis contains Able)
- necessity: -mAlI (zeyrek analysis contains Neces) or zorunlu / zorunda / gerek(ir) / şart / mecburi
- "new" marker: present in the claim but absent from its context
- modal shift: context has -Abil-, claim has necessity and no -Abil-
- claim--context overlap of word types and of lemma types (Turkish-aware lower-casing)
- Gold-480 error rates of the frozen encoder and of 8-shot GPT-4.1 mini by marker subset

zeyrek returns analyses in an order that can change between runs, so the script does not use
the first analysis. A morphological marker counts only when every analysis of the token carries
it, and the lemma is the most frequent lemma over all analyses (ties alphabetical). The output is
therefore identical across runs. Data files and predictions are only read.
Requires: pip install zeyrek
Output: results/statistics/TURKISH_PHENOMENA_v1.json

Usage: python scripts/94_turkish_phenomena.py
"""
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict

# zeyrek's candidate sets depend on Python's string-hash order; fix it so runs are identical.
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env={**os.environ, "PYTHONHASHSEED": "0"}))
from pathlib import Path
from statistics import mean

logging.disable(logging.CRITICAL)
import zeyrek  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "train": "data/primary/train.jsonl",
    "dev_general_reviewed": "data/evaluation/preview_v0.1/dev_general.jsonl",
    "dev_stress_reviewed": "data/evaluation/preview_v0.1/dev_stress.jsonl",
    "gold480": "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl",
}
ENCODER = ROOT / "data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl"
GPT8 = ROOT / "results/llm_baselines/gold_v1.0/openai_gpt-4.1-mini/few_shot_8/predictions.jsonl"
OUT = ROOT / "results/statistics/TURKISH_PHENOMENA_v1.json"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
MARKERS = ["neg", "able", "nec"]
NEG_WORDS = {"değil", "değildir", "değildi", "yok", "yoktur", "hiçbir"}
NEC_WORDS = {"zorunlu", "zorunludur", "zorunda", "gerek", "gerekir", "gerekmektedir", "gereklidir",
             "şart", "şarttır", "mecburi", "mecburidir"}
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
WORD = re.compile(r"\w+")
ANALYZER = zeyrek.MorphAnalyzer()
_feat, _lemma = {}, {}


def tr_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def analyses(word):
    """All zeyrek analyses of a token, sorted so that the result does not depend on their order."""
    try:
        res = ANALYZER.analyze(word)
        parses = [a for a in (res[0] if res else []) if a.lemma and a.lemma != "Unk"]
    except Exception:
        parses = []
    return sorted(parses, key=lambda a: a.formatted)


def token_markers(word):
    """A marker counts only if every analysis of the token carries it (unambiguous tokens)."""
    if word not in _feat:
        f = set()
        if word in NEG_WORDS:
            f.add("neg")
        if word in NEC_WORDS:
            f.add("nec")
        parses = [a.formatted for a in analyses(word)]
        if parses and all(":Neg" in s or "+Neg" in s for s in parses):
            f.add("neg")
        if parses and all(":Able" in s for s in parses):
            f.add("able")
        if parses and all(":Neces" in s for s in parses):
            f.add("nec")
        _feat[word] = f
    return _feat[word]


def lemma(word):
    """Most frequent lemma over all analyses; ties broken alphabetically."""
    if word not in _lemma:
        lemmas = [tr_lower(a.lemma) for a in analyses(word)]
        _lemma[word] = min(set(lemmas), key=lambda x: (-lemmas.count(x), x)) if lemmas else word
    return _lemma[word]


def markers(text):
    out = set()
    for w in TOKEN.findall(tr_lower(text)):
        out |= token_markers(w)
    return out


def overlaps(claim, context):
    cw, xw = set(WORD.findall(tr_lower(claim))), set(WORD.findall(tr_lower(context)))
    if not cw:
        return 0.0, 0.0
    cl, xl = {lemma(w) for w in cw}, {lemma(w) for w in xw}
    return len(cw & xw) / len(cw), len(cl & xl) / len(cl)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    result = {}
    gold_rows = None
    for name, rel in SPLITS.items():
        rows = load_jsonl(ROOT / rel)
        if name == "gold480":
            gold_rows = rows
        ctx_cache = {}
        agg = defaultdict(lambda: defaultdict(float))
        ov = defaultdict(lambda: ([], []))
        for r in rows:
            lab = r["gold_label"]
            cm = markers(r["claim"])
            if r["context"] not in ctx_cache:
                ctx_cache[r["context"]] = markers(r["context"])
            xm = ctx_cache[r["context"]]
            a = agg[lab]
            a["n"] += 1
            for m in MARKERS:
                if m in cm:
                    a[m] += 1
                    if m not in xm:
                        a[m + "_new"] += 1
            if "able" in xm and "nec" in cm and "able" not in cm:
                a["modal_shift"] += 1
            s, l = overlaps(r["claim"], r["context"])
            ov[lab][0].append(s)
            ov[lab][1].append(l)
        result[name] = {}
        print("==", name)
        for lab in LABELS:
            a = agg[lab]
            n = a["n"]
            rates = {k: round(a[k] / n, 6) for k in
                     ["neg", "neg_new", "able", "able_new", "nec", "nec_new", "modal_shift"]}
            rates["modal_shift_count"] = int(a["modal_shift"])
            rates["n"] = int(n)
            rates["overlap_word"] = round(mean(ov[lab][0]), 6)
            rates["overlap_lemma"] = round(mean(ov[lab][1]), 6)
            result[name][lab] = rates
            print(f"  {lab:20s} neg {rates['neg']:.3f} (new {rates['neg_new']:.3f})  able {rates['able']:.3f}  "
                  f"nec {rates['nec']:.3f}  shift {rates['modal_shift_count']:3d}  "
                  f"overlap word {rates['overlap_word']:.3f} lemma {rates['overlap_lemma']:.3f}")

    enc = {r["release_example_id"]: r["predicted_label"] for r in load_jsonl(ENCODER)}
    gpt = {r["example_id"]: r["predicted_label"] for r in load_jsonl(GPT8)}
    err = defaultdict(lambda: [0, 0, 0])
    for r in gold_rows:
        i, lab = r["release_example_id"], r["gold_label"]
        cm = markers(r["claim"]) & set(MARKERS)
        for k in ["all"] + sorted(cm) + ([] if cm else ["none"]):
            e = err[k]
            e[0] += 1
            e[1] += enc[i] != lab
            e[2] += gpt[i] != lab
    result["gold480_error_rates"] = {k: {"n": n, "encoder": round(a / n, 6), "gpt41mini_8shot": round(b / n, 6)}
                                     for k, (n, a, b) in err.items()}
    print("gold480 error rates:", json.dumps(result["gold480_error_rates"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
