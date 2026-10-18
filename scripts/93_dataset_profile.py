"""
scripts/93_dataset_profile.py

Surface profile of the four splits (train, reviewed DEV960, reviewed DEV600, Gold-480):

- mean context length (whitespace words, averaged over distinct contexts)
- mean question and claim length (whitespace words, averaged over examples)
- claim--context word overlap by label: share of claim word types (Turkish-aware lower-cased \\w+)
  that also occur in the context, averaged over the examples of each gold/candidate label

Data files are only read. Output: results/statistics/DATASET_PROFILE_v1.json

Usage: python scripts/93_dataset_profile.py
"""
import json
import re
import sys
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "train": "data/primary/train.jsonl",
    "dev_general_reviewed": "data/evaluation/preview_v0.1/dev_general.jsonl",
    "dev_stress_reviewed": "data/evaluation/preview_v0.1/dev_stress.jsonl",
    "gold480": "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl",
}
OUT = ROOT / "results/statistics/DATASET_PROFILE_v1.json"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
WORD = re.compile(r"\w+")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tr_lower(text):
    """Turkish-aware lower-casing: I -> ı and İ -> i before str.lower()."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def overlap(claim, context):
    cw = set(WORD.findall(tr_lower(claim)))
    return len(cw & set(WORD.findall(tr_lower(context)))) / len(cw) if cw else 0.0


def main() -> None:
    result = {}
    for name, rel in SPLITS.items():
        rows = load_jsonl(ROOT / rel)
        contexts = {r["context"] for r in rows}
        result[name] = {
            "n_examples": len(rows),
            "n_contexts": len(contexts),
            "mean_context_words": round(mean(len(c.split()) for c in contexts), 4),
            "mean_question_words": round(mean(len(r["question"].split()) for r in rows), 4),
            "mean_claim_words": round(mean(len(r["claim"].split()) for r in rows), 4),
            "claim_context_overlap": {
                lab: round(mean(overlap(r["claim"], r["context"]) for r in rows if r["gold_label"] == lab), 6)
                for lab in LABELS},
        }
        p = result[name]
        print(f"{name:21s} ctx {p['mean_context_words']:.1f}  q {p['mean_question_words']:.1f}  "
              f"claim {p['mean_claim_words']:.1f}  overlap "
              + " ".join(f"{p['claim_context_overlap'][lab]:.3f}" for lab in LABELS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
