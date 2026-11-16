"""
scripts/98f_ollama_pilot.py  (pilot, 2026-11-16)

Small pilot of a local model served by Ollama, using the unchanged baseline pipeline
(llm_baselines/run_baseline.py: same Turkish system prompt, same label definitions, same JSON
schema and parsing, temperature 0). Only the endpoint differs.

Sample: 24 Gold-480 examples, 6 per label and 2 per label and domain, fixed seed 7.
Output: results/llm_baselines/pilot_ollama/<model_alias>/zero_shot/  (kept apart from the
reported gold_v1.0 results) and a short summary printed to the console.

Usage: python scripts/98f_ollama_pilot.py MODEL_NAME MODEL_ALIAS [zero_shot|few_shot_8]
"""
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl"
OUT_ROOT = ROOT / "results/llm_baselines/pilot_ollama"
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
model, alias = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "zero_shot"

rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
dom = lambda i: "finance" if int(i[-4:]) <= 160 else ("legal" if int(i[-4:]) <= 320 else "medical")
cells = defaultdict(list)
for r in rows:
    cells[(r["gold_label"], dom(r["release_example_id"]))].append(r)
rng = random.Random(7)
sample = []
for key in sorted(cells):
    sample += rng.sample(cells[key], 2)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
inp = OUT_ROOT / "pilot_input_24.jsonl"
with open(inp, "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps({**r, "example_id": r["release_example_id"]}, ensure_ascii=False) + "\n")

env = {**os.environ, "OLLAMA_API_KEY": os.environ.get("OLLAMA_API_KEY", "ollama")}
cmd = [sys.executable, str(ROOT / "llm_baselines/run_baseline.py"), "--input", str(inp), "--dataset-name", "pilot_ollama",
       "--model", model, "--model-alias", alias, "--prompt-mode", mode, "--output-root", str(OUT_ROOT.parent),
       "--endpoint", "http://localhost:11434/v1/chat/completions", "--api-key-env", "OLLAMA_API_KEY",
       "--free-tier", "--timeout", "600", "--max-retries", "2", "--sleep-between-requests", "0"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=str(ROOT / "llm_baselines"), env=env)

pred_file = next((OUT_ROOT / alias).rglob("predictions.jsonl"), None) if (OUT_ROOT / alias).exists() else None
if pred_file is None:
    sys.exit("no predictions written; check the console output above")
preds = [json.loads(l) for l in open(pred_file, encoding="utf-8") if l.strip()]
gold = {r["release_example_id"]: r["gold_label"] for r in sample}
ok = [p for p in preds if p.get("predicted_label") in LABELS]
correct = sum(p["predicted_label"] == gold[p["example_id"]] for p in ok)
conf = Counter((gold[p["example_id"]], p.get("predicted_label")) for p in preds)
print(f"\n=== {alias} {mode}: {len(preds)} responses, {len(ok)} valid labels, {correct} correct "
      f"({correct / max(1, len(preds)):.2f} accuracy over all 24)")
print("predicted label distribution:", Counter(p.get("predicted_label") for p in preds))
for g in LABELS:
    print(f"  gold {g:20s} ->", {k[1]: v for k, v in conf.items() if k[0] == g})
