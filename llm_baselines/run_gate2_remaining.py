"""Run the remaining frozen gold-480 Gate 2 LLM conditions sequentially.

Already complete:
  openai/gpt-4.1-mini zero_shot
  openai/gpt-4.1-mini few_shot_8

This script does not touch leftover Gemma 2/3/GLM or :free folders.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

JOBS = [
    ("google/gemma-4-26b-a4b-it", "zero_shot"),
    ("google/gemma-4-26b-a4b-it", "few_shot_8"),
    ("meta-llama/llama-3.3-70b-instruct", "zero_shot"),
    ("meta-llama/llama-3.3-70b-instruct", "few_shot_8"),
    ("qwen/qwen-2.5-72b-instruct", "zero_shot"),
    ("qwen/qwen-2.5-72b-instruct", "few_shot_8"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    results: list[tuple[str, str, int]] = []
    for index, (model, mode) in enumerate(JOBS, start=1):
        print("=" * 72, flush=True)
        print(f"[{index}/{len(JOBS)}] START {model} {mode} {utc_now()}", flush=True)
        print("=" * 72, flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "run_gold_llm_eval.py",
                "--model",
                model,
                "--prompt-mode",
                mode,
            ]
        )
        print(
            f"[{index}/{len(JOBS)}] DONE {model} {mode} "
            f"exit={completed.returncode} {utc_now()}",
            flush=True,
        )
        results.append((model, mode, completed.returncode))

    print("\nSUMMARY", flush=True)
    for model, mode, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {status:<10} {model} {mode}", flush=True)
    return 0 if all(code == 0 for _, _, code in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
