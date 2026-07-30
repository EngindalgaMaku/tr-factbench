from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from common import safe_slug

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured model x dataset x prompt-mode matrix.")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.example.yaml")
    parser.add_argument("--benchmark-config", type=Path, default=ROOT / "configs" / "benchmark.example.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    args = parse_args()
    models_cfg = load_yaml(args.models_config)
    bench_cfg = load_yaml(args.benchmark_config)
    models = [model for model in models_cfg.get("models", []) if model.get("enabled", True)]
    datasets = bench_cfg.get("datasets", [])
    prompt_modes = bench_cfg.get("prompt_modes", ["zero_shot", "few_shot_8"])
    common = bench_cfg.get("generation", {})
    output_root = resolve_repo_path(bench_cfg.get("output_root", "results/llm_baselines"))

    if not models:
        raise SystemExit("No enabled models in models config")
    if not datasets:
        raise SystemExit("No datasets in benchmark config")

    for model in models:
        for dataset in datasets:
            input_path = resolve_repo_path(dataset["path"])
            dataset_name = dataset.get("name") or input_path.stem
            for prompt_mode in prompt_modes:
                command = [
                    sys.executable,
                    str(ROOT / "run_baseline.py"),
                    "--input", str(input_path),
                    "--dataset-name", dataset_name,
                    "--model", model["id"],
                    "--model-alias", model.get("alias", model["id"]),
                    "--prompt-mode", prompt_mode,
                    "--output-root", str(output_root),
                    "--temperature", str(common.get("temperature", 0.0)),
                    "--top-p", str(common.get("top_p", 1.0)),
                    "--max-tokens", str(common.get("max_tokens", 1000)),
                    "--timeout", str(common.get("timeout", 240)),
                    "--max-retries", str(common.get("max_retries", 5)),
                    "--sleep-between-requests", str(common.get("sleep_between_requests", 0.25)),
                ]
                if bench_cfg.get("zdr", False):
                    command.append("--zdr")
                if args.dry_run:
                    command.append("--dry-run")
                print("\n$ " + " ".join(command))
                subprocess.run(command, check=True)
                if args.evaluate and not args.dry_run:
                    predictions = output_root / dataset_name / safe_slug(model.get("alias", model["id"])) / prompt_mode / "predictions.jsonl"
                    eval_command = [
                        sys.executable,
                        str(ROOT / "evaluate_baseline.py"),
                        "--dataset", str(input_path),
                        "--predictions", str(predictions),
                    ]
                    print("$ " + " ".join(eval_command))
                    subprocess.run(eval_command, check=True)


if __name__ == "__main__":
    main()
