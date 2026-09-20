"""Generate the TR-FactBench release lock from canonical repository artifacts.

The lock is computed from run-level files and canonical metrics. It separates:
1) the historical G1 dropout ablation, and
2) the H1 final validation series produced after explicitly wiring Trainer
   ``seed`` and ``data_seed``.

Do not edit ``final_config.lock.yaml`` manually; rerun this script instead.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results/csv/metrics_long_canonical.csv"
RUNS_CSV = ROOT / "results/csv/runs.csv"
OUT = ROOT / "final_config.lock.yaml"
ANNOTATION_STATUS = ROOT / "annotation_status.json"
LLM_TABLE = ROOT / "results/tables/table_I_llm_baselines.csv"
DEV600_PREVIEW = ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl"
DEV600_PREVIEW_MANIFEST = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1_manifest.json"
DEV960_PREVIEW = ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl"
DEV960_PREVIEW_MANIFEST = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1_manifest.json"

FINAL_RUN_ID = "H1_DROP10_S45"
SENTINEL_RUN_ID = "H1_SENTINEL_DROP10_S45_RERUN"
FINAL_SEED = 45
REPRO_THRESHOLD = 0.010
SEEDS_5 = [42, 43, 44, 45, 46]
SEEDS_3 = [42, 43, 44]

G1_PREFIXES = {
    "drop0": "G1_C6_M04_ELECTRA_LORA16_DROP0_S",
    "drop5": "G1_C6_M04_ELECTRA_LORA16_DROP5_S",
    "drop10": "G1_C6_M04_ELECTRA_LORA16_DROP10_S",
}
H1_PREFIX = "H1_DROP10_S"
H2_PREFIX = "H2_DROP0_S"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q(value: object) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "null"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


if not METRICS.exists():
    raise SystemExit(f"Missing canonical metrics: {METRICS}")

metrics = list(csv.DictReader(METRICS.open(encoding="utf-8-sig", newline="")))
lookup = {(row["run_id"], row["eval_set"]): float(row["macro_f1"]) for row in metrics}
all_run_ids = {row["run_id"] for row in metrics}

if RUNS_CSV.exists():
    runs = list(csv.DictReader(RUNS_CSV.open(encoding="utf-8-sig", newline="")))
    statuses = collections.Counter(row.get("status") or row.get("qc_status") or "unknown" for row in runs)
else:
    statuses = collections.Counter()
    for run_dir in sorted((ROOT / "runs/completed").iterdir(), key=lambda p: p.name):
        validation = run_dir / "validation_report.json"
        if validation.exists():
            statuses[load_json(validation).get("qc_status", "unknown")] += 1


def seed_stats(prefix: str, seeds: list[int]) -> dict:
    per_seed: dict[int, dict[str, float | None]] = {}
    dev960: list[float] = []
    dev600: list[float] = []
    for seed in seeds:
        run_id = f"{prefix}{seed}"
        score960 = lookup.get((run_id, "DEV960-GEN"))
        score600 = lookup.get((run_id, "DEV600-STR"))
        per_seed[seed] = {"dev960": score960, "dev600": score600}
        if score960 is not None and score600 is not None:
            dev960.append(score960)
            dev600.append(score600)
    return {
        "seeds_present": [seed for seed, values in per_seed.items() if values["dev960"] is not None and values["dev600"] is not None],
        "per_seed": per_seed,
        "dev960_mean": statistics.mean(dev960) if dev960 else None,
        "dev960_sd": statistics.stdev(dev960) if len(dev960) >= 2 else None,
        "dev600_mean": statistics.mean(dev600) if dev600 else None,
        "dev600_sd": statistics.stdev(dev600) if len(dev600) >= 2 else None,
    }


g1_drop0 = seed_stats(G1_PREFIXES["drop0"], SEEDS_5)
g1_drop5 = seed_stats(G1_PREFIXES["drop5"], SEEDS_3)
g1_drop10 = seed_stats(G1_PREFIXES["drop10"], SEEDS_5)
h1_drop10 = seed_stats(H1_PREFIX, SEEDS_5)
h2_drop0 = seed_stats(H2_PREFIX, SEEDS_5)

final_run_dir = ROOT / "runs/completed" / FINAL_RUN_ID
sentinel_run_dir = ROOT / "runs/completed" / SENTINEL_RUN_ID
final_exists = final_run_dir.exists()
sentinel_exists = sentinel_run_dir.exists()

sentinel_data: dict[str, float | bool] | None = None
if final_exists and sentinel_exists:
    original = load_json(final_run_dir / "metrics.json")
    rerun = load_json(sentinel_run_dir / "metrics.json")
    delta960 = rerun["d960_macro_f1"] - original["d960_macro_f1"]
    delta600 = rerun["d600_macro_f1"] - original["d600_macro_f1"]
    reproducibility_pass = max(abs(delta960), abs(delta600)) <= REPRO_THRESHOLD
    sentinel_data = {
        "original_dev960": original["d960_macro_f1"],
        "sentinel_dev960": rerun["d960_macro_f1"],
        "delta_dev960": delta960,
        "original_dev600": original["d600_macro_f1"],
        "sentinel_dev600": rerun["d600_macro_f1"],
        "delta_dev600": delta600,
        "reproducibility_pass": reproducibility_pass,
    }

if final_exists and len(h1_drop10["seeds_present"]) == 5:
    if sentinel_data and sentinel_data["reproducibility_pass"]:
        model_status = "reproducibility_passed_preview_ready"
    elif sentinel_exists:
        model_status = "selected_reproducibility_threshold_not_met"
    else:
        model_status = "selected_pending_reproducibility"
elif final_exists:
    model_status = "selected_partial_seed_validation"
else:
    model_status = "pending_final_model_selection"

if final_exists:
    config = load_json(final_run_dir / "config.json")
    parameters = load_json(final_run_dir / "trainable_parameters.json")
else:
    raise SystemExit(f"Final run not found: {FINAL_RUN_ID}")

final_dev960 = lookup.get((FINAL_RUN_ID, "DEV960-GEN"))
final_dev600 = lookup.get((FINAL_RUN_ID, "DEV600-STR"))
if final_dev960 is None or final_dev600 is None:
    raise SystemExit(f"Final run metrics are missing from {METRICS}: {FINAL_RUN_ID}")
final_average = (final_dev960 + final_dev600) / 2

# A generated file cannot safely embed the current commit hash: committing the
# regenerated lock changes HEAD again. Keep the lock deterministic and let the
# Git tag/release identify the repository revision.
commit = "not_embedded_use_release_tag"

annotation_status = load_json(ANNOTATION_STATUS) if ANNOTATION_STATUS.exists() else {}
llm_rows = list(csv.DictReader(LLM_TABLE.open(encoding="utf-8-sig", newline=""))) if LLM_TABLE.exists() else []
preview_manifest = load_json(DEV600_PREVIEW_MANIFEST) if DEV600_PREVIEW_MANIFEST.exists() else {}
preview960_manifest = load_json(DEV960_PREVIEW_MANIFEST) if DEV960_PREVIEW_MANIFEST.exists() else {}

lines = [
    "# TR-FactBench release lock",
    "# Generated automatically by scripts/35_generate_lock_yaml.py",
    "# Do not edit manually — rerun the script to regenerate.",
    "release:",
    '  version: "0.1.0-preview"',
    '  status: "preview"',
    f"  final_model_status: {q(model_status)}",
    f"  git_commit: {q(commit)}",
    "",
    "canonical_artifacts:",
    '  metrics_file: "results/csv/metrics_long_canonical.csv"',
    f"  metrics_sha256: {q(sha256(METRICS))}",
    f"  total_runs: {len(all_run_ids)}",
    f"  canonical_metric_rows: {len(metrics)}",
    f"  successful_runs: {statuses.get('success', 0)}",
    f"  degenerate_class_collapse: {statuses.get('degenerate_class_collapse', 0)}",
    f"  degenerate_skewed: {statuses.get('degenerate_skewed', 0)}",
    "",
    "final_selected_model:",
    f"  run_id: {q(FINAL_RUN_ID)}",
    f"  run_exists_in_repo: {str(final_exists).lower()}",
    f"  model_id: {q(config['model_name'])}",
    '  method: "lora"',
    f"  rank: {config['lora_r']}",
    f"  alpha: {config['lora_alpha']}",
    f"  dropout: {config['lora_dropout']}",
    f"  seed: {FINAL_SEED}",
    f"  trainer_seed: {config.get('seed', FINAL_SEED)}",
    f"  trainer_data_seed: {config.get('seed', FINAL_SEED)}",
    f"  target_modules: {q(config['lora_target_modules'])}",
    f"  learning_rate: {config['learning_rate']}",
    f"  batch_size: {config['batch_size']}",
    f"  epochs: {config['epochs']}",
    f"  max_length: {config['max_length']}",
    f"  warmup_ratio: {config['warmup_ratio']}",
    f"  weight_decay: {config['weight_decay']}",
    f"  trainable_params: {parameters['num_trainable_params']}",
    f"  total_params: {parameters['num_total_params']}",
    f"  trainable_percentage: {parameters['trainable_percentage']}",
    f"  dev960_macro_f1: {final_dev960:.4f}",
    f"  dev600_macro_f1: {final_dev600:.4f}",
    f"  two_set_average: {final_average:.4f}",
    "  selection_rationale: >",
    "    The final seed-fixed comparison evaluates dropout=0.10 (H1) and dropout=0.00 (H2)",
    "    under the same Trainer seed/data_seed protocol across seeds 42-46. H2 is stronger",
    "    on DEV960, while H1 is stronger and more stable on DEV600-STR and has a slightly",
    "    higher two-set mean. H1 seed 45 is the selected checkpoint within the chosen H1",
    f"    configuration (DEV960={final_dev960:.4f}, DEV600-STR={final_dev600:.4f}, AVG={final_average:.4f}).",
    "    NOTE: H2_DROP0_S45 achieves the highest individual scores on both sets",
    "    (DEV960=0.9058, DEV600-STR=0.8803). It may be reported alongside H1_DROP10_S45",
    "    in the paper as the best single-seed result, while H1 is retained as the",
    "    primary selection based on 5-seed stability criteria.",
    f"  sentinel_run_id: {q(SENTINEL_RUN_ID)}",
    f"  sentinel_exists: {str(sentinel_exists).lower()}",
]

if sentinel_data:
    lines += [
        "  sentinel_comparison:",
        f"    original_dev960: {sentinel_data['original_dev960']:.4f}",
        f"    sentinel_dev960: {sentinel_data['sentinel_dev960']:.4f}",
        f"    delta_dev960: {sentinel_data['delta_dev960']:+.4f}",
        f"    original_dev600: {sentinel_data['original_dev600']:.4f}",
        f"    sentinel_dev600: {sentinel_data['sentinel_dev600']:.4f}",
        f"    delta_dev600: {sentinel_data['delta_dev600']:+.4f}",
        f"    reproducibility_pass: {str(bool(sentinel_data['reproducibility_pass'])).lower()}",
        f"    threshold: {REPRO_THRESHOLD}",
        "    interpretation: \"Performance-level reproducibility; not exact prediction-level identity.\"",
        "    threshold_rationale: >",
        "      A predeclared absolute Macro-F1 tolerance of 0.010 is used for full training",
        "      reruns. Both H1 sentinel deltas are computed directly from the archived metrics.",
    ]

lines += [
    "",
    "historical_g1_dropout_ablation:",
    '  comparability_note: "All rows use the same pre-seed-fix G1 training pipeline."',
]
for name, stats, note in [
    ("drop0", g1_drop0, "Historical seed-46 collapse observed; not reproduced in the seed-fixed H2 rerun."),
    ("drop5", g1_drop5, "Three-seed screening only; eliminated before five-seed extension."),
    ("drop10", g1_drop10, "Historical screening result; final selection is based on the seed-fixed H1/H2 comparison."),
]:
    lines += [
        f"  {name}:",
        f"    seeds_present: {stats['seeds_present']}",
        f"    dev960_mean: {fmt(stats['dev960_mean'])}",
        f"    dev960_sd: {fmt(stats['dev960_sd'])}",
        f"    dev600_mean: {fmt(stats['dev600_mean'])}",
        f"    dev600_sd: {fmt(stats['dev600_sd'])}",
        f"    note: {q(note)}",
    ]

lines += [
    "",
    "final_seed_fixed_dropout_comparison:",
    '  comparability_note: "H1 and H2 use the same Trainer seed/data_seed protocol; intended difference is LoRA dropout."',
    '  selection_basis: "Five-seed configuration-level behavior, not the best individual seed."',
    '  selected_series: "H1_DROP10"',
    "  h1_drop10:",
    f"    seeds_present: {h1_drop10['seeds_present']}",
    f"    dev960_mean: {fmt(h1_drop10['dev960_mean'])}",
    f"    dev960_sd: {fmt(h1_drop10['dev960_sd'])}",
    f"    dev600_mean: {fmt(h1_drop10['dev600_mean'])}",
    f"    dev600_sd: {fmt(h1_drop10['dev600_sd'])}",
    f"    two_set_mean: {fmt((h1_drop10['dev960_mean'] + h1_drop10['dev600_mean']) / 2 if h1_drop10['dev960_mean'] is not None and h1_drop10['dev600_mean'] is not None else None)}",
    "  h2_drop0:",
    f"    seeds_present: {h2_drop0['seeds_present']}",
    f"    dev960_mean: {fmt(h2_drop0['dev960_mean'])}",
    f"    dev960_sd: {fmt(h2_drop0['dev960_sd'])}",
    f"    dev600_mean: {fmt(h2_drop0['dev600_mean'])}",
    f"    dev600_sd: {fmt(h2_drop0['dev600_sd'])}",
    f"    two_set_mean: {fmt((h2_drop0['dev960_mean'] + h2_drop0['dev600_mean']) / 2 if h2_drop0['dev960_mean'] is not None and h2_drop0['dev600_mean'] is not None else None)}",
    "  interpretation: >",
    "    H2/dropout=0.00 has the higher DEV960 mean. H1/dropout=0.10 has the higher",
    "    DEV600-STR mean, lower DEV600-STR standard deviation, and slightly higher two-set",
    "    mean; therefore H1 is retained as the selected configuration.",
]

lines += ["", "datasets:"]
dataset_specs = [
    ("train", ROOT / "data/primary/train.jsonl", 4800),
    ("dev_general", ROOT / "data/evaluation/legacy_experiment_version/dev_general.jsonl", 960),
    ("dev_stress", ROOT / "data/evaluation/legacy_experiment_version/dev_stress.jsonl", 600),
]
for key, path, count in dataset_specs:
    lines += [
        f"  {key}:",
        f"    file: {q(path.relative_to(ROOT).as_posix())}",
        f"    records: {count}",
        f"    sha256: {q(sha256(path))}",
    ]

preview_ready = (
    annotation_status.get("dev_stress", {}).get("annotator_1", {}).get("completed", 0) == 600
    and annotation_status.get("dev_general", {}).get("annotator_1", {}).get("completed", 0) == 960
    and DEV600_PREVIEW.exists()
    and DEV960_PREVIEW.exists()
)

lines += [
    "",
    "annotation_snapshot:",
    f"  status_file: {q(ANNOTATION_STATUS.relative_to(ROOT).as_posix())}",
    f"  last_updated: {q(annotation_status.get('last_updated', 'unknown'))}",
    f"  preview_release_ready: {str(preview_ready).lower()}",
    f"  dev600_annotator_1: {annotation_status.get('dev_stress', {}).get('annotator_1', {}).get('completed', 0)}",
    f"  dev600_annotator_2: {annotation_status.get('dev_stress', {}).get('annotator_2', {}).get('completed', 0)}",
    f"  dev960_annotator_1: {annotation_status.get('dev_general', {}).get('annotator_1', {}).get('completed', 0)}",
    f"  dev960_annotator_2: {annotation_status.get('dev_general', {}).get('annotator_2', {}).get('completed', 0)}",
    '  adjudication_status: "not_required_preview_limitation"',
    f"  provisional_dev960_file: {q(DEV960_PREVIEW.relative_to(ROOT).as_posix())}",
    f"  provisional_dev960_sha256: {q(sha256(DEV960_PREVIEW)) if DEV960_PREVIEW.exists() else q('missing')}",
    f"  provisional_dev960_label_revisions: {preview960_manifest.get('label_changes_from_original_gold', 0)}",
    f"  provisional_dev600_file: {q(DEV600_PREVIEW.relative_to(ROOT).as_posix())}",
    f"  provisional_dev600_sha256: {q(sha256(DEV600_PREVIEW)) if DEV600_PREVIEW.exists() else q('missing')}",
    f"  provisional_dev600_label_revisions: {preview_manifest.get('label_changes_from_original_gold', 0)}",
    f"  provisional_dev600_claim_fixes: {len(preview_manifest.get('claim_fixes', []))}",
    '  evaluation_version_note: "Archived scores use the original files. The preview changes 2/960 DEV960 labels and 3/600 Dev600 labels, with one Dev600 claim correction; runs were not repeated."',
    "",
    "llm_baselines:",
    f"  table_file: {q(LLM_TABLE.relative_to(ROOT).as_posix())}",
    f"  table_sha256: {q(sha256(LLM_TABLE)) if LLM_TABLE.exists() else q('missing')}",
    f"  experiment_conditions: {len(llm_rows)}",
    f"  models: {len(set(row.get('model_alias', '') for row in llm_rows)) if llm_rows else 0}",
    f"  datasets: {len(set(row.get('dataset', '') for row in llm_rows)) if llm_rows else 0}",
    '  prompt_modes: ["zero_shot", "few_shot_8"]',
]

if sentinel_data and sentinel_data["reproducibility_pass"]:
    sentinel_requirement = (
        f'  - "~~Run H1 reproducibility sentinel~~ DONE ({SENTINEL_RUN_ID}; '
        f'max_abs_delta={max(abs(float(sentinel_data["delta_dev960"])), abs(float(sentinel_data["delta_dev600"]))):.4f})."'
    )
else:
    sentinel_requirement = f'  - "Run or resolve the H1 reproducibility sentinel for {FINAL_RUN_ID}."'

lines += [
    "",
    "pending_stable_release_requirements:",
    sentinel_requirement,
    '  - "Complete the second blind-human pass for both evaluation sets."',
    '  - "Compute pre-adjudication IAA and complete adjudication."',
    '  - "Complete the stratified, context-grouped train audit sample with two blind annotators."',
    '  - "Verify LoRA adapter loading and inference in a clean environment before Hugging Face publication."',
    '  - "Review applicable source terms before any rights-cleared commercial redistribution claim."',
    "",
    "commands:",
    '  preflight: "python scripts/00_preflight.py"',
    '  build_derived: "python scripts/04_build_derived_splits.py"',
    '  collect_results: "python scripts/13_collect_results.py"',
    '  validate_runs: "python scripts/14_validate_runs.py"',
    '  generate_final_dropout_report: "python scripts/37_final_dropout_report.py"',
    '  update_annotation_status: "python scripts/34_update_annotation_status.py"',
    '  build_preview_files: "python scripts/43_build_single_annotator_preview.py"',
    '  generate_llm_report: "python scripts/53_generate_llm_report.py"',
    '  generate_lock: "python scripts/35_generate_lock_yaml.py"',
    '  validate_release: "python scripts/99_validate_release.py"',
]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] {OUT}")
print(f"     model_status : {model_status}")
print(f"     total_runs   : {len(all_run_ids)}")
print(f"     metric_rows  : {len(metrics)}")
print(f"     successful   : {statuses.get('success', 0)}")
print(f"     H1 seeds     : {h1_drop10['seeds_present']}")
print(f"     H2 seeds     : {h2_drop0['seeds_present']}")
