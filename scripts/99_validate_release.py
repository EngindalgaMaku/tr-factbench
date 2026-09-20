"""Validate repository-level release invariants without training a model."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID_LABELS = {"supported", "partially_supported", "contradicted", "unverifiable"}
TEXT_EXTENSIONS = {
    ".md", ".py", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".cff", ".toml", ".bat", ".ps1"
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        fail(f"missing JSONL file: {path.relative_to(ROOT)}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def count_jsonl(path: Path) -> int:
    return len(load_jsonl(path))


def run_required_scripts() -> None:
    for script in (
        "scripts/00_preflight.py",
        "scripts/04_build_derived_splits.py",
        "scripts/14_validate_runs.py",
        "scripts/13_collect_results.py",
    ):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def validate_run_archive() -> tuple[int, int]:
    run_dirs = {p.name for p in (ROOT / "runs/completed").iterdir() if p.is_dir()}
    runs = list(csv.DictReader((ROOT / "results/csv/runs.csv").open(encoding="utf-8-sig", newline="")))
    metrics = list(
        csv.DictReader((ROOT / "results/csv/metrics_long_canonical.csv").open(encoding="utf-8-sig", newline=""))
    )
    run_ids = {row["run_id"] for row in runs}
    metric_ids = {row["run_id"] for row in metrics}
    if not (len(run_dirs) == len(run_ids) == len(metric_ids)):
        fail(f"run counts mismatch: dirs={len(run_dirs)}, runs.csv={len(run_ids)}, canonical={len(metric_ids)}")
    keys = [(row["run_id"], row["eval_set"]) for row in metrics]
    if len(keys) != len(set(keys)):
        fail("duplicate (run_id, eval_set) keys")
    deprecated = ["metrics_overall.csv", "metrics_overall_8sets.csv", "metrics_overall_ARCHIVED.csv"]
    if any((ROOT / "results/csv" / name).exists() for name in deprecated):
        fail("deprecated metrics_overall file exists")

    table_c = list(
        csv.DictReader((ROOT / "results/tables/table_C_target_module_ablation.csv").open(encoding="utf-8-sig"))
    )
    expected = {
        "C7_M04_ELECTRA_QV_LORA16_S42",
        "C8_M04_ELECTRA_ALLATTN_LORA16_S42",
        "G1_C6_M04_ELECTRA_LORA16_DROP10_S42",
    }
    if {row["Run ID"] for row in table_c} != expected:
        fail("Table C run IDs are not canonical")

    legacy_prefix = "data/evaluation/legacy_experiment_version/"
    for run_dir in sorted((ROOT / "runs/completed").iterdir()):
        if not run_dir.is_dir():
            continue
        manifest = load_json(run_dir / "dataset_manifest.json")
        for field, hash_field in {
            "train_file": "train_sha256",
            "dev960_file": "dev960_sha256",
            "dev600_file": "dev600_sha256",
        }.items():
            relative = str(manifest.get(field, "")).replace("\\", "/")
            allowed_dev_prefixes = (legacy_prefix, "data/evaluation/preview_v0.1/")
            if field.startswith("dev") and not any(relative.startswith(p) for p in allowed_dev_prefixes):
                fail(f"{run_dir.name}: {field} is outside the archived experiment layout: {relative}")
            if field == "train_file" and not (
                relative.startswith("data/primary/") or relative.startswith("data/derived/")
            ):
                fail(f"{run_dir.name}: train_file is outside the current data layout: {relative}")
            path = ROOT / relative
            if not path.exists():
                fail(f"{run_dir.name}: relocated dataset file does not exist: {relative}")
            if manifest.get(hash_field) != sha256(path):
                fail(f"{run_dir.name}: {hash_field} does not match relocated file")
        # Relocation history is only required for runs that used legacy experiment paths.
        is_legacy_dev = any(
            str(manifest.get(f, "")).replace("\\", "/").startswith(legacy_prefix)
            for f in ("dev960_file", "dev600_file")
        )
        if is_legacy_dev:
            relocation = manifest.get("path_relocation", {})
            required = (
                "original_recorded_train_file",
                "original_recorded_dev960_file",
                "original_recorded_dev600_file",
            )
            if not all(relocation.get(key) for key in required):
                fail(f"{run_dir.name}: missing original recorded dataset paths")

        trainer_state = run_dir / "trainer_state.json"
        if trainer_state.exists():
            checkpoint = str(load_json(trainer_state).get("best_model_checkpoint", ""))
            if re.match(r"^[A-Za-z]:[\\/]", checkpoint):
                fail(f"absolute local checkpoint path in {trainer_state.relative_to(ROOT)}")

    return len(run_dirs), len(metrics)


def validate_annotation_status() -> None:
    status = load_json(ROOT / "annotation_status.json")
    expected = {
        ("dev_stress", "annotator_1"): count_jsonl(
            ROOT / "results/annotation/dev600/dev_stress_blind_annotator_1.jsonl"
        ),
        ("dev_stress", "annotator_2"): count_jsonl(
            ROOT / "results/annotation/dev600/dev_stress_blind_annotator_2.jsonl"
        ),
        ("dev_general", "annotator_1"): count_jsonl(
            ROOT / "results/annotation/dev960/dev_general_blind_annotator_1.jsonl"
        ),
    }
    for (split, annotator), count in expected.items():
        recorded = status.get(split, {}).get(annotator, {}).get("completed")
        if recorded != count:
            fail(f"annotation status mismatch for {split}/{annotator}: file={count}, status={recorded}")


def validate_preview_layout() -> None:
    old_paths = (
        "data/primary/dev_general.jsonl",
        "data/primary/dev_stress.jsonl",
        "data/primary/dev_general_provisional_v0.1.jsonl",
        "data/primary/dev_stress_provisional_v0.1.jsonl",
    )
    for relative in old_paths:
        if (ROOT / relative).exists():
            fail(f"old mixed-layout evaluation file still exists: {relative}")

    preview_general = ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl"
    preview_stress = ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl"
    audit_general = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1.jsonl"
    audit_stress = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1.jsonl"

    for path, expected_count, label in (
        (preview_general, 960, "DEV960"),
        (preview_stress, 600, "Dev600"),
        (audit_general, 960, "DEV960 audit"),
        (audit_stress, 600, "Dev600 audit"),
    ):
        rows = load_jsonl(path)
        if len(rows) != expected_count:
            fail(f"{label} must contain {expected_count} rows, found {len(rows)}")
        seen = set()
        for row in rows:
            if row.get("annotation_status") != "single_annotator_provisional":
                fail(f"{label} rows must use annotation_status=single_annotator_provisional")
            if row.get("gold_source") != "single_annotator_blind_human" or row.get("gold_annotator") != "annotator_1":
                fail(f"{label} rows must identify annotator_1 as the blind-human label source")
            if row.get("gold_label") not in VALID_LABELS:
                fail(f"{label} contains invalid gold_label={row.get('gold_label')}")
            if row.get("example_id") in seen:
                fail(f"{label} contains duplicate example_id={row.get('example_id')}")
            seen.add(row.get("example_id"))

    if preview_general.read_bytes() != audit_general.read_bytes():
        fail("DEV960 release and audit preview files differ")
    if preview_stress.read_bytes() != audit_stress.read_bytes():
        fail("Dev600 release and audit preview files differ")

    # Domain views must be exact filtered views of the reviewed files.
    for stem, rows, expected_per_domain in (
        ("dev960_general", load_jsonl(preview_general), 320),
        ("dev600_stress", load_jsonl(preview_stress), 200),
    ):
        for domain in ("medical", "finance", "legal"):
            domain_path = ROOT / f"data/evaluation/preview_v0.1/domain/{stem}_{domain}.jsonl"
            domain_rows = load_jsonl(domain_path)
            expected_rows = [row for row in rows if row.get("domain") == domain]
            if len(domain_rows) != expected_per_domain or domain_rows != expected_rows:
                fail(f"preview domain view mismatch: {domain_path.relative_to(ROOT)}")

    # Dev600 public IDs must map one-to-one to archived IDs in source order.
    legacy_stress = load_jsonl(ROOT / "data/evaluation/legacy_experiment_version/dev_stress.jsonl")
    preview_rows = load_jsonl(preview_stress)
    map_path = ROOT / "data/manifests/dev600_legacy_to_preview_id_map.csv"
    mapping = list(csv.DictReader(map_path.open(encoding="utf-8-sig", newline="")))
    if len(mapping) != 600:
        fail(f"Dev600 ID map must contain 600 rows, found {len(mapping)}")
    if len({row["legacy_example_id"] for row in mapping}) != 600 or len(
        {row["preview_example_id"] for row in mapping}
    ) != 600:
        fail("Dev600 ID map is not one-to-one")
    for legacy, preview, map_row in zip(legacy_stress, preview_rows, mapping):
        expected = {
            "legacy_example_id": legacy["example_id"],
            "preview_example_id": preview["example_id"],
            "context_id": legacy["context_id"],
            "domain": legacy["domain"],
        }
        if map_row != expected:
            fail(f"Dev600 ID map mismatch for archived ID {legacy['example_id']}")
        if legacy["context_id"] != preview["context_id"] or legacy["domain"] != preview["domain"]:
            fail(f"Dev600 source/preview order mismatch for archived ID {legacy['example_id']}")

    # Layout manifest paths and all registered hashes must resolve.
    layout = load_json(ROOT / "data/manifests/evaluation_layout_manifest_v0.1.json")
    registered_paths = {
        "preview_dev_general": layout["recommended_preview_evaluation"]["dev_general"],
        "preview_dev_stress": layout["recommended_preview_evaluation"]["dev_stress"],
        "legacy_dev_general": layout["archived_experiment_evaluation"]["dev_general"],
        "legacy_dev_stress": layout["archived_experiment_evaluation"]["dev_stress"],
    }
    for relative in layout.get("archived_domain_evaluation_files", []):
        registered_paths[Path(relative).stem] = relative
    for relative in layout.get("preview_domain_evaluation_files", []):
        registered_paths[f"preview_{Path(relative).stem}"] = relative
    if set(layout.get("sha256", {})) != set(registered_paths):
        fail("evaluation layout manifest hash keys do not match registered files")
    for key, relative in registered_paths.items():
        path = ROOT / relative
        if not path.exists() or layout["sha256"].get(key) != sha256(path):
            fail(f"evaluation layout hash mismatch: {key} -> {relative}")


def validate_llm_artifacts() -> None:
    table = ROOT / "results/tables/table_I_llm_baselines.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 16:
        fail(f"LLM table must have 16 rows, found {len(rows)}")
    if len({row["model_alias"] for row in rows}) != 4:
        fail("LLM table must contain four canonical model aliases")
    if {row["prompt_mode"] for row in rows} != {"zero_shot", "few_shot_8"}:
        fail("LLM prompt modes mismatch")
    if {row["dataset"] for row in rows} != {"dev_general", "dev_stress"}:
        fail("LLM dataset matrix mismatch")
    for row in rows:
        run_dir = ROOT / "results/llm_baselines" / row["dataset"] / row["model_alias"] / row["prompt_mode"]
        expected = 960 if row["dataset"] == "dev_general" else 600
        if count_jsonl(run_dir / "predictions.jsonl") != expected:
            fail(f"incomplete LLM prediction file: {run_dir.relative_to(ROOT)}")
        if not (run_dir / "metrics/metrics.json").exists():
            fail(f"missing LLM metrics: {run_dir.relative_to(ROOT)}")

    lock_text = (ROOT / "final_config.lock.yaml").read_text(encoding="utf-8-sig")
    match = re.search(r'(?ms)^llm_baselines:\s+.*?^  table_sha256: "([0-9a-f]{64})"', lock_text)
    if not match or match.group(1) != sha256(table):
        fail("final_config.lock.yaml LLM table hash is stale")


def _is_untracked_gold_llm_tree(path: Path) -> bool:
    """Local Gate 2 leftovers under results/llm_baselines/gold_v1.0/ are not release artifacts."""
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    return relative.startswith("results/llm_baselines/gold_v1.0/")


def validate_portability_and_text() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if _is_untracked_gold_llm_tree(path):
            continue
        if path.read_bytes().startswith(b"\xef\xbb\xbf"):
            fail(f"UTF-8 BOM found in {path.relative_to(ROOT)}")

    # Only the published DEV LLM matrices are release-active. gold_v1.0/ is a
    # local leftover (absolute Windows paths, incomplete runs) and is gitignored.
    active_manifests = []
    for dataset in ("dev_general", "dev_stress"):
        active_manifests.extend((ROOT / "results/llm_baselines" / dataset).glob("*/*/manifest.json"))
    active_manifests += [
        ROOT / "results/annotation/dev600/llm_preannotation_manifest.json",
        ROOT / "results/annotation/dev960/llm_preannotation_manifest.json",
        ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1_manifest.json",
        ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1_manifest.json",
        ROOT / "data/manifests/preview_evaluation_manifest_v0.1.json",
        ROOT / "data/manifests/evaluation_layout_manifest_v0.1.json",
    ]
    for path in active_manifests:
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text):
            fail(f"absolute Windows path in active manifest: {path.relative_to(ROOT)}")

    public_docs = (
        "README.md",
        "README_TR.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "CITATION.cff",
        "data/DATA_LICENSE.md",
        "data/evaluation/preview_v0.1/README.md",
    )
    defensive_phrases = (
        "does not block",
        "not a fully adjudicated",
        "must not be described",
        "must not be treated",
        "not fully adjudicated gold",
        "not required for the preview",
        "intentionally deferred",
    )
    for relative in public_docs:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8-sig")
        if any(token in text for token in ("Ã", "Ä", "Å", "â†", "â€“", "â€”")):
            fail(f"mojibake in {relative}")
        lower = text.lower()
        if any(phrase in lower for phrase in defensive_phrases):
            fail(f"internal or defensive release wording remains in {relative}")
        if "288/960" in text:
            fail(f"stale DEV960 annotation progress in {relative}")

    if any((ROOT / "results/tables" / name).exists() for name in (
        "table_E_cross_domain_matrix.csv",
        "table_E_cross_domain_matrix.md",
    )):
        fail("empty cross-domain placeholder table is present")

    train_text = (ROOT / "src/train.py").read_text(encoding="utf-8-sig")
    if "path_prefix" in train_text:
        fail("undefined legacy path_prefix remains in src/train.py")
    for token in (
        'Path("data/evaluation/preview_v0.1")',
        'preview_domain = preview_eval / "domain"',
        '"dev_general.jsonl"',
        '"dev_stress.jsonl"',
        '"dev960_general_{args.domain}.jsonl"',
        '"dev600_stress_medical.jsonl"',
        '"dev600_stress_finance.jsonl"',
        '"dev600_stress_legal.jsonl"',
    ):
        if token not in train_text:
            fail(f"src/train.py is missing preview path resolver token: {token}")

    lock = (ROOT / "final_config.lock.yaml").read_text(encoding="utf-8-sig")
    if "q1_pipeline/" in lock or "CHECK:" in lock:
        fail("placeholder or old path in final_config.lock.yaml")


def validate_preview_evaluation_run() -> None:
    """Validate the H1_DROP10_S45_PREVIEW run under runs/preview_evaluations/."""
    run_dir = ROOT / "runs/preview_evaluations/H1_DROP10_S45_PREVIEW"
    if not run_dir.is_dir():
        fail("missing preview evaluation run directory: runs/preview_evaluations/H1_DROP10_S45_PREVIEW")

    # --- required metadata files ---
    for fname in ("config.json", "dataset_manifest.json", "metrics.json",
                  "model_manifest.json", "trainable_parameters.json",
                  "trainer_state.json", "validation_report.json"):
        if not (run_dir / fname).exists():
            fail(f"H1_DROP10_S45_PREVIEW: missing {fname}")

    # --- no absolute local paths ---
    trainer_state = load_json(run_dir / "trainer_state.json")
    ckpt = str(trainer_state.get("best_model_checkpoint", ""))
    if re.match(r"^[A-Za-z]:[\\/]", ckpt):
        fail(f"H1_DROP10_S45_PREVIEW/trainer_state.json: absolute local path found: {ckpt}")

    # --- no ckpt/ weights committed ---
    ckpt_dir = run_dir / "ckpt"
    if ckpt_dir.exists() and any(ckpt_dir.rglob("*.safetensors")):
        fail("H1_DROP10_S45_PREVIEW: adapter weights found in ckpt/ — should be gitignored")

    # --- dataset manifest ---
    dm = load_json(run_dir / "dataset_manifest.json")
    preview_dev960 = ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl"
    preview_dev600 = ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl"
    for field, expected_path in (("dev960_file", preview_dev960), ("dev600_file", preview_dev600)):
        recorded = str(dm.get(field, "")).replace("\\", "/")
        if not recorded.startswith("data/evaluation/preview_v0.1/"):
            fail(f"H1_DROP10_S45_PREVIEW: {field} not pointing at preview files: {recorded}")
        actual_hash = sha256(ROOT / recorded)
        stored_hash = dm.get({"dev960_file": "dev960_sha256", "dev600_file": "dev600_sha256"}[field])
        if actual_hash != stored_hash:
            fail(f"H1_DROP10_S45_PREVIEW: {field} hash mismatch")

    # --- prediction file counts and label coverage ---
    for pred_file, expected_n, gold_source in (
        ("predictions_DEV960-GEN.jsonl", 960, preview_dev960),
        ("predictions_DEV600-STR.jsonl", 600, preview_dev600),
    ):
        preds = load_jsonl(run_dir / pred_file)
        if len(preds) != expected_n:
            fail(f"H1_DROP10_S45_PREVIEW/{pred_file}: expected {expected_n} rows, got {len(preds)}")
        gold_records = load_jsonl(gold_source)
        gold_by_id = {r["example_id"]: r["gold_label"] for r in gold_records}
        pred_ids = {r["example_id"] for r in preds}
        if pred_ids != set(gold_by_id):
            fail(f"H1_DROP10_S45_PREVIEW/{pred_file}: ID mismatch with preview gold file")
        for r in preds:
            if r.get("gold_label") != gold_by_id.get(r["example_id"]):
                fail(f"H1_DROP10_S45_PREVIEW/{pred_file}: gold label mismatch for example_id={r['example_id']}")

    # --- metrics sanity (non-zero macro F1) ---
    metrics = load_json(run_dir / "metrics.json")
    for key in ("d960_macro_f1", "d600_macro_f1"):
        val = metrics.get(key, 0)
        if not (0.5 <= val <= 1.0):
            fail(f"H1_DROP10_S45_PREVIEW/metrics.json: {key}={val} out of plausible range")


def main() -> None:
    run_required_scripts()
    run_count, metric_count = validate_run_archive()
    validate_annotation_status()
    validate_preview_layout()
    validate_preview_evaluation_run()
    validate_llm_artifacts()
    validate_portability_and_text()

    required_docs = (
        "docs/LABEL_GUIDE.md",
        "docs/LABEL_GUIDE_TR.md",
        "reports/LLM_BASELINES.md",
        "reports/LLM_BASELINES_TR.md",
    )
    for relative in required_docs:
        if not (ROOT / relative).exists():
            fail(f"missing release document: {relative}")

    print(f"[PASS] release validation: {run_count} runs, {metric_count} canonical rows, no duplicates")
    print("       preview layout, hashes, mappings, portability, and text checks passed")
    print("       preview evaluation run H1_DROP10_S45_PREVIEW validated")


if __name__ == "__main__":
    main()
