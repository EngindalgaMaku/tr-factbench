"""Build the v0.1 single-annotator preview evaluation files.

The archived experiment files remain unchanged so that published run metrics can be
reproduced. New evaluations should use the reviewed preview files produced here.

- DEV960 is rebuilt from the archived experimental file plus blind annotator 1 labels.
- DEV600 uses the already reviewed single-annotator artifact, which also contains one
  documented claim correction and normalized public identifiers.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "TR-FactBench v0.1.0-preview"
LABELS = {"supported", "partially_supported", "contradicted", "unverifiable"}
STATUS = "single_annotator_provisional"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_annotations(rows: list[dict], expected: int, annotator: str) -> dict[str, str]:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} annotations, found {len(rows)}")
    mapping: dict[str, str] = {}
    for row in rows:
        if row.get("review_status") != "final":
            raise ValueError(f"Non-final annotation: {row.get('example_id')}")
        if row.get("annotator_id") != annotator:
            raise ValueError(f"Unexpected annotator: {row.get('annotator_id')}")
        label = row.get("final_label")
        if label not in LABELS:
            raise ValueError(f"Unknown annotation label: {label}")
        example_id = row.get("example_id")
        if example_id in mapping:
            raise ValueError(f"Duplicate annotation ID: {example_id}")
        mapping[example_id] = label
    return mapping


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["gold_label"] for row in rows)
    return {label: counts[label] for label in sorted(LABELS)}


def build_dev960() -> dict:
    source = ROOT / "data/evaluation/legacy_experiment_version/dev_general.jsonl"
    annotation = ROOT / "results/annotation/dev960/dev_general_blind_annotator_1.jsonl"
    data_output = ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl"
    audit_output = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1.jsonl"
    manifest_output = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1_manifest.json"

    source_rows = load_jsonl(source)
    annotation_rows = load_jsonl(annotation)
    labels = validate_annotations(annotation_rows, 960, "annotator_1")
    if {row["example_id"] for row in source_rows} != set(labels):
        raise ValueError("DEV960 source and annotation IDs do not match")

    preview_rows: list[dict] = []
    changes: list[dict] = []
    for row in source_rows:
        reviewed = dict(row)
        old_label = reviewed["gold_label"]
        new_label = labels[reviewed["example_id"]]
        reviewed["gold_label"] = new_label
        reviewed["dataset_version"] = VERSION
        reviewed["annotation_status"] = STATUS
        reviewed["gold_source"] = "single_annotator_blind_human"
        reviewed["gold_annotator"] = "annotator_1"
        preview_rows.append(reviewed)
        if old_label != new_label:
            changes.append({"example_id": reviewed["example_id"], "from": old_label, "to": new_label})

    write_jsonl(data_output, preview_rows)
    write_jsonl(audit_output, preview_rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": VERSION,
        "output_file": audit_output.relative_to(ROOT).as_posix(),
        "data_release_file": data_output.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(audit_output),
        "source_dataset": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "annotation_file": annotation.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation),
        "total_examples": len(preview_rows),
        "gold_source": "single_annotator_blind_human",
        "gold_annotator": "annotator_1",
        "label_changes_from_original_gold": len(changes),
        "label_changes": changes,
        "claim_fixes": [],
        "label_distribution": label_counts(preview_rows),
        "provisional": True,
        "status": "provisional_single_annotator_v0.1",
        "evaluation_note": (
            "Archived encoder and LLM scores use the pre-review DEV960 file. "
            f"The reviewed preview changes {len(changes)}/960 labels; archived runs were not repeated."
        ),
    }
    write_json(manifest_output, manifest)
    return manifest


def build_dev600() -> dict:
    audit_output = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1.jsonl"
    manifest_output = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1_manifest.json"
    data_output = ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl"
    annotation = ROOT / "results/annotation/dev600/dev_stress_blind_annotator_1.jsonl"

    annotation_rows = load_jsonl(annotation)
    validate_annotations(annotation_rows, 600, "annotator_1")
    preview_rows = load_jsonl(audit_output)
    if len(preview_rows) != 600:
        raise ValueError(f"Expected 600 DEV600 preview rows, found {len(preview_rows)}")
    for row in preview_rows:
        if row.get("gold_label") not in LABELS:
            raise ValueError(f"Unknown DEV600 label: {row.get('gold_label')}")
        row["dataset_version"] = VERSION
        row["annotation_status"] = STATUS
        row["gold_source"] = "single_annotator_blind_human"
        row["gold_annotator"] = "annotator_1"

    write_jsonl(audit_output, preview_rows)
    write_jsonl(data_output, preview_rows)
    manifest = json.loads(manifest_output.read_text(encoding="utf-8-sig"))
    manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_release_file": data_output.relative_to(ROOT).as_posix(),
            "output_sha256": sha256(audit_output),
            "annotation_sha256": sha256(annotation),
            "total_examples": len(preview_rows),
            "gold_source": "single_annotator_blind_human",
            "gold_annotator": "annotator_1",
            "label_distribution": label_counts(preview_rows),
            "provisional": True,
            "status": "provisional_single_annotator_v0.1",
        }
    )
    write_json(manifest_output, manifest)
    return manifest


def main() -> None:
    dev960 = build_dev960()
    dev600 = build_dev600()
    combined = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": VERSION,
        "release_status": "research_preview_single_annotator_reviewed",
        "recommended_evaluation_files": {
            "dev_general": "data/evaluation/preview_v0.1/dev_general.jsonl",
            "dev_stress": "data/evaluation/preview_v0.1/dev_stress.jsonl",
        },
        "archived_experiment_files": {
            "dev_general": "data/evaluation/legacy_experiment_version/dev_general.jsonl",
            "dev_stress": "data/evaluation/legacy_experiment_version/dev_stress.jsonl",
        },
        "annotation_summary": {
            "dev_general": {
                "annotator_1": "960/960 completed",
                "annotator_2": "not started",
                "label_changes": dev960["label_changes_from_original_gold"],
                "label_distribution": dev960["label_distribution"],
            },
            "dev_stress": {
                "annotator_1": "600/600 completed",
                "annotator_2": "80/600 completed",
                "label_changes": dev600.get("label_changes_from_original_gold", 0),
                "claim_fixes": len(dev600.get("claim_fixes", [])),
                "label_distribution": dev600["label_distribution"],
            },
        },
        "usage_note": (
            "Use the provisional files for new preview evaluations. Archived benchmark scores remain tied "
            "to the original experimental files and are retained for reproducibility. The second blind pass "
            "and adjudication are planned for a later stable release and are not blockers for v0.1.0-preview."
        ),
    }
    write_json(ROOT / "data/manifests/preview_evaluation_manifest_v0.1.json", combined)

    legacy_root = ROOT / "data/evaluation/legacy_experiment_version"
    domain_files = sorted((legacy_root / "domain").glob("*.jsonl"))
    layout_manifest = {
        "layout_version": "v0.1.0-preview-evaluation-layout",
        "training_file": "data/primary/train.jsonl",
        "recommended_preview_evaluation": combined["recommended_evaluation_files"],
        "archived_experiment_evaluation": combined["archived_experiment_files"],
        "archived_domain_evaluation_files": [p.relative_to(ROOT).as_posix() for p in domain_files],
        "sha256": {
            "preview_dev_general": sha256(ROOT / combined["recommended_evaluation_files"]["dev_general"]),
            "preview_dev_stress": sha256(ROOT / combined["recommended_evaluation_files"]["dev_stress"]),
            "legacy_dev_general": sha256(ROOT / combined["archived_experiment_files"]["dev_general"]),
            "legacy_dev_stress": sha256(ROOT / combined["archived_experiment_files"]["dev_stress"]),
            **{p.stem: sha256(p) for p in domain_files},
        },
        "reproducibility_note": (
            "All archived run manifests use repository-relative, resolvable paths. "
            "Their originally recorded runtime paths are retained under path_relocation, "
            "and all dataset SHA-256 values remain unchanged."
        ),
    }
    write_json(ROOT / "data/manifests/evaluation_layout_manifest_v0.1.json", layout_manifest)
    print("[OK] Built single-annotator preview evaluation files")
    print(f"     DEV960 label revisions: {dev960['label_changes_from_original_gold']}")
    print(f"     DEV600 label revisions: {dev600.get('label_changes_from_original_gold', 0)}")


if __name__ == "__main__":
    main()
