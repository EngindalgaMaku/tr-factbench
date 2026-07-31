"""Build the v0.1 human-reviewed preview evaluation files deterministically.

The archived experiment inputs remain byte-identical under
``data/evaluation/legacy_experiment_version`` so archived encoder and LLM results
can be reproduced. New evaluations use the files under
``data/evaluation/preview_v0.1``.

Both preview sets are rebuilt from the archived inputs and annotator-1 records.
DEV600 additionally applies one documented claim correction and deterministic
public identifier normalization. A legacy-to-preview ID map is emitted for
cross-version comparison.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "TR-FactBench v0.1.0-preview"
LABELS = {"supported", "partially_supported", "contradicted", "unverifiable"}
STATUS = "single_annotator_provisional"
DOMAINS = ("medical", "finance", "legal")

DEV600_PUBLIC_FIELDS = (
    "context_id",
    "domain",
    "context",
    "question",
    "claim",
    "gold_label",
    "origin",
    "source_title",
    "source_url",
    "source_date",
    "retrieved_at",
    "source_status",
    "topic",
    "atom_count",
    "replaces_context_id",
    "question_style",
)

DEV600_CLAIM_FIXES = {
    "devv2_ma_b2_0031": {
        "original_claim": (
            "Emirler geliş zamanına ve fiyatına bakılmadan eşleşir. "
            "Kuponlu araçlarda işlemiş faiz fiyat hesabına katılmaz."
        ),
        "corrected_claim": (
            "Emirler fiyat ve zaman önceliği gözetilmeden eşleşir. "
            "Kuponlu araçlarda kirli fiyat da işlemiş faizi içermez."
        ),
    }
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """SHA-256 of a file, normalised to LF so the hash is identical on all platforms."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        content = handle.read()
    digest.update(content.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def validate_annotations(
    rows: list[dict], expected: int, annotator: str
) -> tuple[dict[str, str], str]:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} annotations, found {len(rows)}")
    mapping: dict[str, str] = {}
    timestamps: list[str] = []
    for row in rows:
        if row.get("review_status") != "final":
            raise ValueError(f"Non-final annotation: {row.get('example_id')}")
        if row.get("annotator_id") != annotator:
            raise ValueError(f"Unexpected annotator: {row.get('annotator_id')}")
        label = row.get("final_label")
        if label not in LABELS:
            raise ValueError(f"Unknown annotation label: {label}")
        example_id = row.get("example_id")
        if not example_id or example_id in mapping:
            raise ValueError(f"Missing or duplicate annotation ID: {example_id}")
        mapping[example_id] = label
        if row.get("reviewed_at_utc"):
            timestamps.append(row["reviewed_at_utc"])
    completed_at = max(timestamps) if timestamps else "2026-07-30T00:00:00+00:00"
    return mapping, completed_at


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["gold_label"] for row in rows)
    return {label: counts[label] for label in sorted(LABELS)}


def write_domain_views(split_name: str, rows: list[dict]) -> dict[str, str]:
    output_dir = ROOT / "data/evaluation/preview_v0.1/domain"
    hashes: dict[str, str] = {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row.get("domain") == domain]
        path = output_dir / f"{split_name}_{domain}.jsonl"
        write_jsonl(path, domain_rows)
        hashes[path.stem] = sha256(path)
    return hashes


def build_dev960() -> tuple[dict, list[dict]]:
    source = ROOT / "data/evaluation/legacy_experiment_version/dev_general.jsonl"
    annotation = ROOT / "results/annotation/dev960/dev_general_blind_annotator_1.jsonl"
    data_output = ROOT / "data/evaluation/preview_v0.1/dev_general.jsonl"
    audit_output = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1.jsonl"
    manifest_output = ROOT / "results/annotation/dev960/dev960_provisional_single_annotator_v0.1_manifest.json"

    source_rows = load_jsonl(source)
    annotation_rows = load_jsonl(annotation)
    labels, completed_at = validate_annotations(annotation_rows, 960, "annotator_1")
    if {row["example_id"] for row in source_rows} != set(labels):
        raise ValueError("DEV960 source and annotation IDs do not match")

    preview_rows: list[dict] = []
    changes: list[dict] = []
    for row in source_rows:
        reviewed = dict(row)
        reviewed["gold_label"] = labels[row["example_id"]]
        reviewed["dataset_version"] = VERSION
        reviewed["annotation_status"] = STATUS
        reviewed["gold_source"] = "single_annotator_blind_human"
        reviewed["gold_annotator"] = "annotator_1"
        preview_rows.append(reviewed)
        if row["gold_label"] != reviewed["gold_label"]:
            changes.append(
                {
                    "example_id": row["example_id"],
                    "from": row["gold_label"],
                    "to": reviewed["gold_label"],
                }
            )

    write_jsonl(data_output, preview_rows)
    write_jsonl(audit_output, preview_rows)
    manifest = {
        "created_at_utc": completed_at,
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
    return manifest, preview_rows


def build_dev600() -> tuple[dict, list[dict]]:
    source = ROOT / "data/evaluation/legacy_experiment_version/dev_stress.jsonl"
    annotation = ROOT / "results/annotation/dev600/dev_stress_blind_annotator_1.jsonl"
    audit_output = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1.jsonl"
    manifest_output = ROOT / "results/annotation/dev600/dev600_provisional_single_annotator_v0.1_manifest.json"
    data_output = ROOT / "data/evaluation/preview_v0.1/dev_stress.jsonl"
    id_map_output = ROOT / "data/manifests/dev600_legacy_to_preview_id_map.csv"

    source_rows = load_jsonl(source)
    annotation_rows = load_jsonl(annotation)
    labels, completed_at = validate_annotations(annotation_rows, 600, "annotator_1")
    if {row["example_id"] for row in source_rows} != set(labels):
        raise ValueError("Dev600 source and annotation IDs do not match")

    counters: defaultdict[str, int] = defaultdict(int)
    preview_rows: list[dict] = []
    id_map_rows: list[dict] = []
    changes: list[dict] = []
    claim_fixes: list[dict] = []

    for source_row in source_rows:
        legacy_id = source_row["example_id"]
        domain = source_row["domain"]
        counters[domain] += 1
        preview_id = f"{domain}_stress_{counters[domain]:04d}"

        public = {field: source_row.get(field) for field in DEV600_PUBLIC_FIELDS}
        claim = public["claim"]
        if legacy_id in DEV600_CLAIM_FIXES:
            fix = DEV600_CLAIM_FIXES[legacy_id]
            if claim != fix["original_claim"]:
                raise ValueError(f"Dev600 claim patch source mismatch for {legacy_id}")
            claim = fix["corrected_claim"]
            claim_fixes.append(
                {
                    "legacy_example_id": legacy_id,
                    "preview_example_id": preview_id,
                    "original_claim": fix["original_claim"],
                    "corrected_claim": claim,
                }
            )

        reviewed: dict = {
            "example_id": preview_id,
            "context_id": public["context_id"],
            "domain": domain,
            "context": public["context"],
            "question": public["question"],
            "claim": claim,
            "gold_label": labels[legacy_id],
            "dataset_version": VERSION,
        }
        for field in DEV600_PUBLIC_FIELDS[6:]:
            reviewed[field] = public[field]
        reviewed["annotation_status"] = STATUS
        # Keep annotation_status next to origin in the public record order.
        if "origin" in reviewed:
            ordered = {
                "example_id": reviewed.pop("example_id"),
                "context_id": reviewed.pop("context_id"),
                "domain": reviewed.pop("domain"),
                "context": reviewed.pop("context"),
                "question": reviewed.pop("question"),
                "claim": reviewed.pop("claim"),
                "gold_label": reviewed.pop("gold_label"),
                "dataset_version": reviewed.pop("dataset_version"),
                "origin": reviewed.pop("origin"),
                "annotation_status": reviewed.pop("annotation_status"),
            }
            ordered.update(reviewed)
            reviewed = ordered
        reviewed["gold_source"] = "single_annotator_blind_human"
        reviewed["gold_annotator"] = "annotator_1"
        preview_rows.append(reviewed)

        if source_row["gold_label"] != reviewed["gold_label"]:
            changes.append(
                {
                    "legacy_example_id": legacy_id,
                    "preview_example_id": preview_id,
                    "from": source_row["gold_label"],
                    "to": reviewed["gold_label"],
                }
            )
        id_map_rows.append(
            {
                "legacy_example_id": legacy_id,
                "preview_example_id": preview_id,
                "context_id": source_row["context_id"],
                "domain": domain,
            }
        )

    write_jsonl(audit_output, preview_rows)
    write_jsonl(data_output, preview_rows)
    write_csv(
        id_map_output,
        ["legacy_example_id", "preview_example_id", "context_id", "domain"],
        id_map_rows,
    )

    manifest = {
        "created_at_utc": completed_at,
        "dataset_version": VERSION,
        "output_file": audit_output.relative_to(ROOT).as_posix(),
        "data_release_file": data_output.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(audit_output),
        "source_dataset": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "annotation_file": annotation.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation),
        "id_map_file": id_map_output.relative_to(ROOT).as_posix(),
        "id_map_sha256": sha256(id_map_output),
        "total_examples": len(preview_rows),
        "gold_source": "single_annotator_blind_human",
        "gold_annotator": "annotator_1",
        "label_changes_from_original_gold": len(changes),
        "label_changes": changes,
        "claim_fixes": claim_fixes,
        "label_distribution": label_counts(preview_rows),
        "provisional": True,
        "status": "provisional_single_annotator_v0.1",
        "evaluation_note": (
            "Archived encoder and LLM scores use the pre-review Dev600 file. "
            f"The reviewed preview changes {len(changes)}/600 labels and applies "
            f"{len(claim_fixes)} documented claim correction; archived runs were not repeated."
        ),
    }
    write_json(manifest_output, manifest)
    return manifest, preview_rows


def main() -> None:
    dev960, dev960_rows = build_dev960()
    dev600, dev600_rows = build_dev600()
    completed_at = max(dev960["created_at_utc"], dev600["created_at_utc"])

    preview_domain_hashes = {}
    preview_domain_hashes.update(write_domain_views("dev960_general", dev960_rows))
    preview_domain_hashes.update(write_domain_views("dev600_stress", dev600_rows))

    combined = {
        "created_at_utc": completed_at,
        "dataset_version": VERSION,
        "release_status": "research_preview_single_annotator_reviewed",
        "recommended_evaluation_files": {
            "dev_general": "data/evaluation/preview_v0.1/dev_general.jsonl",
            "dev_stress": "data/evaluation/preview_v0.1/dev_stress.jsonl",
        },
        "preview_domain_evaluation_directory": "data/evaluation/preview_v0.1/domain",
        "archived_experiment_files": {
            "dev_general": "data/evaluation/legacy_experiment_version/dev_general.jsonl",
            "dev_stress": "data/evaluation/legacy_experiment_version/dev_stress.jsonl",
        },
        "dev600_id_map": "data/manifests/dev600_legacy_to_preview_id_map.csv",
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
                "label_changes": dev600["label_changes_from_original_gold"],
                "claim_fixes": len(dev600["claim_fixes"]),
                "label_distribution": dev600["label_distribution"],
            },
        },
        "usage_note": (
            "Use the human-reviewed preview files for new evaluations. Archived benchmark "
            "scores remain tied to the original experiment files retained for reproducibility. "
            "A second annotation pass and adjudication are planned for a future release."
        ),
    }
    write_json(ROOT / "data/manifests/preview_evaluation_manifest_v0.1.json", combined)

    legacy_root = ROOT / "data/evaluation/legacy_experiment_version"
    legacy_domain_files = sorted((legacy_root / "domain").glob("*.jsonl"))
    preview_domain_files = sorted((ROOT / "data/evaluation/preview_v0.1/domain").glob("*.jsonl"))
    layout_manifest = {
        "layout_version": "v0.1.0-preview-evaluation-layout",
        "training_file": "data/primary/train.jsonl",
        "recommended_preview_evaluation": combined["recommended_evaluation_files"],
        "preview_domain_evaluation_files": [p.relative_to(ROOT).as_posix() for p in preview_domain_files],
        "archived_experiment_evaluation": combined["archived_experiment_files"],
        "archived_domain_evaluation_files": [p.relative_to(ROOT).as_posix() for p in legacy_domain_files],
        "sha256": {
            "preview_dev_general": sha256(ROOT / combined["recommended_evaluation_files"]["dev_general"]),
            "preview_dev_stress": sha256(ROOT / combined["recommended_evaluation_files"]["dev_stress"]),
            "legacy_dev_general": sha256(ROOT / combined["archived_experiment_files"]["dev_general"]),
            "legacy_dev_stress": sha256(ROOT / combined["archived_experiment_files"]["dev_stress"]),
            **{p.stem: sha256(p) for p in legacy_domain_files},
            **{f"preview_{p.stem}": sha256(p) for p in preview_domain_files},
        },
        "reproducibility_note": (
            "Archived run manifests use repository-relative, resolvable paths. Original runtime "
            "paths are retained under path_relocation, and archived dataset SHA-256 values are unchanged."
        ),
    }
    write_json(ROOT / "data/manifests/evaluation_layout_manifest_v0.1.json", layout_manifest)

    print("[OK] Built human-reviewed preview evaluation files")
    print(f"     DEV960 label revisions: {dev960['label_changes_from_original_gold']}")
    print(f"     Dev600 label revisions: {dev600['label_changes_from_original_gold']}")
    print(f"     Dev600 claim corrections: {len(dev600['claim_fixes'])}")
    print(f"     Preview domain files: {len(preview_domain_hashes)}")


if __name__ == "__main__":
    main()
