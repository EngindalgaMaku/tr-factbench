"""Shared deterministic normalization for LLM pre-annotation artifacts."""
from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LABELS = {"supported", "partially_supported", "contradicted", "unverifiable"}


def derive_label(atoms: list[dict]) -> str:
    relations = {str(atom.get("relation", "")) for atom in atoms}
    has_entailed = "entailed" in relations
    has_contradicted = "contradicted" in relations
    has_not_in_context = "not_in_context" in relations
    if has_entailed and not has_contradicted and not has_not_in_context:
        return "supported"
    if has_entailed and (has_contradicted or has_not_in_context):
        return "partially_supported"
    if not has_entailed and has_contradicted:
        return "contradicted"
    return "unverifiable"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def process(*, root: Path, annotation_dir: Path, dataset_file: Path) -> dict:
    raw = annotation_dir / "llm_preannotations_raw.jsonl"
    legacy = annotation_dir / "llm_preannotations.jsonl"
    normalized = annotation_dir / "llm_preannotations_normalized.jsonl"
    manifest = annotation_dir / "llm_postprocess_manifest.json"

    if raw.exists():
        source = raw
        raw_available = True
        source_kind = "raw"
    elif legacy.exists():
        source = legacy
        raw_available = False
        source_kind = "legacy_alias_already_normalized_or_raw_unknown"
    else:
        raise FileNotFoundError(f"No LLM preannotation input in {annotation_dir}")

    rows = read_jsonl(source)
    output: list[dict] = []
    changes: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in rows:
        atoms = row.get("atomic_propositions") or []
        derived = derive_label(atoms)
        raw_label = str(row.get("llm_label_raw") or row.get("llm_label") or "")
        if raw_label not in LABELS:
            raise ValueError(f"Invalid raw label for {row.get('example_id')}: {raw_label!r}")
        if derived != raw_label:
            changes[(raw_label, derived)] += 1
        normalized_row = dict(row)
        normalized_row["llm_label_raw"] = raw_label
        normalized_row["llm_label"] = derived
        normalized_row["label_corrected_by_rule"] = derived != raw_label
        normalized_row["normalization_rule_version"] = "atomic_relation_rule_v1"
        output.append(normalized_row)

    output.sort(key=lambda row: str(row.get("example_id", "")))
    write_jsonl(normalized, output)

    # Keep the legacy path as a byte-identical compatibility alias for older scripts.
    write_jsonl(legacy, output)

    candidate_by_id = {
        row["example_id"]: row.get("gold_label")
        for row in read_jsonl(dataset_file)
    }
    candidate_matches = sum(candidate_by_id.get(row.get("example_id")) == row.get("llm_label") for row in output)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "rule_version": "atomic_relation_rule_v1",
        "source_file": source.resolve().relative_to(root.resolve()).as_posix(),
        "source_kind": source_kind,
        "raw_artifact_available": raw_available,
        "source_sha256": sha256(source),
        "normalized_file": normalized.resolve().relative_to(root.resolve()).as_posix(),
        "normalized_sha256": sha256(normalized),
        "legacy_compatibility_alias": legacy.resolve().relative_to(root.resolve()).as_posix(),
        "records": len(output),
        "labels_changed_by_rule": sum(changes.values()),
        "change_pairs": [
            {"from": old, "to": new, "count": count}
            for (old, new), count in sorted(changes.items())
        ],
        "candidate_label_matches_after_normalization": candidate_matches,
        "candidate_label_match_rate": candidate_matches / len(output) if output else None,
        "note": (
            "The original raw artifact was unavailable in this snapshot; the legacy file was used as the source. "
            "Its embedded llm_label_raw fields were preserved when available."
            if not raw_available else
            "The raw artifact is preserved separately and the normalized artifact was generated without overwriting it."
        ),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
