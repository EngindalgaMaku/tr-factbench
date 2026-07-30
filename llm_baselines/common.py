from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LABELS = [
    "supported",
    "partially_supported",
    "contradicted",
    "unverifiable",
]
RELATIONS = ["entailed", "contradicted", "not_in_context"]
AMBIGUITY_FLAGS = [
    "multi_interpretation",
    "weak_contradiction",
    "scope_mismatch",
    "temporal_ambiguity",
    "numerical_ambiguity",
    "terminology_ambiguity",
    "none",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_keyed_jsonl(path: Path, key: str = "example_id") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_key = str(row.get(key, "")).strip()
        if not item_key:
            raise ValueError(f"Missing {key} in {path}")
        if item_key in result:
            raise ValueError(f"Duplicate {key}={item_key!r} in {path}")
        result[item_key] = row
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_slug(value: str, fallback: str = "model") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def validate_dataset(rows: list[dict[str, Any]], require_gold: bool = False) -> None:
    if not rows:
        raise ValueError("Dataset is empty")
    seen: set[str] = set()
    required = {"example_id", "context", "claim"}
    if require_gold:
        required.add("gold_label")
    for index, row in enumerate(rows, start=1):
        missing = [key for key in required if key not in row or row[key] in (None, "")]
        if missing:
            raise ValueError(f"Row {index} is missing fields: {missing}")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"Duplicate example_id: {example_id}")
        seen.add(example_id)
        if require_gold and row["gold_label"] not in LABELS:
            raise ValueError(f"Invalid gold_label for {example_id}: {row['gold_label']!r}")
