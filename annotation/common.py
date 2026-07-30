from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
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
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_by_context(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        context_id = str(row.get("context_id", "")).strip()
        if not context_id:
            raise ValueError(f"Missing context_id for example {row.get('example_id')}")
        grouped[context_id].append(row)

    result: list[dict[str, Any]] = []
    for context_id, examples in grouped.items():
        contexts = {str(x.get("context", "")) for x in examples}
        domains = {str(x.get("domain", "")) for x in examples}
        if len(contexts) != 1:
            raise ValueError(f"Context mismatch inside context_id={context_id}")
        if len(domains) != 1:
            raise ValueError(f"Domain mismatch inside context_id={context_id}")
        example_ids = [str(x.get("example_id", "")) for x in examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError(f"Duplicate example_id inside context_id={context_id}")
        result.append(
            {
                "context_id": context_id,
                "domain": next(iter(domains)),
                "context": next(iter(contexts)),
                "source_title": examples[0].get("source_title"),
                "source_url": examples[0].get("source_url"),
                "examples": sorted(examples, key=lambda x: str(x.get("example_id", ""))),
            }
        )
    return sorted(result, key=lambda x: x["context_id"])


def validate_dev600(rows: list[dict[str, Any]], *, expected_examples: int | None = None) -> dict[str, int]:
    if expected_examples is not None and len(rows) != expected_examples:
        raise ValueError(f"Expected {expected_examples} examples, found {len(rows)}")
    elif expected_examples is None and len(rows) % 4 != 0:
        raise ValueError(f"Expected example count divisible by 4, found {len(rows)}")
    example_ids = [str(r.get("example_id", "")) for r in rows]
    if any(not x for x in example_ids):
        raise ValueError("At least one example is missing example_id")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Duplicate example_id detected")
    groups = group_by_context(rows)
    expected_contexts = len(rows) // 4
    if len(groups) != expected_contexts:
        raise ValueError(f"Expected {expected_contexts} context groups, found {len(groups)}")
    bad_group_sizes = {g["context_id"]: len(g["examples"]) for g in groups if len(g["examples"]) != 4}
    if bad_group_sizes:
        preview = list(bad_group_sizes.items())[:10]
        raise ValueError(f"Expected 4 examples per context; mismatches: {preview}")
    return {"examples": len(rows), "contexts": len(groups)}


def safe_identifier(value: str, fallback: str = "annotator") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def load_keyed_jsonl(path: Path, key: str = "example_id") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_key = str(row.get(key, "")).strip()
        if not item_key:
            raise ValueError(f"Missing {key} in {path}")
        if item_key in result:
            raise ValueError(f"Duplicate {key}={item_key} in {path}")
        result[item_key] = row
    return result
