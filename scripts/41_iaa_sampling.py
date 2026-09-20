"""Historical builder for the unused train_audit_600 quality-audit sample.

That pack was prepared but never annotated and is cancelled / out of scope.
Do not treat a re-run as a live annotation task. Prefix-150 is a different freeze.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data/primary/train.jsonl"
OUT_DIR = ROOT / "results/annotation/train_audit_600"
DEPRECATED_DIR = ROOT / "results/annotation/deprecated/legacy_iaa_sample_500"
BLIND_INPUT = OUT_DIR / "train_audit_600_blind_input.jsonl"
REFERENCE = OUT_DIR / "train_audit_600_candidate_reference.jsonl"
MANIFEST = OUT_DIR / "train_audit_600_manifest.json"
README = OUT_DIR / "README.md"

DOMAINS = ["finance", "legal", "medical"]
LABELS = ["supported", "partially_supported", "contradicted", "unverifiable"]
CONTEXTS_PER_DOMAIN = 50
LENGTH_STRATA = 5
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_legacy_files() -> list[str]:
    archived: list[str] = []
    legacy_names = [
        "iaa_sample_500.jsonl",
        "iaa_sample_500_annotator2.jsonl",
        "iaa_sample_manifest.json",
    ]
    DEPRECATED_DIR.mkdir(parents=True, exist_ok=True)
    for name in legacy_names:
        source = ROOT / "results/annotation" / name
        destination = DEPRECATED_DIR / name
        if source.exists():
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
            archived.append(destination.relative_to(ROOT).as_posix())
    if archived:
        (DEPRECATED_DIR / "README.md").write_text(
            "# Deprecated legacy IAA sample\n\n"
            "These files used item-level sampling and copied the candidate label into an "
            "annotator field. They are retained only for audit history and **must not** be "
            "used as human-human IAA evidence. The later `train_audit_600` pack was also "
            "prepared but never annotated and is cancelled.\n",
            encoding="utf-8",
        )
    return archived


rows = read_jsonl(TRAIN_FILE)
by_context: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    by_context[str(row["context_id"])].append(row)

contexts_by_domain: dict[str, list[dict]] = defaultdict(list)
for context_id, examples in by_context.items():
    if len(examples) != 4:
        raise SystemExit(f"Expected four claims for {context_id}, found {len(examples)}")
    domains = {str(row["domain"]) for row in examples}
    contexts = {str(row["context"]) for row in examples}
    labels = {str(row["gold_label"]) for row in examples}
    if len(domains) != 1 or len(contexts) != 1:
        raise SystemExit(f"Inconsistent group: {context_id}")
    if labels != set(LABELS):
        raise SystemExit(f"Expected four labels for {context_id}, found {sorted(labels)}")
    domain = next(iter(domains))
    contexts_by_domain[domain].append(
        {
            "source_context_id": context_id,
            "domain": domain,
            "context": next(iter(contexts)),
            "examples": sorted(examples, key=lambda row: str(row["example_id"])),
        }
    )

rng = random.Random(SEED)
selected: list[dict] = []
stratum_counts: Counter[str] = Counter()
for domain in DOMAINS:
    pool = sorted(contexts_by_domain[domain], key=lambda group: (len(group["context"]), group["source_context_id"]))
    if len(pool) < CONTEXTS_PER_DOMAIN:
        raise SystemExit(f"Not enough contexts for {domain}: {len(pool)}")
    strata: list[list[dict]] = [[] for _ in range(LENGTH_STRATA)]
    for rank, group in enumerate(pool):
        stratum = min(LENGTH_STRATA - 1, rank * LENGTH_STRATA // len(pool))
        strata[stratum].append(group)
    per_stratum = CONTEXTS_PER_DOMAIN // LENGTH_STRATA
    domain_selected: list[dict] = []
    for stratum, candidates in enumerate(strata):
        chosen = rng.sample(candidates, per_stratum)
        domain_selected.extend(chosen)
        stratum_counts[f"{domain}:length_q{stratum + 1}"] = len(chosen)
    rng.shuffle(domain_selected)
    selected.extend(domain_selected)

# Stable cross-domain shuffle prevents a long domain block while remaining reproducible.
rng.shuffle(selected)

blind_rows: list[dict] = []
reference_rows: list[dict] = []
context_map: list[dict] = []
example_counter = 0
for context_index, group in enumerate(selected, start=1):
    opaque_context_id = f"train_audit_c{context_index:03d}"
    context_map.append(
        {
            "opaque_context_id": opaque_context_id,
            "source_context_id": group["source_context_id"],
            "domain": group["domain"],
            "context_length_chars": len(group["context"]),
        }
    )
    for source_row in group["examples"]:
        example_counter += 1
        opaque_example_id = f"train_audit_e{example_counter:04d}"
        blind_rows.append(
            {
                "example_id": opaque_example_id,
                "context_id": opaque_context_id,
                "domain": group["domain"],
                "context": source_row["context"],
                "question": source_row.get("question", ""),
                "claim": source_row["claim"],
            }
        )
        reference_rows.append(
            {
                "example_id": opaque_example_id,
                "context_id": opaque_context_id,
                "source_example_id": source_row["example_id"],
                "source_context_id": source_row["context_id"],
                "domain": source_row["domain"],
                "candidate_label": source_row["gold_label"],
            }
        )

if len(blind_rows) != 600 or len({row["context_id"] for row in blind_rows}) != 150:
    raise SystemExit("Unexpected audit sample size")

OUT_DIR.mkdir(parents=True, exist_ok=True)
write_jsonl(BLIND_INPUT, blind_rows)
write_jsonl(REFERENCE, reference_rows)
archived = archive_legacy_files()

manifest = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "sampling_version": "train_audit_context_grouped_v1",
    "source_file": TRAIN_FILE.relative_to(ROOT).as_posix(),
    "source_sha256": sha256(TRAIN_FILE),
    "seed": SEED,
    "sampling_unit": "context_group",
    "contexts_per_domain": CONTEXTS_PER_DOMAIN,
    "length_strata_per_domain": LENGTH_STRATA,
    "selected_contexts": 150,
    "selected_examples": 600,
    "domain_context_counts": dict(Counter(group["domain"] for group in selected)),
    "candidate_label_counts": dict(Counter(row["candidate_label"] for row in reference_rows)),
    "stratum_counts": dict(stratum_counts),
    "blind_input_file": BLIND_INPUT.relative_to(ROOT).as_posix(),
    "blind_input_sha256": sha256(BLIND_INPUT),
    "candidate_reference_file": REFERENCE.relative_to(ROOT).as_posix(),
    "candidate_reference_sha256": sha256(REFERENCE),
    "opaque_ids": True,
    "candidate_labels_present_in_blind_input": False,
    "independent_blind_annotators_required": 2,
    "adjudication_required_after_iaa": True,
    "legacy_files_archived": archived,
    "context_map": context_map,
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
README.write_text(
    "# Train audit 600 — cancelled / unused pack\n\n"
    "This directory contains a context-grouped quality-audit sample from the training set. "
    "It includes 150 contexts (50 per domain) and all four claims from each context, for 600 examples.\n\n"
    "**Status: cancelled / out of scope.** Files were generated but never annotated. "
    "Do not start two blinds. Prefix-150 is a different freeze.\n",
    encoding="utf-8",
)

print(f"[OK] {BLIND_INPUT.relative_to(ROOT)} — {len(blind_rows)} examples")
print(f"[OK] {REFERENCE.relative_to(ROOT)}")
print(f"[OK] {MANIFEST.relative_to(ROOT)}")
if archived:
    print(f"[INFO] Archived {len(archived)} legacy files under {DEPRECATED_DIR.relative_to(ROOT)}")
