"""Rebuild annotation_status.json from annotation artifacts present in the repo.

Do not resurrect cancelled train packs as live work. Full Gate 1 4,800,
leftover train candidate-label review, and train_audit_600 are out of scope.
Preserve the frozen train_4800_prefix150 block if present in the current file.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "annotation_status.json"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())




def latest_review_date(paths: list[Path]) -> str:
    dates: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("reviewed_at_utc")
            if value:
                dates.append(value[:10])
    return max(dates) if dates else "2026-07-30"


def progress(completed: int, total: int) -> dict:
    if completed <= 0:
        status = "not_started"
    elif completed >= total:
        status = "completed"
    else:
        status = "in_progress"
    return {"completed": completed, "total": total, "status": status}


dev600_dir = ROOT / "results/annotation/dev600"
dev960_dir = ROOT / "results/annotation/dev960"
train_dir = ROOT / "results/annotation/train"
train_audit_dir = ROOT / "results/annotation/train_audit_600"

review_files = [
    dev600_dir / "dev_stress_blind_annotator_1.jsonl",
    dev600_dir / "dev_stress_blind_annotator_2.jsonl",
    dev960_dir / "dev_general_blind_annotator_1.jsonl",
]

existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
prefix150 = existing.get("train_4800_prefix150")

status = {
    "last_updated": latest_review_date(review_files),
    "note": (
        "This preview resource is frozen as-is. Progress is derived from repository artifacts. "
        "The first blind-human review pass is complete for both evaluation sets. Gold-480 is a "
        "frozen dual-annotator evaluation set. A predeclared 600-example prefix of train_4800 "
        "(seed 42, annotator 1) is frozen as a silver-train noise estimate; training labels remain "
        "construction-time candidates. Full Gate 1 4,800, leftover train candidate-label review, "
        "and the stratified two-person train_audit_600 pack are cancelled / out of scope. A second "
        "development-set pass (Dev600 A2 leftover; DEV960 A2 not started) is a known limitation, "
        "not remaining work required to close this resource."
    ),
    "dev_stress": {
        "total": 600,
        "candidate_label_review": progress(count_jsonl(dev600_dir / "dev_stress_candidate_label_review.jsonl"), 600),
        "annotator_1": {"role": "blind_human", **progress(count_jsonl(dev600_dir / "dev_stress_blind_annotator_1.jsonl"), 600)},
        "annotator_2": {
            "role": "blind_human",
            **progress(count_jsonl(dev600_dir / "dev_stress_blind_annotator_2.jsonl"), 600),
            "status": "leftover_not_required",
        },
        "iaa_adjudication": "not_required",
    },
    "dev_general": {
        "total": 960,
        "candidate_label_review": {
            **progress(count_jsonl(dev960_dir / "dev_general_candidate_label_review.jsonl"), 960),
            "status": "leftover_not_required",
        },
        "annotator_1": {"role": "blind_human", **progress(count_jsonl(dev960_dir / "dev_general_blind_annotator_1.jsonl"), 960)},
        "annotator_2": {
            "role": "blind_human",
            **progress(count_jsonl(dev960_dir / "dev_general_blind_annotator_2.jsonl"), 960),
            "status": "leftover_not_required",
        },
        "iaa_adjudication": "not_required",
    },
    "train": {
        "total": 4800,
        "llm_preannotation": "completed" if count_jsonl(train_dir / "llm_preannotations_normalized.jsonl") == 4800 else "incomplete",
        "candidate_label_review": {
            **progress(count_jsonl(train_dir / "train_candidate_label_review.jsonl"), 4800),
            "status": "cancelled",
        },
        "labels": "construction_time_silver",
        "gold_relabel": False,
    },
    "train_audit_600": {
        "sample_file_present": (train_audit_dir / "train_audit_600_blind_input.jsonl").exists(),
        "status": "cancelled",
        "note": (
            "Stratified 50-contexts-per-domain sample was prepared but never annotated. "
            "Cancelled / out of scope. Distinct from the frozen train_4800_prefix150 silver audit. "
            "Do not start two blinds."
        ),
        "annotator_1": {"role": "blind_human", "completed": 0, "total": 600, "status": "cancelled"},
        "annotator_2": {"role": "blind_human", "completed": 0, "total": 600, "status": "cancelled"},
        "iaa_adjudication": "cancelled",
    },
}
if prefix150:
    status["train_4800_prefix150"] = prefix150
OUT.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] {OUT}")
