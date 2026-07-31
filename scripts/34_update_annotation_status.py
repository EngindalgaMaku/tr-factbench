"""Rebuild annotation_status.json from annotation artifacts present in the repo."""
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

status = {
    "last_updated": latest_review_date(review_files),
    "note": "Progress is derived from repository artifacts. The first blind-human review pass is complete for both evaluation sets. Agreement statistics and adjudication will be added after paired annotation is complete.",
    "dev_stress": {
        "total": 600,
        "candidate_label_review": progress(count_jsonl(dev600_dir / "dev_stress_candidate_label_review.jsonl"), 600),
        "annotator_1": {"role": "blind_human", **progress(count_jsonl(dev600_dir / "dev_stress_blind_annotator_1.jsonl"), 600)},
        "annotator_2": {"role": "blind_human", **progress(count_jsonl(dev600_dir / "dev_stress_blind_annotator_2.jsonl"), 600)},
        "iaa_adjudication": "not_started",
    },
    "dev_general": {
        "total": 960,
        "candidate_label_review": progress(count_jsonl(dev960_dir / "dev_general_candidate_label_review.jsonl"), 960),
        "annotator_1": {"role": "blind_human", **progress(count_jsonl(dev960_dir / "dev_general_blind_annotator_1.jsonl"), 960)},
        "annotator_2": {"role": "blind_human", **progress(count_jsonl(dev960_dir / "dev_general_blind_annotator_2.jsonl"), 960)},
        "iaa_adjudication": "not_started",
    },
    "train": {
        "total": 4800,
        "llm_preannotation": "completed" if count_jsonl(train_dir / "llm_preannotations_normalized.jsonl") == 4800 else "incomplete",
        "candidate_label_review": progress(count_jsonl(train_dir / "train_candidate_label_review.jsonl"), 4800),
    },
    "train_audit_600": {
        "sample_file_present": (train_audit_dir / "train_audit_600_blind_input.jsonl").exists(),
        "annotator_1": {"role": "blind_human", **progress(count_jsonl(train_audit_dir / "train_audit_600_blind_annotator_1.jsonl"), 600)},
        "annotator_2": {"role": "blind_human", **progress(count_jsonl(train_audit_dir / "train_audit_600_blind_annotator_2.jsonl"), 600)},
        "iaa_adjudication": "not_started",
    },
}
OUT.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] {OUT}")
