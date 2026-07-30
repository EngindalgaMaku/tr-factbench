import hashlib, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = [
    "data/primary/train.jsonl",
    "data/evaluation/legacy_experiment_version/dev_general.jsonl",
    "data/evaluation/legacy_experiment_version/dev_stress.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev960_general_finance.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev960_general_legal.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev960_general_medical.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev600_stress_finance.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev600_stress_legal.jsonl",
    "data/evaluation/legacy_experiment_version/domain/dev600_stress_medical.jsonl",
    "data/evaluation/preview_v0.1/dev_general.jsonl",
    "data/evaluation/preview_v0.1/dev_stress.jsonl",
    "data/manifests/preview_evaluation_manifest_v0.1.json",
    "data/manifests/evaluation_layout_manifest_v0.1.json",
    "data/sources_manifest.csv",
    "data/example_metadata.csv",
    "data/schema.json",
]
lines = []
for f in files:
    p = ROOT / f
    if p.exists():
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {f}")
        print(f"OK: {f}")
    else:
        print(f"MISSING: {f}")
(ROOT / "data/checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Updated {len(lines)} hashes -> data/checksums.sha256")