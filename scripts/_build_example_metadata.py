"""
Builds data/example_metadata.csv — a sidecar metadata file that adds
split, dataset_version, and annotation_status fields without modifying
the primary JSONL files (which would change their checksums).

Run once after any primary split update, then commit the output.
"""
import csv, json, pathlib, sys

Q1_DIR = pathlib.Path(__file__).resolve().parent.parent
PRIMARY = Q1_DIR / "data/primary"
LEGACY_EVAL = Q1_DIR / "data/evaluation/legacy_experiment_version"
OUT_CSV = Q1_DIR / "data/example_metadata.csv"

sys.stdout.reconfigure(encoding="utf-8")

FILES = [
    (PRIMARY / "train.jsonl",       "train",       "v1.0"),
    (LEGACY_EVAL / "dev_general.jsonl", "dev_general", "v1.0"),
    (LEGACY_EVAL / "dev_stress.jsonl",  "dev_stress",  "v1.0"),
]

FIELDS = ["example_id", "context_id", "split", "domain", "gold_label",
          "dataset_version", "annotation_status", "source_id"]

rows = []
for fpath, split, version in FILES:
    if not fpath.exists():
        print(f"  [SKIP] {fpath.name} not found")
        continue
    for line in open(fpath, encoding="utf-8"):
        rec = json.loads(line)
        annotation_status = rec.get("annotation_status", "")
        # dev600 has source_url; use it as source_id if available
        source_id = rec.get("source_url") or rec.get("source_title") or ""
        rows.append({
            "example_id":        rec["example_id"],
            "context_id":        rec.get("context_id", ""),
            "split":             split,
            "domain":            rec.get("domain", ""),
            "gold_label":        rec.get("gold_label", ""),
            "dataset_version":   rec.get("dataset_version", version),
            "annotation_status": annotation_status,
            "source_id":         source_id,
        })
    print(f"  {fpath.name}: {sum(1 for r in rows if r['split']==split)} rows")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nOK: {len(rows)} rows written to {OUT_CSV}")