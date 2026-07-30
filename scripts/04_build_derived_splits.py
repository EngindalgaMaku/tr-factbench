"""
scripts/04_build_derived_splits.py
Protokol: data/turkish-multidomain-hallucination-dataset/docs/turkce_hallucination_q1_deney_reproducibility_plani_v2_0.md Bölüm 4.2 & 7.1

Kanonik train.jsonl'den deterministik olarak:
  - TMED (train_1600_medical.jsonl)
  - TFIN (train_1600_finance.jsonl)
  - TLEG (train_1600_legal.jsonl)
  - TMIX (train_1600_mixed.jsonl)
veri setlerini üretir ve SHA-256 manifestini yazar.
"""
import os, sys, json, hashlib
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
TRAIN_DIR = Q1_DIR / "data/primary"
LEGACY_EVAL_DIR = Q1_DIR / "data/evaluation/legacy_experiment_version"
DERIVED_TRAIN_DIR = Q1_DIR / "data/derived/train_domain"
DERIVED_DEV_DIR   = Q1_DIR / "data/derived/dev_domain"
DERIVED_MIXED_DIR = Q1_DIR / "data/derived/mixed"
MANIFESTS_DIR     = Q1_DIR / "data/manifests"

for d in [DERIVED_TRAIN_DIR, DERIVED_DEV_DIR, DERIVED_MIXED_DIR, MANIFESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8")

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("04_BUILD_DERIVED_SPLITS: TÜREV VERİ KÜMELERİ OLUŞTURULUYOR")
print("=" * 80)

raw_train_file = TRAIN_DIR / "train.jsonl"
raw_dev960_file = LEGACY_EVAL_DIR / "dev_general.jsonl"
raw_dev600_file = LEGACY_EVAL_DIR / "dev_stress.jsonl"

train_records = [json.loads(line) for line in open(raw_train_file, encoding="utf-8")]
dev960_records = [json.loads(line) for line in open(raw_dev960_file, encoding="utf-8")]
dev600_records = [json.loads(line) for line in open(raw_dev600_file, encoding="utf-8")]

# 1. Domain Train Splits (1,600 records each)
domains = ["medical", "finance", "legal"]
derived_manifest = {}

for dom in domains:
    dom_records = [r for r in train_records if r.get("domain") == dom]
    out_file = DERIVED_TRAIN_DIR / f"train_1600_{dom}.jsonl"
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        for r in dom_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\r\n")
            
    sha = get_file_sha256(out_file)
    derived_manifest[f"T{dom[:3].upper()}"] = {
        "file_name": out_file.name,
        "rel_path": str(out_file.relative_to(PROJECT_ROOT)),
        "num_records": len(dom_records),
        "sha256": sha,
        "label_distribution": dict(Counter(r.get("gold_label") for r in dom_records))
    }
    print(f"[T{dom[:3].upper():<4}] n={len(dom_records)} SHA256: {sha[:16]}...")

# 2. TMIX Split (1,600 balanced mixed records: 536 medical, 532 finance, 532 legal)
med_records = [r for r in train_records if r.get("domain") == "medical"][:536]
fin_records = [r for r in train_records if r.get("domain") == "finance"][:532]
leg_records = [r for r in train_records if r.get("domain") == "legal"][:532]
mix_records = med_records + fin_records + leg_records

mix_file = DERIVED_MIXED_DIR / "train_1600_mixed.jsonl"
with open(mix_file, "w", encoding="utf-8", newline="") as f:
    for r in mix_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\r\n")

sha_mix = get_file_sha256(mix_file)
derived_manifest["TMIX"] = {
    "file_name": mix_file.name,
    "rel_path": str(mix_file.relative_to(PROJECT_ROOT)),
    "num_records": len(mix_records),
    "sha256": sha_mix,
    "label_distribution": dict(Counter(r.get("gold_label") for r in mix_records))
}
print(f"[TMIX] n={len(mix_records)} SHA256: {sha_mix[:16]}...")

# 3. Domain Dev Splits
for dom in domains:
    d960_dom = [r for r in dev960_records if r.get("domain") == dom]
    d960_out = DERIVED_DEV_DIR / f"dev960_general_{dom}.jsonl"
    with open(d960_out, "w", encoding="utf-8", newline="") as f:
        for r in d960_dom:
            f.write(json.dumps(r, ensure_ascii=False) + "\r\n")
            
    d600_dom = [r for r in dev600_records if r.get("domain") == dom]
    d600_out = DERIVED_DEV_DIR / f"dev600_stress_{dom}.jsonl"
    with open(d600_out, "w", encoding="utf-8", newline="") as f:
        for r in d600_dom:
            f.write(json.dumps(r, ensure_ascii=False) + "\r\n")

manifest_file = MANIFESTS_DIR / "derived_datasets_manifest.json"
with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(derived_manifest, f, indent=2, ensure_ascii=False)

print(f"\nTürev Veri Kümeleri Hazırlandı ve Manifest Kaydedildi: {manifest_file}")
