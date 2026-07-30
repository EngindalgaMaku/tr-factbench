"""
scripts/00_preflight.py
Protokol: data/turkish-multidomain-hallucination-dataset/docs/turkce_hallucination_q1_deney_reproducibility_plani_v2_0.md Bölüm 7.1 (P0.1 - P0.5)

Eğitim öncesi donanım, ortam, veri hashleri ve bağımlılıkları doğrulayarak preflight_report.json üretir.
"""
import os, sys, json, hashlib, subprocess
try:
    import torch
except ImportError:
    torch = None
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
TRAIN_DIR = Q1_DIR / "data/primary"
LEGACY_EVAL_DIR = Q1_DIR / "data/evaluation/legacy_experiment_version"
REPORT_DIR = Q1_DIR / "reports/data_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8")

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("00_PREFLIGHT: ORTAM VE VERİ ÖN DENETİMİ BAŞLATILIYOR")
print("=" * 80)

# 1. Environment Audit
cuda_available = bool(torch and torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU or torch unavailable)"
gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0.0

env_info = {
    "python_version": sys.version,
    "pytorch_version": torch.__version__ if torch else "not-installed",
    "cuda_available": cuda_available,
    "gpu_name": gpu_name,
    "gpu_total_vram_gb": gpu_vram,
    "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False
}

print(f"GPU: {gpu_name} ({gpu_vram} GB VRAM) | PyTorch: {torch.__version__ if torch else 'not-installed'}")

# 2. Data Files Hash & Record Audit
raw_files = {
    "TALL": TRAIN_DIR / "train.jsonl",
    "D960": LEGACY_EVAL_DIR / "dev_general.jsonl",
    "D600": LEGACY_EVAL_DIR / "dev_stress.jsonl",
}

data_audit = {}
all_passed = True

for key, path in raw_files.items():
    if not path.exists():
        print(f"[HATA] Veri dosyası bulunamadı: {path}")
        all_passed = False
        continue
    
    sha = get_file_sha256(path)
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    data_audit[key] = {
        "file_name": path.name,
        "sha256": sha,
        "num_records": len(records)
    }
    print(f"[{key:<4}] n={len(records):<4} SHA256: {sha}")

preflight_data = {
    "status": "PASS" if all_passed else "FAIL",
    "environment": env_info,
    "data_manifest": data_audit
}

out_report = REPORT_DIR / "preflight_report.json"
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(preflight_data, f, indent=2, ensure_ascii=False)

print(f"\nPreflight Raporu Kaydedildi: {out_report}")
print("Preflight Statüsü: PASS" if all_passed else "Preflight Statüsü: FAIL")

if not all_passed:
    raise SystemExit(1)
