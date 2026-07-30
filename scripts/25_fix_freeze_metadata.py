"""
scripts/25_fix_freeze_metadata.py

Freeze koşularındaki iki metadata hatasını düzeltir:
1. metrics.json'da trainable_params → ckpt/config.json'dan gerçek freeze-sonrası değer
2. metrics.json'da peak_vram_mb=0 → peak_memory.json'dan gerçek VRAM
3. runs.csv'de freeze_ratio → gerçek oran (freeze_layers / toplam_katman)

ELECTRA-base için toplam katman: 12
"""
import sys, json, csv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR = Q1_DIR / "runs/completed"

TOTAL_LAYERS = 12   # ELECTRA-base encoder katman sayısı
TOTAL_PARAMS = 110918408  # ELECTRA-base toplam parametre

def get_ckpt_config(run_dir):
    """ckpt altındaki ana config.json'u bul (best_model ve checkpoints hariç)"""
    for p in run_dir.glob("ckpt/**/config.json"):
        if "best_model" not in str(p) and "checkpoints" not in str(p):
            return json.load(open(p, encoding="utf-8"))
    return {}

def get_peak_memory(run_dir):
    for p in run_dir.glob("ckpt/**/peak_memory.json"):
        return json.load(open(p, encoding="utf-8")).get("peak_gpu_memory_mb")
    return None

print("=" * 60)
print("Freeze metadata düzeltme")
print("=" * 60)

fixed = []
for run_dir in sorted(RUNS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        continue
    m = json.load(open(metrics_path, encoding="utf-8"))
    if m.get("method") != "freeze":
        continue

    # Ckpt config'den gerçek değerleri al
    ckpt_cfg = get_ckpt_config(run_dir)
    peak_mb  = get_peak_memory(run_dir)
    freeze_layers = m.get("freeze_layers", 0)
    trainable_pct = ckpt_cfg.get("trainable_pct")

    old_trainable = m.get("trainable_params")
    old_vram      = m.get("peak_vram_mb")

    if trainable_pct is not None:
        new_trainable = round(TOTAL_PARAMS * trainable_pct / 100)
    else:
        new_trainable = old_trainable

    new_vram = peak_mb if (peak_mb and peak_mb > 0) else None
    new_freeze_ratio = round(freeze_layers / TOTAL_LAYERS, 4) if TOTAL_LAYERS else 0

    print(f"\n{run_dir.name}")
    print(f"  freeze_layers={freeze_layers}, freeze_ratio={new_freeze_ratio}")
    print(f"  trainable_params: {old_trainable:,} → {new_trainable:,} ({trainable_pct:.1f}%)" if trainable_pct else f"  trainable_params: no ckpt cfg")
    print(f"  peak_vram_mb: {old_vram} → {new_vram}")

    # metrics.json güncelle
    m["trainable_params"] = new_trainable
    m["peak_vram_mb"] = new_vram
    m["freeze_ratio"] = new_freeze_ratio
    m["trainable_pct"] = round(trainable_pct, 4) if trainable_pct else None
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    fixed.append(run_dir.name)

print(f"\n{len(fixed)} freeze koşusu düzeltildi.")
print("[TAMAMLANDI]")
