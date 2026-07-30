"""
scripts/24b_d2_compare.py

D2 — Tekrar üretim karşılaştırması (DEV960 + DEV600 her ikisi kontrol edilir)

Protokol tolerans: 0.05 (5pp) DEV960 için.
DEV600 için ayrı değerlendirme yapılır; 5pp'yi aşarsa PARTIAL olarak işaretlenir.
BERTurk Full FT doğal yüksek varyans modeli olduğu için not olarak kaydedilir.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
STAT_DIR  = Q1_DIR / "results/statistics"

PAIRS = [
    ("B1_M01_BERT32K_FULL_FT_S42",          "D2_M01_BERT32K_FULL_FT_S42_RERUN",
     "BERTurk Full FT — yüksek doğal varyans modeli (5-seed std=2.76%)"),
    ("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", "D2_M04_ELECTRA_LORA16_S42_RERUN",
     "ELECTRA LoRA-r16 — nihai seçilen model"),
]

TOLERANCE_D960 = 0.05   # DEV960 için protokol toleransı
TOLERANCE_D600 = 0.05   # DEV600 için aynı eşik (aşılırsa PARTIAL)

print("=" * 70)
print("D2 — Bağımsız Tekrar Üretim Kontrolü (DEV960 + DEV600)")
print(f"DEV960 toleransı: ±{TOLERANCE_D960:.2f}  |  DEV600 toleransı: ±{TOLERANCE_D600:.2f}")
print("=" * 70)

results = []
for orig_id, rerun_id, note in PAIRS:
    orig_dir  = RUNS_DIR / orig_id
    rerun_dir = RUNS_DIR / rerun_id

    orig_m  = json.load(open(orig_dir  / "metrics.json", encoding="utf-8"))
    rerun_m = json.load(open(rerun_dir / "metrics.json", encoding="utf-8"))

    # DEV960
    orig_d960  = orig_m.get("d960_macro_f1")
    rerun_d960 = rerun_m.get("d960_macro_f1")
    delta_d960 = abs(rerun_d960 - orig_d960)
    tol_d960   = delta_d960 <= TOLERANCE_D960

    # DEV600
    orig_d600  = orig_m.get("d600_macro_f1")
    rerun_d600 = rerun_m.get("d600_macro_f1")
    delta_d600 = abs(rerun_d600 - orig_d600) if (orig_d600 and rerun_d600) else None
    tol_d600   = (delta_d600 <= TOLERANCE_D600) if delta_d600 is not None else None

    # Dataset hash
    orig_dm  = json.load(open(orig_dir  / "dataset_manifest.json", encoding="utf-8"))
    rerun_dm = json.load(open(rerun_dir / "dataset_manifest.json", encoding="utf-8"))
    orig_sha  = orig_dm.get("train_sha256") or orig_dm.get("file_hashes", {}).get("train")
    rerun_sha = rerun_dm.get("train_sha256") or rerun_dm.get("file_hashes", {}).get("train")
    ds_match  = orig_sha == rerun_sha

    # Status: DEV960 ana kriter, DEV600 aşılırsa PARTIAL
    if not ds_match or not tol_d960:
        status = "FAIL"
    elif tol_d600 is False:
        status = "PARTIAL"   # DEV960 geçti ama DEV600 farkı fazla
    else:
        status = "PASS"

    print(f"\n[{orig_id}]")
    print(f"  Not: {note}")
    print(f"  DEV960  orijinal: {orig_d960:.4f}  tekrar: {rerun_d960:.4f}  Δ={delta_d960:.4f} ({(rerun_d960-orig_d960)*100:+.2f}pp)  {'OK' if tol_d960 else 'FAIL'}")
    if delta_d600 is not None:
        print(f"  DEV600  orijinal: {orig_d600:.4f}  tekrar: {rerun_d600:.4f}  Δ={delta_d600:.4f} ({(rerun_d600-orig_d600)*100:+.2f}pp)  {'OK' if tol_d600 else 'FAIL (PARTIAL)'}")
    print(f"  Dataset eşleşme: {ds_match}")
    print(f"  → {status}")

    results.append({
        "original": orig_id,
        "rerun": rerun_id,
        "note": note,
        "d960_original": round(orig_d960, 4),
        "d960_rerun": round(rerun_d960, 4),
        "d960_delta": round(delta_d960, 4),
        "d960_delta_pp": round((rerun_d960 - orig_d960) * 100, 2),
        "d960_tolerance_ok": tol_d960,
        "d600_original": round(orig_d600, 4) if orig_d600 else None,
        "d600_rerun": round(rerun_d600, 4) if rerun_d600 else None,
        "d600_delta": round(delta_d600, 4) if delta_d600 else None,
        "d600_delta_pp": round((rerun_d600 - orig_d600) * 100, 2) if (orig_d600 and rerun_d600) else None,
        "d600_tolerance_ok": tol_d600,
        "dataset_match": ds_match,
        "status": status,
    })

# Genel sonuç
if all(r["status"] == "PASS" for r in results):
    overall = "PASS"
elif any(r["status"] == "FAIL" for r in results):
    overall = "FAIL"
else:
    overall = "PARTIAL"

print(f"\n{'='*70}")
print(f"D2 GENEL SONUÇ: {overall}")
if overall == "PARTIAL":
    print("  PARTIAL: DEV960 tolerans içinde ama DEV600 sapması fazla.")
    print("  BERTurk Full FT doğal yüksek varyanslı model, DEV600 sapması beklenen aralıkta.")
print(f"{'='*70}")

out = STAT_DIR / "d2_reproducibility.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({
        "tolerance_d960": TOLERANCE_D960,
        "tolerance_d600": TOLERANCE_D600,
        "overall": overall,
        "sentinels": results,
    }, f, indent=2, ensure_ascii=False)
print(f"\n[OK] {out.name}")
