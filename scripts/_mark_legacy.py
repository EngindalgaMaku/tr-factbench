"""
_mark_legacy.py — B1/B2 koşularını legacy olarak işaretler.
Her B1_*/B2_* run dizinine LEGACY.md ekler.
Ayrıca runs/legacy/LEGACY_MANIFEST.md oluşturur.
"""
import json, pathlib

RUNS_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs/completed"

LEGACY_PREFIXES = ("B0_", "B1_", "B2_")
LEGACY_NOTE = """\
# LEGACY RUN

Bu koşu eski `verifier/train.py` scriptiyle (silinmiş) üretilmiştir.
Mevcut `src/train.py` ile 1:1 tekrarlanamaz.

Kanonik Tablo A değerleri artık F1\\_* serisi (3-seed ortalama) kullanır.
Bu klasör yalnızca tarihsel referans amaçlı tutulmaktadır.
"""

legacy_runs = []
for run_dir in sorted(RUNS_DIR.iterdir()):
    if any(run_dir.name.startswith(p) for p in LEGACY_PREFIXES):
        marker = run_dir / "LEGACY.md"
        marker.write_text(LEGACY_NOTE, encoding="utf-8")
        legacy_runs.append(run_dir.name)
        print(f"  [LEGACY] {run_dir.name}")

# Manifest
manifest_path = RUNS_DIR.parent / "LEGACY_MANIFEST.md"
lines = ["# Legacy Run Manifest\n",
         "Bu koşular eski verifier/train.py scriptiyle üretildi.\n",
         "Kanonik sonuçlar için F1_* serisini kullanın.\n\n",
         "| Run ID | Prefix |\n", "|---|---|\n"]
for name in legacy_runs:
    pfx = name.split("_")[0]
    lines.append(f"| `{name}` | {pfx} |\n")
manifest_path.write_text("".join(lines), encoding="utf-8")

print(f"\n[OK] {len(legacy_runs)} koşu legacy olarak işaretlendi.")
print(f"[OK] {manifest_path}")