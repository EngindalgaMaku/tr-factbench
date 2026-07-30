import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

bom = b'\xef\xbb\xbf'
total = 0
fixed = 0

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "completed"
print(f"Scanning: {RUNS_DIR}")

for run_path in sorted(RUNS_DIR.iterdir()):
    if not run_path.is_dir():
        continue
    mf = run_path / "metrics.json"
    if not mf.exists():
        continue
    total += 1
    b = mf.read_bytes()
    if b.startswith(bom):
        mf.write_bytes(b[3:])
        fixed += 1
        print(f"  BOM fixed: {run_path.name}")
    else:
        try:
            json.loads(b.decode("utf-8"))
        except Exception as e:
            print(f"  ERR {run_path.name}: {e}")
            print(f"  first bytes: {b[:20]}")

print(f"Scanned: {total}  Fixed: {fixed}")