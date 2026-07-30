"""
scripts/22_prepare_a3_lodo_data.py

A3 — Leave-One-Domain-Out (LODO) için veri dosyaları üretir.

3 kombinasyon:
  lodo_no_legal:   tıp + finans (1600+1600=3200) → hukuk görülmemiş
  lodo_no_finance: tıp + hukuk  (1600+1600=3200) → finans görülmemiş
  lodo_no_medical: finans + hukuk (1600+1600=3200) → tıp görülmemiş

Çıktılar: data/derived/leave_one_domain_out/
"""
import sys, json, random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
TRAIN_DIR = Q1_DIR / "data/derived/train_domain"
LODO_DIR  = Q1_DIR / "data/derived/leave_one_domain_out"
LODO_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\r\n")

medical  = load_jsonl(TRAIN_DIR / "train_1600_medical.jsonl")
finance  = load_jsonl(TRAIN_DIR / "train_1600_finance.jsonl")
legal    = load_jsonl(TRAIN_DIR / "train_1600_legal.jsonl")

print("Kaynak dosyalar:")
print(f"  Medical:  {len(medical)} kayıt")
print(f"  Finance:  {len(finance)} kayıt")
print(f"  Legal:    {len(legal)} kayıt")

combinations = [
    ("lodo_no_legal",   medical + finance, "tıp+finans → hukuk görülmemiş"),
    ("lodo_no_finance", medical + legal,   "tıp+hukuk → finans görülmemiş"),
    ("lodo_no_medical", finance + legal,   "finans+hukuk → tıp görülmemiş"),
]

for name, records, desc in combinations:
    # Sabit seed ile karıştır (deterministik)
    rng = random.Random(42)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    
    out_path = LODO_DIR / f"train_3200_{name}.jsonl"
    save_jsonl(shuffled, out_path)
    
    from collections import Counter
    label_counts = Counter(r.get("gold_label") for r in shuffled)
    domain_counts = Counter(r.get("domain", r.get("source_domain", "?")) for r in shuffled)
    
    print(f"\n{out_path.name} ({desc})")
    print(f"  Toplam: {len(shuffled)} kayıt")
    print(f"  Etiket dağılımı: {dict(label_counts)}")
    print(f"  Domain dağılımı: {dict(domain_counts)}")

print("\n[TAMAMLANDI] LODO veri dosyaları hazır.")