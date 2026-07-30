"""
IAA (Inter-Annotator Agreement) hesaplama scripti.

Protokol:
  - LLM destekli mод → aday altın etiket (gold candidate), IAA'ya DAHIL EDILMEZ
  - Kor insan 1 (blind_annotator_1) + Kor insan 2 (blind_annotator_2) → GERCEK IAA

Bu script iki kör insan anotasyonunu karşılaştırır.
"""
import json, sys, collections, pathlib
sys.stdout.reconfigure(encoding="utf-8")

BASE = pathlib.Path("results/annotation/dev600")

def load_blind(annotator_id):
    path = BASE / f"dev600_blind_{annotator_id}.jsonl"
    if not path.exists():
        return None, path
    rows = {json.loads(l)["example_id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return rows, path

a1, p1 = load_blind("annotator_1")
a2, p2 = load_blind("annotator_2")

print(f"Kor annotator_1: {len(a1) if a1 else 'YOK'} ornek  ({p1})")
print(f"Kor annotator_2: {len(a2) if a2 else 'YOK'} ornek  ({p2})")

if a1 is None or a2 is None:
    print("\nIkinci kor anotator tamamlanmadan IAA hesaplanamaz.")
    print("Simdilik durum:")
    if a1:
        lbl1 = collections.Counter(r.get("final_label") for r in a1.values())
        print(f"  annotator_1 etiketleri: {dict(lbl1)}")
    sys.exit(0)

common = set(a1) & set(a2)
n = len(common)
match = 0
confusion = collections.defaultdict(int)
diff = []

for eid in common:
    l1 = a1[eid].get("final_label")
    l2 = a2[eid].get("final_label")
    if l1 == l2:
        match += 1
    else:
        diff.append((eid, l1, l2))
        confusion[(l1, l2)] += 1

print(f"\nOrtak: {n}  Uyum: {match}  Celisen: {n-match}  IAA: {match/n:.1%}")
print("\nUyusmazlik matrisi (annotator_1 -> annotator_2):")
for (a, b), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
    print(f"  {a:<25} -> {b:<25}: {cnt}")
if diff:
    print("\nCelisen ornekler:")
    for eid, l1, l2 in diff[:20]:
        print(f"  {eid}  ann1={l1}  ann2={l2}")