"""
analyze_partial_errors.py
gold_v1.0 - partially_supported olup contradicted tahmin edilenleri incele
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_jsonl

GOLD_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "gold_v1.0" / "gold" / "TR-FactBench_controlled480_GOLD_v1.0.jsonl"
PRED_PATH = Path(__file__).parent.parent / "results" / "llm_baselines" / "gold_v1.0" / "openai_gpt-4.1-mini" / "zero_shot" / "predictions.jsonl"

gold = read_jsonl(GOLD_PATH)
preds = read_jsonl(PRED_PATH)
pred_by_id = {r["example_id"]: r for r in preds}

# Hata kategorileri
ps_to_con = []   # partially -> contradicted
ps_to_other = [] # partially -> her seyden
correct_ps = []

for g in gold:
    eid = g.get("release_example_id", "")
    p = pred_by_id.get(eid)
    if not p:
        continue
    if g["gold_label"] != "partially_supported":
        continue
    pred = p.get("predicted_label", "")
    if pred == "partially_supported":
        correct_ps.append((g, p))
    elif pred == "contradicted":
        ps_to_con.append((g, p))
    else:
        ps_to_other.append((g, p))

print(f"\n=== partially_supported analizi ===")
print(f"Toplam PS ornegi   : {len(ps_to_con)+len(ps_to_other)+len(correct_ps)}")
print(f"Dogru tahmin (PS)  : {len(correct_ps)}")
print(f"  -> contradicted  : {len(ps_to_con)}  ({len(ps_to_con)/(len(ps_to_con)+len(ps_to_other)+len(correct_ps))*100:.1f}%)")
print(f"  -> diger hatalar : {len(ps_to_other)}")

# Atom analizi: PS->contradicted orneklerinde model ne goriyor
print(f"\n=== PS->contradicted orneklerinde atom iliskileri ===")
from collections import Counter
relation_counts = Counter()
for g, p in ps_to_con:
    for a in p.get("atomic_propositions", []):
        relation_counts[a["relation"]] += 1
print("Toplam atom iliskileri:", dict(relation_counts))
print()

# PS->contradicted: kural ihlali mi?
# Kurala gore: hic entailed yok + en az bir contradicted -> contradicted dogru
# Ama gold PS -> en az bir entailed olmali -> model yanlis
no_entailed = sum(
    1 for g, p in ps_to_con
    if not any(a["relation"] == "entailed" for a in p.get("atomic_propositions", []))
)
print(f"Entailed atom OLMAYAN PS->contradicted ornekleri: {no_entailed}/{len(ps_to_con)}")
print(f"(Bunlarda model 'hic destek yok' gormus, contradicted secmis - ama gold PS)")
print()

# Ilk 5 ornek detay
print("=== Ilk 5 PS->contradicted hatasi ===\n")
for i, (g, p) in enumerate(ps_to_con[:5]):
    print(f"--- Ornek {i+1} ---")
    print(f"ID     : {g['release_example_id']}")
    print(f"CLAIM  : {g['claim']}")
    print(f"CONTEXT: {g['context'][:200]}...")
    print(f"GOLD   : {g['gold_label']}")
    print(f"PRED   : {p['predicted_label']}  (confidence={p.get('confidence','-')})")
    print(f"RATIONALE: {p['brief_rationale']}")
    print(f"ATOMS:")
    for a in p.get("atomic_propositions", []):
        print(f"  [{a['relation']:>14}] {a['text'][:90]}")
        if a.get("evidence_quote"):
            print(f"                   KANIT: {a['evidence_quote'][:80]}")
    print()

# Sistemik soru: goldun PS label'inin parcial destekli kismi nerede?
print("=== Gold tarafinda PS icin hangi kisim destekleniyor? ===")
print("(Bu soruya cevap vermek icin gold annotator notlari gerekir)")
print()

# Confidence analizi
conf_wrong = [p.get("confidence", 0) for g, p in ps_to_con]
conf_right = [p.get("confidence", 0) for g, p in correct_ps]
if conf_wrong:
    print(f"Yanlis (PS->con) confidence ort: {sum(conf_wrong)/len(conf_wrong):.3f}")
if conf_right:
    print(f"Dogru  (PS->PS)  confidence ort: {sum(conf_right)/len(conf_right):.3f}")