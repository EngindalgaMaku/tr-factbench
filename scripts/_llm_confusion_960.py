import json, sys, collections, pathlib
sys.stdout.reconfigure(encoding="utf-8")

dev960_path = pathlib.Path("data/evaluation/legacy_experiment_version/dev_general.jsonl")
llm_path = pathlib.Path("results/annotation/dev960/llm_preannotations.jsonl")

if not llm_path.exists():
    print(f"HATA: {llm_path} bulunamadi")
    sys.exit(1)

dev960 = {json.loads(l)["example_id"]: json.loads(l) for l in dev960_path.read_text(encoding="utf-8").splitlines() if l.strip()}
llm = {json.loads(l)["example_id"]: json.loads(l) for l in llm_path.read_text(encoding="utf-8").splitlines() if l.strip()}

n = len(llm)
match = 0
diff = []
confusion = collections.defaultdict(int)

for eid, ann in llm.items():
    gold = dev960.get(eid, {}).get("gold_label", "?")
    lbl = ann["llm_label"]
    if gold == lbl:
        match += 1
    else:
        diff.append((eid, gold, lbl))
        confusion[(gold, lbl)] += 1

print(f"Toplam: {n}  Uyum: {match}  Celisen: {len(diff)}  Oran: {match/n:.1%}")
print()
print("Karisiklik matrisi (gold -> llm):")
for (g, l), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
    print(f"  {g:<25} -> {l:<25}: {cnt}")
print()
print("Ilk 15 celisen:")
for eid, g, l in diff[:15]:
    print(f"  {eid}  gold={g}  llm={l}")