import json, sys, collections
sys.stdout.reconfigure(encoding="utf-8")

dev600 = {json.loads(l)["example_id"]: json.loads(l) for l in open("data/evaluation/legacy_experiment_version/dev_stress.jsonl", encoding="utf-8")}
llm = {json.loads(l)["example_id"]: json.loads(l) for l in open("results/annotation/dev600/llm_preannotations.jsonl", encoding="utf-8")}

n = len(llm)
match = 0
diff = []
confusion = collections.defaultdict(int)

for eid, ann in llm.items():
    gold = dev600.get(eid, {}).get("gold_label", "?")
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
    print(f"  {g:<25} -> {l:<25} : {cnt}")
print()
print("Ilk 15 celisen:")
for eid, g, l in diff[:15]:
    print(f"  {eid}  gold={g}  llm={l}")