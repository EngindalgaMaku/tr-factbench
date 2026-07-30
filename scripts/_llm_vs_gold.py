import json, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")

dev600 = {json.loads(l)["example_id"]: json.loads(l) for l in open("data/evaluation/legacy_experiment_version/dev_stress.jsonl", encoding="utf-8")}
llm = {json.loads(l)["example_id"]: json.loads(l) for l in open("results/annotation/dev600/llm_preannotations.jsonl", encoding="utf-8")}

print(f"{'example_id':<22} {'gold_label':<22} {'llm_label':<22} {'match'}")
print("-"*85)
match = 0
for eid, ann in llm.items():
    gold = dev600.get(eid, {}).get("gold_label", "?")
    llm_label = ann["llm_label"]
    ok = gold == llm_label
    if ok: match += 1
    print(f"{eid:<22} {gold:<22} {llm_label:<22} {'OK' if ok else 'DIFF'}")

n = len(llm)
print(f"\nAnlaşma: {match}/{n} = {match/n:.1%}")