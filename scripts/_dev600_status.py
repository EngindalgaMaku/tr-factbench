import json, sys, collections, pathlib
sys.stdout.reconfigure(encoding="utf-8")

BASE = pathlib.Path("results/annotation/dev600")

def analyze_reviews(path, label):
    if not path.exists():
        print(f"{label}: dosya yok ({path})")
        return
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)
    actions = collections.Counter(r.get("review_action") for r in rows)
    statuses = collections.Counter(r.get("review_status") for r in rows)
    labels = collections.Counter(r.get("final_label") for r in rows)
    print(f"{label}: {total} ornek")
    print(f"  accepted   : {actions['accepted']}")
    print(f"  corrected  : {actions['corrected']}")
    print(f"  adjudication: {statuses['needs_adjudication']}")
    print(f"  exclude    : {statuses['exclude_candidate']}")
    print(f"  Etiketler  : {dict(labels)}")

# LLM preannotations
llm_path = BASE / "llm_preannotations.jsonl"
if llm_path.exists():
    llm = [json.loads(l) for l in llm_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"LLM pre-anotasyon: {len(llm)} ornek")
    print(f"  Etiket: {dict(collections.Counter(r['llm_label'] for r in llm))}")
    print(f"  Kural duzeltmesi: {sum(1 for r in llm if r.get('label_corrected_by_rule'))}")
else:
    print("LLM pre-anotasyon: yok")

print()
analyze_reviews(BASE / "dev600_candidate_label_review.jsonl", "Aday etiket incelemesi (LLM destekli)")
print()
analyze_reviews(BASE / "dev600_blind_annotator_1.jsonl", "Kor insan (annotator_1)")
print()
analyze_reviews(BASE / "dev600_blind_annotator_2.jsonl", "Kor insan (annotator_2)")