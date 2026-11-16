"""Post-review analyses for the LRE manuscript (no model training).
1 lemma overlap all splits  2 Fisher test  3 char n-gram overlap scan  4 claim cue productivity/coverage  5 hard subsets.
Usage: python scripts/97_post_review_analyses.py <repo_root> results/statistics/POST_REVIEW_ANALYSES_v1.json"""
import json, csv, re, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

R = sys.argv[1]; OUT = {}
L = lambda p: [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]
LABS = ['supported', 'partially_supported', 'contradicted', 'unverifiable']
train = L(f'{R}/data/primary/train.jsonl'); dg = L(f'{R}/data/evaluation/preview_v0.1/dev_general.jsonl')
ds = L(f'{R}/data/evaluation/preview_v0.1/dev_stress.jsonl'); gold = L(f'{R}/data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl')
for g in gold: g['label'] = g['gold_label']
for s in (train, dg, ds):
    for r in s: r['label'] = r['gold_label']
splits = {'train': train, 'dev_general': dg, 'dev_stress': ds, 'heldout': gold}

# 1 lemma overlap
tp = json.load(open(f'{R}/results/statistics/TURKISH_PHENOMENA_v1.json', encoding='utf-8'))
o = {}
for k, n in [('train', 'train'), ('dev_general_reviewed', 'dev_general'), ('dev_stress_reviewed', 'dev_stress'), ('gold480', 'heldout')]:
    o[n] = {l: (round(tp[k][l]['overlap_word'], 3), round(tp[k][l]['overlap_lemma'], 3), round(tp[k][l]['overlap_lemma'] - tp[k][l]['overlap_word'], 3)) for l in LABS}
diffs = [v[2] for s in o.values() for v in s.values()]
OUT['lemma_overlap'] = {'by_split': o, 'diff_min': min(diffs), 'diff_max': max(diffs)}

# 2 Fisher: encoder error rate on disputed vs agreed examples
ab = {r['release_example_id']: r['agree'] == 'True' for r in csv.DictReader(open(f'{R}/data/evaluation/gold_v1.0/agreement/AB_ALIGNED_480_v1.0.csv', encoding='utf-8'))}
pred = {r['release_example_id']: r['predicted_label'] for r in L(f'{R}/data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl')}
gl = {g['release_example_id']: g['gold_label'] for g in gold}
dis_err = sum(1 for i, a in ab.items() if not a and pred[i] != gl[i]); dis_n = sum(1 for a in ab.values() if not a)
agr_err = sum(1 for i, a in ab.items() if a and pred[i] != gl[i]); agr_n = sum(1 for a in ab.values() if a)
orr, p = fisher_exact([[dis_err, dis_n - dis_err], [agr_err, agr_n - agr_err]])
OUT['fisher_disputed_vs_agreed'] = {'disputed': [dis_err, dis_n], 'agreed': [agr_err, agr_n], 'odds_ratio': round(orr, 2), 'p_two_sided': round(p, 4)}

# 3 char n-gram similarity scan
def ctx(rows): return list(dict.fromkeys(r['context'] for r in rows))
C = {k: ctx(v) for k, v in splits.items()}
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
vec.fit(sum(C.values(), []))
X = {k: vec.transform(v) for k, v in C.items()}
def scan(a, b):
    S = linear_kernel(X[a], X[b]); m = S.max(axis=1)
    return {'n': len(m), 'max': round(float(m.max()), 3), 'p95': round(float(np.percentile(m, 95)), 3), 'n_ge_0.45': int((m >= .45).sum()), 'n_ge_0.60': int((m >= .60).sum()), 'n_ge_0.80': int((m >= .80).sum())}
OUT['char_ngram_scan'] = {
    'heldout_vs_train': scan('heldout', 'train'), 'heldout_vs_dev_general': scan('heldout', 'dev_general'), 'heldout_vs_dev_stress': scan('heldout', 'dev_stress'),
    'dev_general_vs_train': scan('dev_general', 'train'), 'dev_stress_vs_train': scan('dev_stress', 'train'), 'dev_stress_vs_dev_general': scan('dev_stress', 'dev_general')}
# top pairs dev->train above 0.80 for inspection
S = linear_kernel(X['dev_general'], X['train']); hi = np.argwhere(S >= .8)
OUT['char_ngram_scan']['examples_dev_general_train_ge_0.80'] = [[C['dev_general'][i][:90], C['train'][j][:90], round(float(S[i, j]), 3)] for i, j in hi[:5]]

# 4 claim cue productivity / coverage (word unigrams, Turkish lower-casing)
def tl(s): return s.replace('I', 'ı').replace('İ', 'i').lower()
def toks(s): return set(re.findall(r"[a-zçğıöşü]+|;", tl(s)))
cue = {}
for name, rows in splits.items():
    n = len(rows); cnt = defaultdict(Counter)
    for r in rows:
        for t in toks(r['claim']): cnt[t][r['label']] += 1
    res = []
    for t, c in cnt.items():
        k = sum(c.values()); cov = k / n
        if k < max(10, 0.02 * n): continue
        lab, top = c.most_common(1)[0]; prod = top / k
        res.append((t, round(cov, 3), round(prod, 3), lab, k))
    res.sort(key=lambda x: -(x[2] - 0.25) * x[1])
    cue[name] = res[:12]
    cue[name + '_selected'] = {t: next(([cv, pr, lb] for (tt, cv, pr, lb, k) in res if tt == t), None) for t in ['ve', 'değil', 'yok', ';', 'zorunlu', 'sadece', 'yalnızca', 'her', 'tüm']}
    # manuscript table (Table "cues"): fixed word list, no frequency threshold
    tab = {}
    for t in ['ve', ';', 'veya', 'yalnız', 'ancak', 'bütün', 'aynı']:
        c = cnt[t]; k = sum(c.values())
        tab[t] = [round(k / n, 3), round(c.most_common(1)[0][1] / k, 3) if k else None, c.most_common(1)[0][0] if k else None, k]
    cue[name + '_table'] = tab
OUT['claim_cues'] = cue

# 5 hard subsets
strong = {'gpt8': 'openai_gpt-4.1-mini/few_shot_8', 'gemma0': 'google_gemma-4-26b-a4b-it/zero_shot', 'gemma8': 'google_gemma-4-26b-a4b-it/few_shot_8', 'llama8': 'meta-llama_llama-3.3-70b-instruct/few_shot_8'}
P = {}
for k, d in strong.items():
    rr = L(f'{R}/results/llm_baselines/gold_v1.0/{d}/predictions.jsonl')
    lk = [x for x in rr[0] if 'pred' in x and 'label' in x][0]; ik = [x for x in rr[0] if 'example_id' in x][0]
    P[k] = {r[ik]: r[lk] for r in rr}
wrong2 = {i for i in gl if sum(P[k][i] != gl[i] for k in P) >= 2}
disputed = {i for i, a in ab.items() if not a}
dom = lambda i: 'finance' if int(i[-4:]) <= 160 else ('legal' if int(i[-4:]) <= 320 else 'medical')
med_u = {i for i in gl if dom(i) == 'medical' and gl[i] == 'unverifiable'}
hard = disputed | wrong2 | med_u
OUT['hard_subset_circular_do_not_use_for_strong_models'] = {'disputed': len(disputed), 'wrong_for_>=2_of_4_strongest': len(wrong2), 'medical_unverifiable': len(med_u), 'union': len(hard),
    'acc_on_hard': {k: round(sum(P[k][i] == gl[i] for i in hard) / len(hard), 3) for k in P} | {'encoder': round(sum(pred[i] == gl[i] for i in hard) / len(hard), 3)},
    'ids': sorted(hard)}
# model-independent hard subset: disputed examples plus medical unverifiable claims (no model output used)
mi = sorted(disputed | med_u); rest = sorted(set(gl) - set(mi))
acc = lambda pr, ids: round(sum(pr[i] == gl[i] for i in ids) / len(ids), 3)
names = {'gpt8': 'gpt-4.1-mini 8-shot', 'gemma0': 'gemma zero-shot', 'gemma8': 'gemma 8-shot', 'llama8': 'llama 8-shot'}
OUT['hard_subset_model_independent'] = {'definition': '48 annotator-disputed examples plus 40 medical unverifiable examples (union, 83)', 'n': len(mi),
    'accuracy_hard_vs_rest': {'encoder': [acc(pred, mi), acc(pred, rest)]} | {names[k]: [acc(P[k], mi), acc(P[k], rest)] for k in P},
    'note': 'hard_subset (union with model-error criterion) is circular for the four strongest models; use the model-independent subset for comparisons.',
    'ids': mi}
json.dump(OUT, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if 'hard' not in k else {a: b for a, b in v.items() if a != 'ids'}) for k, v in OUT.items()}, ensure_ascii=False, indent=1)[:6000])
