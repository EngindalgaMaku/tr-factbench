"""Seed-level and prediction-level statistical comparisons for historical D1 runs.

The script uses only explicit seed mappings from d1_summary.json. Paired tests are
computed on common seed identifiers; unequal seed sets are never paired by position.
"""
from __future__ import annotations
import csv, json, warnings
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent.parent
STAT_DIR=ROOT/'results/statistics'; FIG_DIR=ROOT/'results/figures/historical'; TABLE_DIR=ROOT/'results/tables'; RUNS_DIR=ROOT/'runs/completed'
N_BOOTSTRAP=10000
rng=np.random.default_rng(42)

d1=json.loads((STAT_DIR/'d1_summary.json').read_text(encoding='utf-8'))
config_maps={
    'ELECTRA_LoRA_r16': d1['electra_lora_r16']['seed_scores_by_id'],
    'ELECTRA_Freeze50': d1['electra_freeze50']['seed_scores_by_id'],
    'BERTurk_FullFT': d1['bert_full_ft']['seed_scores_by_id'],
}
labels={
    'ELECTRA_LoRA_r16':'ELECTRA-TR LoRA r=16 (historical B1)',
    'ELECTRA_Freeze50':'ELECTRA-TR Freeze-50',
    'BERTurk_FullFT':'BERTurk Full FT',
}

def bootstrap_ci(values, n=N_BOOTSTRAP, alpha=.05):
    x=np.asarray(values,dtype=float)
    samples=rng.choice(x,size=(n,len(x)),replace=True).mean(axis=1)
    return {
        'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else 0.0,
        'ci95_lo':float(np.quantile(samples,alpha/2)),'ci95_hi':float(np.quantile(samples,1-alpha/2)),
        'n':len(x)
    }

ci={k:bootstrap_ci(list(v.values())) for k,v in config_maps.items()}
pairwise=[]
for a,b in combinations(config_maps,2):
    ma,mb=config_maps[a],config_maps[b]
    common=sorted(set(ma)&set(mb))
    xa=np.array([ma[s] for s in common],float); xb=np.array([mb[s] for s in common],float)
    row={'comparison':f'{a} vs {b}','common_seeds':common,'n_common':len(common)}
    if len(common)>=2:
        diff=xa-xb
        t=stats.ttest_rel(xa,xb)
        try: w=stats.wilcoxon(diff)
        except Exception: w=None
        row.update({'test_type':'paired_by_seed','mean_paired_difference_pp':float(diff.mean()*100),
                    'paired_t_stat':round(float(t.statistic), 15),'paired_t_p':round(float(t.pvalue), 15),
                    'wilcoxon_stat':round(float(w.statistic), 15) if w else None,'wilcoxon_p':round(float(w.pvalue), 15) if w else None})
    else:
        row.update({'test_type':'insufficient_common_seeds','mean_paired_difference_pp':None})
    pairwise.append(row)

# Prediction-level McNemar on clean F1 seed-43 runs.
def load_preds(run_id):
    p=RUNS_DIR/run_id/'predictions_DEV960-GEN.jsonl'
    out={}
    for line in p.read_text(encoding='utf-8').splitlines():
        r=json.loads(line); out[r['example_id']]=(r['gold_label'],r['predicted_label'])
    return out
mcnemar=None
run_a='F1_M04_ELECTRA_LORA16_S43'; run_b='F1_M01_BERT32K_FULLFT_S43'
pa,pb=load_preds(run_a),load_preds(run_b)
ids=sorted(set(pa)&set(pb)); n10=n01=n11=n00=0
for eid in ids:
    ca=pa[eid][0]==pa[eid][1]; cb=pb[eid][0]==pb[eid][1]
    if ca and cb:n11+=1
    elif ca and not cb:n10+=1
    elif not ca and cb:n01+=1
    else:n00+=1
n=n10+n01
if n:
    chi2=(abs(n10-n01)-1)**2/n
    p=float(stats.chi2.sf(chi2,1))
    mcnemar={'run_a':run_a,'run_b':run_b,'n_common':len(ids),'n10_a_correct_b_wrong':n10,
             'n01_a_wrong_b_correct':n01,'n11_both_correct':n11,'n00_both_wrong':n00,
             'chi2_continuity_corrected':chi2,'p_value':p}

out={'scope':'historical D1 seed-level analysis; final release statistics pending locked configuration',
     'bootstrap_percentile_ci':ci,'paired_tests':pairwise,'mcnemar_f1_seed43':mcnemar,
     'caution':'Seed counts are small. Report effect sizes and success/failure rates alongside p-values.'}
(STAT_DIR/'d1_statistical_tests.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

with (TABLE_DIR/'table_stat_bootstrap_ci.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f, lineterminator="\n"); w.writerow(['Model','n','Mean_F1','Std','CI95_Lo','CI95_Hi'])
    for k,v in ci.items(): w.writerow([labels[k],v['n'],f"{v['mean']:.4f}",f"{v['std']:.4f}",f"{v['ci95_lo']:.4f}",f"{v['ci95_hi']:.4f}"])
with (TABLE_DIR/'table_stat_pairwise_tests.csv').open('w',newline='',encoding='utf-8') as f:
    cols=['comparison','common_seeds','n_common','test_type','mean_paired_difference_pp','paired_t_stat','paired_t_p','wilcoxon_stat','wilcoxon_p']
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore', lineterminator="\n");w.writeheader()
    for r in pairwise:
        r=dict(r);r['common_seeds']='/'.join(r['common_seeds']);w.writerow(r)
print('[OK] d1_statistical_tests.json and statistical tables')
