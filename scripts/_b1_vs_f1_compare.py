"""B1 tek-seed vs F1 3-seed ortalama karsilastirma"""
b1_single = {
    'ELECTRA':      90.75,
    'ConvBERT':     88.57,
    'BERT-128k':    86.60,
    'BERT-32k-r32': 82.12,
    'BERT-32k-FT':  81.89,
    'mBERT':        79.65,
}
f1_avg = {
    'ELECTRA':      90.58,
    'ConvBERT':     87.35,
    'BERT-128k':    80.97,
    'BERT-32k-r32': 83.59,
    'BERT-32k-FT':  84.44,
    'mBERT':        77.59,
}
print(f"{'Model':<14} | {'B1 S42':>7} | {'F1 avg':>7} | {'Delta':>7} | Not")
print('-'*65)
for name in ['ELECTRA','ConvBERT','BERT-128k','BERT-32k-r32','BERT-32k-FT','mBERT']:
    b = b1_single[name]
    f = f1_avg[name]
    d = f - b
    flag = ' *** BUYUK FARK' if abs(d) > 3 else ' ok'
    print(f"{name:<14} | {b:>6.2f}% | {f:>6.2f}% | {d:>+6.2f}pp |{flag}")