import json, pathlib

models = {
    'ELECTRA':      ['F1_M04_ELECTRA_LORA16_S42','F1_M04_ELECTRA_LORA16_S43','F1_M04_ELECTRA_LORA16_S44'],
    'ConvBERT':     ['F1_M03_CONVBERT_LORA16_S42','F1_M03_CONVBERT_LORA16_S43','F1_M03_CONVBERT_LORA16_S44'],
    'BERT-128k':    ['F1_M02_BERT128K_LORA16_S42','F1_M02_BERT128K_LORA16_S43','F1_M02_BERT128K_LORA16_S44'],
    'BERT-32k-r32': ['F1_M01_BERT32K_LORA32_S42','F1_M01_BERT32K_LORA32_S43','F1_M01_BERT32K_LORA32_S44'],
    'BERT-32k-FT':  ['F1_M01_BERT32K_FULLFT_S43','F1_M01_BERT32K_FULLFT_S44','F1_M01_BERT32K_FULLFT_S45'],
    'mBERT':        ['F1_M07_MBERT_LORA16_S42','F1_M07_MBERT_LORA16_S43','F1_M07_MBERT_LORA16_S44'],
}

base = pathlib.Path('runs/completed')
print(f"{'Model':<14} | {'DEV960 avg':>10} | {'DEV600 avg':>10} | Individual DEV960 seeds")
print('-'*75)
for name, runs in models.items():
    d960s, d600s = [], []
    for r in runs:
        m = json.loads((base / r / 'metrics.json').read_text(encoding='utf-8'))
        d960s.append(m['d960_macro_f1'])
        d600s.append(m['d600_macro_f1'])
    avg960 = sum(d960s)/len(d960s)
    avg600 = sum(d600s)/len(d600s)
    vals = ' / '.join(f'{v*100:.2f}' for v in d960s)
    print(f"{name:<14} | {avg960*100:>9.2f}% | {avg600*100:>9.2f}% | {vals}")