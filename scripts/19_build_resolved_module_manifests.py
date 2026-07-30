"""Build resolved LoRA target-module manifests from trainable parameter names."""
from __future__ import annotations
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
RUNS=ROOT/'runs/completed'

def build(run:Path):
    tp=run/'trainable_parameters.json'
    if not tp.exists(): return False
    data=json.loads(tp.read_text(encoding='utf-8-sig'))
    modules=[]
    for name in data.get('trainable_param_names') or []:
        if '.lora_A.' in name: modules.append(name.split('.lora_A.')[0])
        elif '.lora_B.' in name: modules.append(name.split('.lora_B.')[0])
    modules=sorted(set(modules))
    if not modules: return False
    attn=[m for m in modules if '.attention.' in m]
    ffn=[m for m in modules if '.intermediate.' in m or re.search(r'\.encoder\.layer\.\d+\.output\.dense$',m)]
    other=[m for m in modules if m not in attn and m not in ffn]
    cfg={}
    if (run/'config.json').exists(): cfg=json.loads((run/'config.json').read_text(encoding='utf-8-sig'))
    manifest={
        'requested_target_modules':cfg.get('lora_target_modules'),
        'resolved_target_modules':modules,
        'resolved_module_count':len(modules),
        'attention_modules':attn,'attention_module_count':len(attn),
        'ffn_modules':ffn,'ffn_module_count':len(ffn),
        'other_modules':other,'other_module_count':len(other),
        'has_ffn_modules':bool(ffn),
        'num_trainable_params':data.get('num_trainable_params'),
        'num_total_params':data.get('num_total_params'),
        'source':'trainable_parameters.json',
    }
    (run/'resolved_target_modules.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return True

def main():
    n=0
    for run in sorted(RUNS.iterdir()):
        if run.is_dir() and build(run): n+=1
    print(f'[OK] resolved_target_modules.json generated for {n} LoRA runs')

if __name__=='__main__': main()
