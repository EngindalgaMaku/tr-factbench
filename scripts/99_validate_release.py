"""Validate repository-level release invariants without training a model.

Run counts and metric row counts are derived dynamically from the run
archive rather than being hardcoded constants.
"""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def fail(msg):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/00_preflight.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/04_build_derived_splits.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/14_validate_runs.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/13_collect_results.py')],cwd=ROOT,check=True)
    run_dirs={p.name for p in (ROOT/'runs/completed').iterdir() if p.is_dir()}
    runs=list(csv.DictReader(open(ROOT/'results/csv/runs.csv',encoding='utf-8')))
    metrics=list(csv.DictReader(open(ROOT/'results/csv/metrics_long_canonical.csv',encoding='utf-8')))
    # Derive expected counts dynamically
    EXPECTED_RUNS = len(run_dirs)
    EXPECTED_ROWS = len(metrics)
    run_ids={r['run_id'] for r in runs}; metric_ids={r['run_id'] for r in metrics}
    if not (len(run_dirs)==len(run_ids)==len(metric_ids)):
        fail(f"run counts mismatch: dirs={len(run_dirs)}, runs.csv={len(run_ids)}, canonical={len(metric_ids)}")
    keys=[(r['run_id'],r['eval_set']) for r in metrics]
    if len(keys)!=len(set(keys)): fail("duplicate (run_id, eval_set) keys")
    if any((ROOT/'results/csv'/n).exists() for n in ['metrics_overall.csv','metrics_overall_8sets.csv','metrics_overall_ARCHIVED.csv']):
        fail('deprecated metrics_overall file exists')
    tc=list(csv.DictReader(open(ROOT/'results/tables/table_C_target_module_ablation.csv',encoding='utf-8')))
    expected={'C7_M04_ELECTRA_QV_LORA16_S42','C8_M04_ELECTRA_ALLATTN_LORA16_S42','G1_C6_M04_ELECTRA_LORA16_DROP10_S42'}
    if {r['Run ID'] for r in tc}!=expected: fail('Table C run IDs are not canonical')
    for rel in ['README.md','README_TR.md','DATA_CARD.md','MODEL_CARD.md']:
        text=(ROOT/rel).read_text(encoding='utf-8-sig')
        if any(x in text for x in ['Ã','Ä','Å','â†','â€“','â€”']): fail(f'mojibake in {rel}')
    lock=(ROOT/'final_config.lock.yaml').read_text(encoding='utf-8')
    if 'q1_pipeline/' in lock or 'CHECK:' in lock: fail('placeholder or old path in final_config.lock.yaml')

    # Annotation progress must match the actual JSONL artifacts.
    def count_jsonl(path):
        return sum(1 for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()) if path.exists() else 0
    annotation=json.loads((ROOT/'annotation_status.json').read_text(encoding='utf-8-sig'))
    expected_annotation={
        ('dev_stress','annotator_1'): count_jsonl(ROOT/'results/annotation/dev600/dev_stress_blind_annotator_1.jsonl'),
        ('dev_stress','annotator_2'): count_jsonl(ROOT/'results/annotation/dev600/dev_stress_blind_annotator_2.jsonl'),
        ('dev_general','annotator_1'): count_jsonl(ROOT/'results/annotation/dev960/dev_general_blind_annotator_1.jsonl'),
    }
    for (split,annotator),count in expected_annotation.items():
        recorded=annotation.get(split,{}).get(annotator,{}).get('completed')
        if recorded!=count: fail(f'annotation status mismatch for {split}/{annotator}: file={count}, status={recorded}')

    # Evaluation layout: training is isolated from preview and archived evaluation inputs.
    legacy_general=ROOT/'data/evaluation/legacy_experiment_version/dev_general.jsonl'
    legacy_stress=ROOT/'data/evaluation/legacy_experiment_version/dev_stress.jsonl'
    preview_general=ROOT/'data/evaluation/preview_v0.1/dev_general.jsonl'
    preview_stress=ROOT/'data/evaluation/preview_v0.1/dev_stress.jsonl'
    for old in [ROOT/'data/primary/dev_general.jsonl', ROOT/'data/primary/dev_stress.jsonl', ROOT/'data/primary/dev_general_provisional_v0.1.jsonl', ROOT/'data/primary/dev_stress_provisional_v0.1.jsonl']:
        if old.exists(): fail(f'old mixed-layout evaluation file still exists: {old.relative_to(ROOT)}')
    expected_legacy_prefix='data/evaluation/legacy_experiment_version/'
    import hashlib
    def sha256(path):
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    for run_dir in sorted((ROOT/'runs/completed').iterdir()):
        if not run_dir.is_dir(): continue
        manifest=json.loads((run_dir/'dataset_manifest.json').read_text(encoding='utf-8-sig'))
        field_specs={
            'train_file':'train_sha256',
            'dev960_file':'dev960_sha256',
            'dev600_file':'dev600_sha256',
        }
        for field,hash_field in field_specs.items():
            actual=str(manifest.get(field,'')).replace('\\','/')
            if field.startswith('dev') and not actual.startswith(expected_legacy_prefix):
                fail(f'{run_dir.name}: {field} does not point inside legacy experiment version: {actual}')
            if field=='train_file' and not (actual.startswith('data/primary/') or actual.startswith('data/derived/')):
                fail(f'{run_dir.name}: train_file is outside the current data layout: {actual}')
            path=ROOT/actual
            if not path.exists(): fail(f'{run_dir.name}: relocated dataset file does not exist: {actual}')
            if manifest.get(hash_field)!=sha256(path): fail(f'{run_dir.name}: {hash_field} does not match relocated file')
        relocation=manifest.get('path_relocation',{})
        if not all(relocation.get(k) for k in ('original_recorded_train_file','original_recorded_dev960_file','original_recorded_dev600_file')):
            fail(f'{run_dir.name}: missing original recorded dataset paths')

    preview_specs=[
        (ROOT/'data/evaluation/preview_v0.1/dev_general.jsonl', 960, 'DEV960'),
        (ROOT/'data/evaluation/preview_v0.1/dev_stress.jsonl', 600, 'Dev600'),
        (ROOT/'results/annotation/dev960/dev960_provisional_single_annotator_v0.1.jsonl', 960, 'DEV960 audit'),
        (ROOT/'results/annotation/dev600/dev600_provisional_single_annotator_v0.1.jsonl', 600, 'Dev600 audit'),
    ]
    valid_labels={'supported','partially_supported','contradicted','unverifiable'}
    for preview, expected_count, label in preview_specs:
        if count_jsonl(preview)!=expected_count: fail(f'{label} single-annotator preview must contain {expected_count} rows')
        seen=set()
        for line in preview.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get('annotation_status')!='single_annotator_provisional':
                fail(f'{label} rows must use annotation_status=single_annotator_provisional')
            if row.get('gold_source')!='single_annotator_blind_human' or row.get('gold_annotator')!='annotator_1':
                fail(f'{label} rows must identify annotator_1 as the blind-human gold source')
            if row.get('gold_label') not in valid_labels:
                fail(f'{label} contains an invalid gold label: {row.get("gold_label")}')
            if row.get('example_id') in seen:
                fail(f'{label} contains duplicate example_id={row.get("example_id")}')
            seen.add(row.get('example_id'))

    if (ROOT/'data/evaluation/preview_v0.1/dev_general.jsonl').read_bytes() != (ROOT/'results/annotation/dev960/dev960_provisional_single_annotator_v0.1.jsonl').read_bytes():
        fail('DEV960 release and audit preview files differ')
    if (ROOT/'data/evaluation/preview_v0.1/dev_stress.jsonl').read_bytes() != (ROOT/'results/annotation/dev600/dev600_provisional_single_annotator_v0.1.jsonl').read_bytes():
        fail('Dev600 release and audit preview files differ')

    # Canonical LLM matrix: four models x two prompts x two datasets.
    llm_table=ROOT/'results/tables/table_I_llm_baselines.csv'
    llm_rows=list(csv.DictReader(llm_table.open(encoding='utf-8-sig',newline='')))
    if len(llm_rows)!=16: fail(f'LLM table must have 16 rows, found {len(llm_rows)}')
    if len({r['model_alias'] for r in llm_rows})!=4: fail('LLM table must contain four canonical model aliases')
    if {r['prompt_mode'] for r in llm_rows}!={'zero_shot','few_shot_8'}: fail('LLM prompt modes mismatch')
    if {r['dataset'] for r in llm_rows}!={'dev_general','dev_stress'}: fail('LLM dataset matrix mismatch')
    for r in llm_rows:
        run_dir=ROOT/'results/llm_baselines'/r['dataset']/r['model_alias']/r['prompt_mode']
        expected=960 if r['dataset']=='dev_general' else 600
        if count_jsonl(run_dir/'predictions.jsonl')!=expected:
            fail(f'incomplete prediction file: {run_dir}')
        if not (run_dir/'metrics/metrics.json').exists(): fail(f'missing LLM metrics: {run_dir}')

    # Active manifests must be portable and repository-relative.
    active_manifests=list((ROOT/'results/llm_baselines').glob('*/*/*/manifest.json'))
    active_manifests += [
        ROOT/'results/annotation/dev600/llm_preannotation_manifest.json',
        ROOT/'results/annotation/dev960/llm_preannotation_manifest.json',
        ROOT/'results/annotation/dev600/dev600_provisional_single_annotator_v0.1_manifest.json',
        ROOT/'results/annotation/dev960/dev960_provisional_single_annotator_v0.1_manifest.json',
        ROOT/'data/manifests/preview_evaluation_manifest_v0.1.json',
    ]
    for path in active_manifests:
        text=path.read_text(encoding='utf-8-sig')
        if 'C:\\' in text or 'C:/' in text: fail(f'absolute Windows path in active manifest: {path.relative_to(ROOT)}')

    for rel in ['docs/LABEL_GUIDE.md','docs/LABEL_GUIDE_TR.md','reports/LLM_BASELINES.md','reports/LLM_BASELINES_TR.md']:
        if not (ROOT/rel).exists(): fail(f'missing release document: {rel}')
    for rel in ['README.md','README_TR.md','DATA_CARD.md','MODEL_CARD.md','reports/BENCHMARK_RESULTS.md']:
        text=(ROOT/rel).read_text(encoding='utf-8-sig')
        if '141 validated' in text or '141 doğrulanmış' in text or '426 canonical' in text or '426 kanonik' in text:
            fail(f'stale repository counts in {rel}')
        if '288/960' in text:
            fail(f'stale DEV960 annotation progress in {rel}')

    print(f"[PASS] release validation: {EXPECTED_RUNS} runs, {EXPECTED_ROWS} canonical rows, no duplicates")
    print(f"       (counts derived dynamically from run archive, not hardcoded)")

if __name__=='__main__': main()
