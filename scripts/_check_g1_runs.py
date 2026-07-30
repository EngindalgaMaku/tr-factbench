import pathlib, json, sys
sys.stdout.reconfigure(encoding="utf-8")
runs_dir = pathlib.Path("runs/completed")
g1_runs = sorted([r for r in runs_dir.iterdir() if r.name.startswith("G1_C6")])
print(f"G1_C6 mevcut ({len(g1_runs)}):")
for r in g1_runs:
    cfg = json.loads((r/"config.json").read_text(encoding="utf-8"))
    m_path = r / "metrics.json"
    if m_path.exists():
        m = json.loads(m_path.read_text(encoding="utf-8"))
        print(f"  {r.name}  dropout={cfg.get('lora_dropout','?')}  seed={cfg['seed']}  DEV960={m['d960_macro_f1']:.4f}")
    else:
        print(f"  {r.name}  dropout={cfg.get('lora_dropout','?')}  seed={cfg['seed']}  DEV960=INCOMPLETE")
print()
print("Eksik (metrics.json olmayan veya dizin hiç yok):")
for dropout, tag in [(0.0,"DROP0"),(0.05,"DROP5"),(0.10,"DROP10")]:
    for seed in [42,43,44]:
        name = f"G1_C6_M04_ELECTRA_LORA16_{tag}_S{seed}"
        p = runs_dir / name
        if not p.exists():
            print(f"  {name}  [dizin yok]")
        elif not (p/"metrics.json").exists():
            print(f"  {name}  [metrics.json eksik - kos tamamlanmamis]")