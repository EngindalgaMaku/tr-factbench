"""
scripts/40_c_ablation_target_modules.py

ELECTRA target-module ablasyonu — 3 izole koşu.

Model: ELECTRA-TR, r=16, alpha=32, seed=42, train4800
Koşular:
  C7_M04_ELECTRA_QV_LORA16_S42       — query + value (attention-only, minimal)
  C8_M04_ELECTRA_ALLATTN_LORA16_S42  — query + key + value + attention.output.dense
  C9_M04_ELECTRA_ALLLINEAR_LORA16_S42 — all-linear (full: attn + FFN)

Her koşu üretir:
  resolved_target_modules.json
  trainable_parameter_names.txt
  trainable_parameter_summary.json
  predictions_DEV960-GEN.jsonl / predictions_DEV600-STR.jsonl
  metrics.json

Assertion kontrolleri:
  qv_params < attn_params < alllinear_params
  attn koşusunda FFN modülü OLMAMALI
  alllinear koşusunda FFN modülü OLMALI
"""
import sys, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR = Q1_DIR / "runs/completed"

ELECTRA_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
TRAIN_FILE    = str(Q1_DIR / "data/primary/train.jsonl")
DEV960_FILE   = str(Q1_DIR / "data/evaluation/legacy_experiment_version/dev_general.jsonl")
DEV600_FILE   = str(Q1_DIR / "data/evaluation/legacy_experiment_version/dev_stress.jsonl")

ABLATION_RUNS = [
    {
        "run_id": "C7_M04_ELECTRA_QV_LORA16_S42",
        "target_modules": "query,value",
        "description": "Attention q+v only (minimal)",
    },
    {
        "run_id": "C8_M04_ELECTRA_ALLATTN_LORA16_S42",
        "target_modules": "query,key,value,attention.output.dense",
        "description": "All attention projections (no FFN)",
    },
    {
        # B1 already exists — skip training, just verify
        "run_id": "C9_M04_ELECTRA_ALLLINEAR_LORA16_S42",
        "target_modules": "all-linear",
        "description": "All linear including FFN",
    },
]

print("=" * 60)
print("C ablasyon: ELECTRA target-module izole deney")
print("=" * 60)

results = {}

for cfg in ABLATION_RUNS:
    rid   = cfg["run_id"]
    mods  = cfg["target_modules"]
    desc  = cfg["description"]
    rdir  = RUNS_DIR / rid

    # C9 için B1 mevcut checkpoint kullanılabilir — zaten eğitildi
    b1_dir = RUNS_DIR / "B1_M04_ELECTRA_ALLLINEAR_LORA16_S42"
    if rid == "C9_M04_ELECTRA_ALLLINEAR_LORA16_S42" and b1_dir.exists():
        print(f"\n[{rid}] B1 mevcut koşu kullanılacak → trainable_parameters.json kontrol ediliyor")
        tp = json.load(open(b1_dir/"trainable_parameters.json", encoding="utf-8"))
        n_params = tp["num_trainable_params"]
        names = tp["trainable_param_names"]
        has_ffn = any("intermediate" in n or ("output.dense" in n and "attention" not in n) for n in names)
        print(f"  trainable_params={n_params:,} has_ffn={has_ffn}")
        results[rid] = {"params": n_params, "names": names, "has_ffn": has_ffn, "run_dir": b1_dir}
        continue

    print(f"\n[{rid}] Çalıştırılıyor: {desc}")
    print(f"  target_modules: {mods}")

    cmd = [
        sys.executable,
        str(Q1_DIR / "scripts/10_run_experiment.py"),
        "--run_id", rid,
        "--model_name", ELECTRA_MODEL,
        "--train_file", TRAIN_FILE,
        "--dev960_file", DEV960_FILE,
        "--dev600_file", DEV600_FILE,
        "--method", "lora",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_target_modules", mods,
        "--learning_rate", "0.0002",
        "--seed", "42",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  [HATA] {rid}: {result.stderr[-500:]}")
        continue

    # F1 satırı çıkar
    for line in result.stdout.split("\n"):
        if "DEV960" in line and "F1" in line:
            print(f"  {line.strip()}")

    # trainable_parameters.json oku
    tp_path = rdir / "trainable_parameters.json"
    if not tp_path.exists():
        print(f"  [WARN] trainable_parameters.json bulunamadı")
        continue
    tp = json.load(open(tp_path, encoding="utf-8"))
    n_params = tp["num_trainable_params"]
    names = tp["trainable_param_names"]
    has_ffn = any("intermediate" in n or ("output.dense" in n and "attention" not in n) for n in names)
    results[rid] = {"params": n_params, "names": names, "has_ffn": has_ffn, "run_dir": rdir}
    print(f"  trainable_params={n_params:,} has_ffn={has_ffn}")

    # resolved_target_modules.json kaydet
    resolved = {
        "run_id": rid,
        "target_modules_arg": mods,
        "description": desc,
        "num_trainable_params": n_params,
        "has_ffn_modules": has_ffn,
        "unique_module_types": sorted(set(
            n.split(".")[-3] if "lora_A" in n else n.split(".")[-2]
            for n in names if "lora_A" in n
        ))
    }
    with open(rdir / "resolved_target_modules.json", "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)

    # trainable_parameter_names.txt
    with open(rdir / "trainable_parameter_names.txt", "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

# ── Assertions ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Assertion kontrolleri")
print("=" * 60)

r_qv   = results.get("C7_M04_ELECTRA_QV_LORA16_S42")
r_attn = results.get("C8_M04_ELECTRA_ALLATTN_LORA16_S42")
r_all  = results.get("C9_M04_ELECTRA_ALLLINEAR_LORA16_S42")

errors = []

if r_qv and r_attn:
    if not (r_qv["params"] < r_attn["params"]):
        errors.append(f"FAIL: qv_params ({r_qv['params']}) < attn_params ({r_attn['params']}) bekleniyor")
    else:
        print(f"  OK: qv_params ({r_qv['params']:,}) < attn_params ({r_attn['params']:,})")

if r_attn and r_all:
    if not (r_attn["params"] < r_all["params"]):
        errors.append(f"FAIL: attn_params ({r_attn['params']}) < alllinear_params ({r_all['params']}) bekleniyor")
    else:
        print(f"  OK: attn_params ({r_attn['params']:,}) < alllinear_params ({r_all['params']:,})")

if r_attn:
    if r_attn["has_ffn"]:
        errors.append("FAIL: C8 (all-attention) FFN modülü içermemeli")
    else:
        print(f"  OK: C8 (all-attention) FFN yok")

if r_all:
    if not r_all["has_ffn"]:
        errors.append("FAIL: C9 (all-linear) FFN modülü içermeli")
    else:
        print(f"  OK: C9 (all-linear) FFN var")

if errors:
    print("\nASSERTION HATALARI:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

# ── Özet tablo ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ablasyon özet")
print("=" * 60)
for cfg in ABLATION_RUNS:
    rid = cfg["run_id"]
    r   = results.get(rid)
    if not r:
        print(f"  {rid}: YOK")
        continue
    # F1 metrics.json'dan oku
    mdir = r["run_dir"]
    mpath = mdir / "metrics.json"
    if mpath.exists():
        mm = json.load(open(mpath, encoding="utf-8"))
        f1 = mm.get("d960_macro_f1", "?")
    else:
        # predictions'tan hesapla
        from sklearn.metrics import f1_score
        L = {"supported":0,"partially_supported":1,"contradicted":2,"unverifiable":3}
        pred_p = mdir / "predictions_DEV960-GEN.jsonl"
        if pred_p.exists():
            rows = [json.loads(l) for l in open(pred_p,encoding="utf-8") if l.strip()]
            gold = [L[rr["gold_label"]] for rr in rows]
            pred = [L[rr["predicted_label"]] for rr in rows]
            f1 = round(float(f1_score(gold,pred,average="macro",zero_division=0)),4)
        else:
            f1 = "?"
    has_ffn = r["has_ffn"]
    print(f"  {rid}: params={r['params']:,} has_ffn={has_ffn} DEV960_F1={f1}")

print("\n[TAMAMLANDI] 40_c_ablation_target_modules.py")
