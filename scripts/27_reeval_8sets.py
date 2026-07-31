"""
scripts/27_reeval_8sets.py

Her başarılı A1, A3 ve TALL run'ı 8 eval setinde yeniden inference eder.
Yeniden eğitim yok — sadece best_model/ adapter ağırlıkları yüklenir.

8 eval seti:
  DEV960-GEN  n=960  (genel tam set)
  DEV960-MED  n=320  (medikal alt set)
  DEV960-FIN  n=320  (finans alt set)
  DEV960-LEG  n=320  (hukuk alt set)
  DEV600-STR  n=600  (stres tam set)
  DEV600-MED  n=200  (medikal stres)
  DEV600-FIN  n=200  (finans stres)
  DEV600-LEG  n=200  (hukuk stres)

Çıktılar:
  Her run klasörüne predictions_<SET>.jsonl dosyaları
  metrics_overall.csv yeniden yazılır (n_examples sütunu eklenir)
"""
import sys, json, csv, torch, argparse
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR = Q1_DIR / "runs/completed"
DATA_DIR = Q1_DIR / "data/derived"
CSV_DIR  = Q1_DIR / "results/csv"

LABEL2ID    = {"supported": 0, "partially_supported": 1, "contradicted": 2, "unverifiable": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
BATCH_SIZE  = 32
MAX_LEN     = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# 8 eval seti tanımları
LEGACY_EVAL_DIR = Q1_DIR / "data/evaluation/legacy_experiment_version"
DOM_DIR = Q1_DIR / "data/derived/dev_domain"

EVAL_SETS = [
    ("DEV960-GEN", LEGACY_EVAL_DIR / "dev_general.jsonl"),
    ("DEV960-MED", DOM_DIR / "dev960_general_medical.jsonl"),
    ("DEV960-FIN", DOM_DIR / "dev960_general_finance.jsonl"),
    ("DEV960-LEG", DOM_DIR / "dev960_general_legal.jsonl"),
    ("DEV600-STR", LEGACY_EVAL_DIR / "dev_stress.jsonl"),
    ("DEV600-MED", DOM_DIR / "dev600_stress_medical.jsonl"),
    ("DEV600-FIN", DOM_DIR / "dev600_stress_finance.jsonl"),
    ("DEV600-LEG", DOM_DIR / "dev600_stress_legal.jsonl"),
]

# Gerçek yolları düzelt
EVAL_SETS_RESOLVED = []
for name, p in EVAL_SETS:
    resolved = p.resolve()
    if resolved.exists():
        EVAL_SETS_RESOLVED.append((name, resolved))
    else:
        print(f"  [UYARI] {name}: {resolved} bulunamadı, atlanıyor")

# A1 + A3 + T-ALL başarılı run'lar
TARGET_PREFIXES = ("A1_M04_ELECTRA", "A3_M04_ELECTRA",
                   "B1_M04_ELECTRA", "B2_M04_ELECTRA")


def get_run_status(run_dir: Path) -> str:
    m = run_dir / "metrics.json"
    if not m.exists():
        return "unknown"
    return json.load(open(m, encoding="utf-8")).get("qc_status", "unknown")


def find_adapter_dir(run_dir: Path) -> Path:
    """best_model/ klasörünü bulur."""
    for candidate in run_dir.rglob("best_model"):
        if candidate.is_dir() and (candidate / "adapter_config.json").exists():
            return candidate
    return None


def find_base_model(run_dir: Path) -> str:
    m = run_dir / "metrics.json"
    if m.exists():
        return json.load(open(m, encoding="utf-8")).get("model_id", "")
    c = run_dir / "config.json"
    if c.exists():
        return json.load(open(c, encoding="utf-8")).get("model_name", "")
    return ""


def infer_on_file(model, tokenizer, data_file: Path):
    records = [json.loads(l) for l in open(data_file, encoding="utf-8")]
    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(records), BATCH_SIZE):
            batch   = records[i:i+BATCH_SIZE]
            texts_a = [f"[CONTEXT] {r.get('context', '')}" for r in batch]
            texts_b = [f"[QUESTION] {r.get('question', '')} [CLAIM] {r.get('claim', '')}" for r in batch]
            enc = tokenizer(texts_a, texts_b, max_length=MAX_LEN, truncation=True,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            for r in batch:
                lbl = r.get("gold_label", "")
                if lbl not in LABEL2ID:
                    raise ValueError(f"Record {r.get('example_id','?')}: unknown gold_label '{lbl}'")
                all_labels.append(LABEL2ID[lbl])

    acc = sum(g == p for g, p in zip(all_labels, all_preds)) / len(all_labels)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )
    preds_out = [
        {"example_id": r.get("example_id"), "domain": r.get("domain"),
         "gold_label": r.get("gold_label"),
         "predicted_label": ID2LABEL[p],
         "probabilities": {ID2LABEL[i]: float(prob[i]) for i in range(4)}}
        for r, p, prob in zip(records, all_preds, all_probs)
    ]
    return {
        "n_examples": len(records),
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "supported_f1": round(report["supported"]["f1-score"], 4),
        "partially_supported_f1": round(report["partially_supported"]["f1-score"], 4),
        "contradicted_f1": round(report["contradicted"]["f1-score"], 4),
        "unverifiable_f1": round(report["unverifiable"]["f1-score"], 4),
        "predictions": preds_out,
    }


def process_run(run_dir: Path, force: bool = False):
    run_id = run_dir.name
    status = get_run_status(run_dir)
    if status != "success":
        print(f"  [{run_id}] degenerate, atlanıyor")
        return None

    adapter_dir = find_adapter_dir(run_dir)
    if adapter_dir is None:
        print(f"  [{run_id}] adapter_dir bulunamadı, atlanıyor")
        return None

    base_model_id = find_base_model(run_dir)
    if not base_model_id:
        print(f"  [{run_id}] model_id bulunamadı, atlanıyor")
        return None

    print(f"\n  Loading: {run_id}")
    print(f"    Base: {base_model_id}")
    print(f"    Adapter: {adapter_dir.relative_to(run_dir)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_id, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"  [HATA] Model yükleme başarısız: {e}")
        return None

    run_results = {"run_id": run_id}
    for set_name, set_path in EVAL_SETS_RESOLVED:
        pred_file = run_dir / f"predictions_{set_name}.jsonl"
        if pred_file.exists() and not force:
            # Sadece n_examples kontrolü yap
            first = json.loads(open(pred_file, encoding="utf-8").readline())
            existing_count = sum(1 for _ in open(pred_file, encoding="utf-8"))
            expected = sum(1 for _ in open(set_path, encoding="utf-8"))
            if existing_count == expected:
                print(f"    {set_name}: zaten mevcut ({existing_count} örnek), atlanıyor")
                # Metrikleri dosyadan hesapla
                preds = [json.loads(l) for l in open(pred_file, encoding="utf-8")]
                all_g, all_p = [], []
                for r in preds:
                    for key, lst in (("gold_label", all_g), ("predicted_label", all_p)):
                        lbl = r.get(key, "")
                        if lbl not in LABEL2ID:
                            raise ValueError(f"Record {r.get('example_id','?')}: unknown {key} '{lbl}'")
                        lst.append(LABEL2ID[lbl])
                acc = sum(g == p for g, p in zip(all_g, all_p)) / len(all_g)
                rep = classification_report(all_g, all_p, labels=list(range(4)),
                                            target_names=LABEL_NAMES, output_dict=True, zero_division=0)
                run_results[set_name] = {
                    "n_examples": len(preds),
                    "accuracy": round(acc, 4),
                    "macro_f1": round(rep["macro avg"]["f1-score"], 4),
                    "supported_f1": round(rep["supported"]["f1-score"], 4),
                    "partially_supported_f1": round(rep["partially_supported"]["f1-score"], 4),
                    "contradicted_f1": round(rep["contradicted"]["f1-score"], 4),
                    "unverifiable_f1": round(rep["unverifiable"]["f1-score"], 4),
                }
                print(f"    {set_name}: F1={run_results[set_name]['macro_f1']:.4f}")
                continue

        result = infer_on_file(model, tokenizer, set_path)
        # Predictions kaydet
        with open(pred_file, "w", encoding="utf-8") as f:
            for row in result["predictions"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        run_results[set_name] = {k: v for k, v in result.items() if k != "predictions"}
        print(f"    {set_name}: n={result['n_examples']}, F1={result['macro_f1']:.4f}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return run_results


def write_metrics_overall(all_results: list):
    """metrics_long_canonical.csv'ye yeni satırlar ekler/günceller."""
    headers = ["run_id", "eval_set", "n_examples", "accuracy", "macro_f1",
               "supported_f1", "partially_supported_f1", "contradicted_f1",
               "unverifiable_f1", "source"]

    # Mevcut kanonik CSV'yi oku (TARGET_PREFIXES dışı satırlar korunacak)
    existing_path = CSV_DIR / "metrics_long_canonical.csv"
    preserved = []
    if existing_path.exists():
        for row in csv.DictReader(open(existing_path, encoding="utf-8")):
            rid = row["run_id"]
            if not any(rid.startswith(p) for p in TARGET_PREFIXES):
                preserved.append(row)

    # Yeni satırlar
    new_rows = []
    for run_result in all_results:
        if run_result is None:
            continue
        run_id = run_result["run_id"]
        for set_name, metrics in run_result.items():
            if set_name == "run_id":
                continue
            new_rows.append({
                "run_id": run_id,
                "eval_set": set_name,
                "n_examples": metrics.get("n_examples", ""),
                "accuracy": metrics.get("accuracy", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "supported_f1": metrics.get("supported_f1", ""),
                "partially_supported_f1": metrics.get("partially_supported_f1", ""),
                "contradicted_f1": metrics.get("contradicted_f1", ""),
                "unverifiable_f1": metrics.get("unverifiable_f1", ""),
                "source": "sklearn_predictions",
            })

    final_rows = list(preserved)
    final_rows.extend(new_rows)

    out = CSV_DIR / "metrics_long_canonical.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(final_rows)
    print(f"\n  [CSV] {out.name} ({len(new_rows)} yeni satır, {len(preserved)} korunan)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Mevcut predictions dosyalarını yeniden hesapla")
    parser.add_argument("--run_ids", nargs="*",
                        help="Yalnız belirtilen run_id'leri işle (varsayılan: tüm hedefler)")
    args = parser.parse_args()

    print("=" * 70)
    print("27_reeval_8sets — A1/A3/TALL Domain 8 Eval Set Inference")
    print(f"Device: {DEVICE}")
    print(f"Eval setleri: {len(EVAL_SETS_RESOLVED)}/8 bulundu")
    print("=" * 70)

    # Hedef run'ları topla
    target_runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if args.run_ids:
            if d.name not in args.run_ids:
                continue
        elif not any(d.name.startswith(p) for p in TARGET_PREFIXES):
            continue
        target_runs.append(d)

    print(f"\nHedef run sayısı: {len(target_runs)}")

    all_results = []
    for i, run_dir in enumerate(target_runs):
        print(f"\n[{i+1}/{len(target_runs)}] {run_dir.name}")
        result = process_run(run_dir, force=args.force)
        all_results.append(result)

    # Prediction files are updated; canonical CSV is rebuilt by script 13.
    out_csv = write_metrics_overall(all_results)

    print(f"\n{'='*70}")
    print(f"[TAMAMLANDI] {out_csv}")
    print(f"  Başarılı işlenen: {sum(1 for r in all_results if r is not None)}/{len(target_runs)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()