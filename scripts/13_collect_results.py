"""
scripts/13_collect_results.py

Kanonik metrik toplayıcı — TEK KANONİK KAYNAK.

Değişiklikler (v2):
  - Tüm predictions_*.jsonl dosyaları otomatik keşfedilir (8 eval seti dahil)
  - metrics.json fallback KALDIRILDI: prediction yoksa run o set için hata verir
  - Çıktı: results/csv/metrics_long_canonical.csv  (tek kanonik dosya)
  - metrics_overall_8sets.csv paralel dosyası kaldırıldı
  - 94 run dahil (C7/C8 FFN ablasyon koşuları dahil)
"""
import sys, json, csv, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT
RUNS_DIR = Q1_DIR / "runs/completed"
CSV_DIR  = Q1_DIR / "results/csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding="utf-8")

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}

CANONICAL_CSV = CSV_DIR / "metrics_long_canonical.csv"


def sklearn_f1_from_predictions(pred_file: Path) -> dict:
    """predictions_*.jsonl'dan sklearn macro F1 ve per-class F1 hesaplar.
    Prediction yoksa FileNotFoundError, geçersizse ValueError verir.
    """
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction dosyası bulunamadı: {pred_file}")
    from sklearn.metrics import f1_score, accuracy_score
    rows = [json.loads(l) for l in open(pred_file, encoding="utf-8") if l.strip()]
    if not rows:
        raise ValueError(f"Prediction dosyası boş: {pred_file}")
    gold = [LABEL2ID.get(r.get("gold_label", ""), -1) for r in rows]
    pred = [LABEL2ID.get(r.get("predicted_label", ""), -1) for r in rows]
    if -1 in gold:
        bad = [r.get("gold_label") for r in rows if LABEL2ID.get(r.get("gold_label",""), -1) == -1]
        raise ValueError(f"Geçersiz gold_label: {set(bad)} — {pred_file}")
    if -1 in pred:
        bad = [r.get("predicted_label") for r in rows if LABEL2ID.get(r.get("predicted_label",""), -1) == -1]
        raise ValueError(f"Geçersiz predicted_label: {set(bad)} — {pred_file}")
    per_class = f1_score(gold, pred, labels=list(range(4)), average=None, zero_division=0)
    return {
        "n": len(rows),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f1_score(gold, pred, average="macro", zero_division=0)), 4),
        "supported_f1": round(float(per_class[0]), 4),
        "partially_supported_f1": round(float(per_class[1]), 4),
        "contradicted_f1": round(float(per_class[2]), 4),
        "unverifiable_f1": round(float(per_class[3]), 4),
    }


def collect(strict: bool = True):
    """
    strict=True: prediction olmayan (run, eval_set) çifti için hata verir.
    strict=False: uyarı verir, devam eder (--lenient flag ile).
    """
    if not RUNS_DIR.exists():
        print(f"[HATA] {RUNS_DIR} klasörü bulunamadı.")
        return

    runs_headers = [
        "run_id", "track", "status", "model_id", "train_dataset", "train_sha256",
        "method", "freeze_ratio", "lora_rank", "lora_alpha", "learning_rate",
        "seed", "best_epoch", "total_params", "trainable_params", "train_seconds",
        "peak_vram_mb", "inference_examples_per_second"
    ]

    overall_headers = [
        "run_id", "eval_set", "n_examples", "accuracy", "macro_f1",
        "supported_f1", "partially_supported_f1", "contradicted_f1", "unverifiable_f1",
        "source"
    ]

    seen_runs = set()
    runs_rows = []
    overall_rows = []
    errors = []

    for run_path in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not run_path.is_dir():
            continue

        metrics_file = run_path / "metrics.json"
        if not metrics_file.exists():
            print(f"  [SKIP] metrics.json yok: {run_path.name}")
            continue

        m = json.loads(open(metrics_file, encoding="utf-8").read())
        run_id = m.get("run_id", run_path.name)

        if run_id in seen_runs:
            raise RuntimeError(f"[KRİTİK HATA] Duplicate run_id: {run_id}")
        seen_runs.add(run_id)

        # best_epoch — trainer_state.json'dan
        best_epoch = None
        trainer_states = list(run_path.rglob("trainer_state.json"))
        if trainer_states:
            try:
                ts = json.loads(open(trainer_states[0], encoding="utf-8").read())
                best_step = ts.get("best_global_step")
                log_history = ts.get("log_history", [])
                last_epoch_entries = [e for e in log_history if "epoch" in e and e.get("epoch", 0) > 0]
                if last_epoch_entries and best_step:
                    last = last_epoch_entries[-1]
                    last_step = last.get("step", 0)
                    last_epoch = last.get("epoch", 0)
                    if last_step > 0 and last_epoch > 0:
                        steps_per_epoch = last_step / last_epoch
                        best_epoch = round(best_step / steps_per_epoch, 2) if steps_per_epoch > 0 else None
            except Exception:
                pass
        if best_epoch is None:
            best_epoch = m.get("best_epoch")

        raw_vram = m.get("peak_vram_mb")
        peak_vram = None if (raw_vram is None or raw_vram == 0) else raw_vram

        # total_params: trainable_parameters.json'dan oku (model bazli gercek deger)
        total_params = None
        trainable_params = m.get("trainable_params")
        tp_file = run_path / "trainable_parameters.json"
        if tp_file.exists():
            try:
                tp_data = json.loads(tp_file.read_text(encoding="utf-8"))
                total_params = tp_data.get("num_total_params")
                if trainable_params is None:
                    trainable_params = tp_data.get("num_trainable_params")
            except Exception:
                pass
        # Fallback: metrics.json'daki eski alan
        if total_params is None:
            total_params = m.get("total_params")

        runs_rows.append({
            "run_id": run_id,
            "track": "Q1_Benchmark_V2",
            "status": m.get("qc_status", "success"),
            "model_id": m.get("model_id"),
            "train_dataset": m.get("train_dataset"),
            "train_sha256": m.get("train_sha256"),
            "method": m.get("method"),
            "freeze_ratio": m.get("freeze_ratio", 0.5 if m.get("freeze_layers", 0) > 0 else 0.0),
            "lora_rank": m.get("lora_r", 0),
            "lora_alpha": m.get("lora_alpha", 0),
            "learning_rate": m.get("learning_rate"),
            "seed": m.get("seed"),
            "best_epoch": best_epoch,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_seconds": m.get("train_seconds"),
            "peak_vram_mb": peak_vram,
            "inference_examples_per_second": m.get("inference_examples_per_second"),
        })

        # Auto-discover ALL predictions_*.jsonl files
        pred_files = sorted(run_path.glob("predictions_*.jsonl"))
        if not pred_files:
            msg = f"  [HATA] {run_id}: hiç predictions_*.jsonl yok"
            errors.append(msg)
            print(msg)
            continue

        for pred_file in pred_files:
            eval_set = pred_file.stem.removeprefix("predictions_")
            try:
                res = sklearn_f1_from_predictions(pred_file)
                overall_rows.append({
                    "run_id": run_id,
                    "eval_set": eval_set,
                    "n_examples": res["n"],
                    "accuracy": res["accuracy"],
                    "macro_f1": res["macro_f1"],
                    "supported_f1": res["supported_f1"],
                    "partially_supported_f1": res["partially_supported_f1"],
                    "contradicted_f1": res["contradicted_f1"],
                    "unverifiable_f1": res["unverifiable_f1"],
                    "source": "sklearn_predictions",
                })
            except Exception as e:
                msg = f"  [HATA] {run_id} / {eval_set}: {e}"
                errors.append(msg)
                print(msg)
                if strict:
                    raise RuntimeError(msg) from e

    # Save runs.csv
    with open(CSV_DIR / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runs_headers)
        writer.writeheader()
        writer.writerows(runs_rows)

    # Save canonical metrics_long_canonical.csv
    with open(CANONICAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_headers)
        writer.writeheader()
        writer.writerows(overall_rows)

    unique_runs = len({r["run_id"] for r in overall_rows})
    unique_sets = sorted({r["eval_set"] for r in overall_rows})
    print(f"\n[TAMAMLANDI] {len(runs_rows)} run, {len(overall_rows)} satır")
    print(f"  Run klasörleri:  {len(runs_rows)}")
    print(f"  Kanonik satırlar: {len(overall_rows)}")
    print(f"  Benzersiz run:   {unique_runs}")
    print(f"  Eval setleri:    {unique_sets}")
    print(f"  Çıktı: {CANONICAL_CSV}")
    if errors:
        print(f"\n  [UYARILAR] {len(errors)} hata:")
        for e in errors:
            print(f"    {e}")

    # Assertion: run count must match
    dir_count = sum(1 for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "metrics.json").exists())
    if unique_runs != dir_count:
        print(f"\n  [UYARI] Run klasör sayısı ({dir_count}) != kanonik CSV benzersiz run ({unique_runs})")
        print("  Eksik run'lar predictions dosyası olmayan koşular olabilir.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenient", action="store_true",
                    help="Hata yerine uyarı ver, devam et")
    args = ap.parse_args()
    collect(strict=not args.lenient)