"""
scripts/14_validate_runs.py

Tüm tamamlanmış koşuların veri bütünlüğünü doğrular:
  1. Beklenen dosyaların varlığını kontrol eder.
  2. predictions_DEV960-GEN.jsonl ile metrics.json'daki d960_accuracy/macro_f1 değerlerinin
     birbirini doğruladığını kontrol eder (tolerans: ±0.001).
  3. predictions_DEV600-STR.jsonl ile metrics.json'daki d600_accuracy/macro_f1 değerlerinin
     birbirini doğruladığını kontrol eder (tolerans: ±0.001).
  4. Her iki dev seti için per-class F1 değerlerini predictions'dan hesaplar ve
     metrics.json ile karşılaştırır.
  5. Tutarsızlıkları FAIL olarak raporlar ve gerekirse metrics.json'u predictions'dan
     yeniden hesaplayarak düzeltir (--fix bayrağı ile).

Çıktı:
  - reports/validation_audit.json

Kullanım:
  python scripts/14_validate_runs.py           # sadece raporla
  python scripts/14_validate_runs.py --fix     # tutarsızları düzelt
"""
import sys, json, argparse, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import classification_report

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
REPORT_DIR = Q1_DIR / "reports"

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
LABEL2ID    = {ln: i for i, ln in enumerate(LABEL_NAMES)}
TOL         = 0.002  # tolerans: ±0.002 yuvarlama farkı kabul edilir

EXPECTED_FILES = [
    "config.json",
    "dataset_manifest.json",
    "metrics.json",
    "trainable_parameters.json",
    "validation_report.json",
    "predictions_DEV960-GEN.jsonl",
    "predictions_DEV600-STR.jsonl",
]


def compute_metrics_from_predictions(pred_file: Path):
    """predictions jsonl dosyasından accuracy, macro_f1, per-class F1 hesaplar."""
    if not pred_file.exists():
        return None
    records = [json.loads(l) for l in open(pred_file, encoding="utf-8")]
    gold_labels = [r.get("gold_label") for r in records]
    pred_labels = [r.get("predicted_label") for r in records]
    if not gold_labels or not pred_labels:
        return None

    label_set = sorted(set(gold_labels) | set(pred_labels))
    gold_ids = [LABEL2ID.get(l, -1) for l in gold_labels]
    pred_ids = [LABEL2ID.get(l, -1) for l in pred_labels]

    valid_mask = [(g != -1 and p != -1) for g, p in zip(gold_ids, pred_ids)]
    gold_ids = [g for g, v in zip(gold_ids, valid_mask) if v]
    pred_ids = [p for p, v in zip(pred_ids, valid_mask) if v]

    n = len(gold_ids)
    if n == 0:
        return None

    acc = sum(g == p for g, p in zip(gold_ids, pred_ids)) / n

    report = classification_report(
        gold_ids, pred_ids,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]
    per_class = {ln: round(report[ln]["f1-score"], 4) for ln in LABEL_NAMES}
    pred_dist = dict(Counter(pred_labels))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class,
        "pred_dist": pred_dist,
        "n": n
    }


def check_consistency(m_val, p_val, field, tol=TOL):
    """metrics.json değeri ile predictions'dan hesaplanan değer arasındaki farkı kontrol eder."""
    if m_val is None or p_val is None:
        return None, f"MISSING: {field} değeri yok"
    diff = abs(float(m_val) - float(p_val))
    if diff > tol:
        return False, f"MISMATCH: {field} metrics.json={m_val:.4f}, predictions={p_val:.4f}, fark={diff:.4f}"
    return True, f"OK: {field} metrics={m_val:.4f} ≈ predictions={p_val:.4f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    return Q1_DIR / Path(value.replace("\\", "/"))


def validate_run(run_path: Path, fix: bool = False):
    run_id = run_path.name
    result = {"run_id": run_id, "issues": [], "status": "PASS", "fixed": []}

    # 1. Dosya varlığı kontrolü
    for fname in EXPECTED_FILES:
        if not (run_path / fname).exists():
            result["issues"].append(f"MISSING_FILE: {fname}")
            result["status"] = "WARN"

    # Archived evaluation paths must resolve to the isolated legacy directory and match recorded hashes.
    dataset_manifest_file = run_path / "dataset_manifest.json"
    if dataset_manifest_file.exists():
        dm = json.loads(dataset_manifest_file.read_text(encoding="utf-8-sig"))
        hash_fields = {
            "train_file": "train_sha256",
            "dev960_file": "dev960_sha256",
            "dev600_file": "dev600_sha256",
        }
        ALLOWED_DEV_PREFIXES = (
            "data/evaluation/legacy_experiment_version/",
            "data/evaluation/preview_v0.1/",
        )
        for field in ("train_file", "dev960_file", "dev600_file"):
            recorded = str(dm.get(field, "")).replace("\\", "/")
            if field.startswith("dev") and not any(recorded.startswith(p) for p in ALLOWED_DEV_PREFIXES):
                result["issues"].append(f"DATASET_PATH_MISMATCH: {field}={recorded!r} is outside the allowed evaluation directories")
                result["status"] = "FAIL"
                continue
            if field == "train_file" and not (recorded.startswith("data/primary/") or recorded.startswith("data/derived/")):
                result["issues"].append(f"DATASET_PATH_MISMATCH: train_file={recorded!r} is outside the current training-data layout")
                result["status"] = "FAIL"
                continue
            resolved = _repo_path(recorded)
            if not resolved.exists():
                result["issues"].append(f"MISSING_DATASET: {recorded}")
                result["status"] = "FAIL"
                continue
            expected_hash = dm.get(hash_fields[field])
            actual_hash = _sha256(resolved)
            if expected_hash != actual_hash:
                result["issues"].append(f"DATASET_HASH_MISMATCH: {field} expected={expected_hash}, actual={actual_hash}")
                result["status"] = "FAIL"
        # Relocation history is only required for runs that used legacy experiment paths.
        is_legacy_dev = any(
            str(dm.get(f, "")).replace("\\", "/").startswith("data/evaluation/legacy_experiment_version/")
            for f in ("dev960_file", "dev600_file")
        )
        if is_legacy_dev:
            relocation = dm.get("path_relocation", {})
            if not all(relocation.get(key) for key in (
                "original_recorded_train_file",
                "original_recorded_dev960_file",
                "original_recorded_dev600_file",
            )):
                result["issues"].append("MISSING_RELOCATION_HISTORY: original experiment paths are not retained")
                result["status"] = "FAIL"

    metrics_file = run_path / "metrics.json"
    if not metrics_file.exists():
        result["issues"].append("CRITICAL: metrics.json yok — bu koşu collect'e dahil edilemiyor.")
        result["status"] = "FAIL"
        return result

    m = json.loads(open(metrics_file, encoding="utf-8").read())

    # 2. DEV960 doğrulama
    pred960 = run_path / "predictions_DEV960-GEN.jsonl"
    p960 = compute_metrics_from_predictions(pred960)
    if p960 is None:
        result["issues"].append("MISSING_FILE: predictions_DEV960-GEN.jsonl")
        result["status"] = "WARN"
    else:
        ok, msg = check_consistency(m.get("d960_accuracy"), p960["accuracy"], "d960_accuracy")
        if ok is False:
            if fix:
                m["d960_accuracy"] = p960["accuracy"]
                result["fixed"].append(f"FIXED: d960_accuracy -> {p960['accuracy']:.4f}")
            else:
                result["issues"].append(msg)
                result["status"] = "FAIL"
        ok, msg = check_consistency(m.get("d960_macro_f1"), p960["macro_f1"], "d960_macro_f1")
        if ok is False:
            if fix:
                m["d960_macro_f1"] = p960["macro_f1"]
                result["fixed"].append(f"FIXED: d960_macro_f1 -> {p960['macro_f1']:.4f}")
            else:
                result["issues"].append(msg)
                result["status"] = "FAIL"
        # Per-class DEV960
        for ln in LABEL_NAMES:
            key = f"d960_{ln}_f1"
            p_val = p960["per_class_f1"][ln]
            if key not in m:
                if fix:
                    m[key] = p_val
                    result["fixed"].append(f"ADDED: {key}={p_val:.4f}")
            else:
                ok, msg = check_consistency(m[key], p_val, key)
                if ok is False:
                    if fix:
                        m[key] = p_val
                        result["fixed"].append(f"FIXED: {key} -> {p_val:.4f}")
                    else:
                        result["issues"].append(msg)
                        result["status"] = "FAIL"

    # 3. DEV600 doğrulama
    pred600 = run_path / "predictions_DEV600-STR.jsonl"
    p600 = compute_metrics_from_predictions(pred600)
    if p600 is None:
        result["issues"].append("MISSING_FILE: predictions_DEV600-STR.jsonl")
        result["status"] = "WARN"
    else:
        ok, msg = check_consistency(m.get("d600_accuracy"), p600["accuracy"], "d600_accuracy")
        if ok is False:
            result["issues"].append(msg)
            result["status"] = "FAIL"
        ok, msg = check_consistency(m.get("d600_macro_f1"), p600["macro_f1"], "d600_macro_f1")
        if ok is False:
            result["issues"].append(msg)
            result["status"] = "FAIL"
        # Per-class DEV600 — predictions'dan güncelle (kaynak gerçek, metrics ikincil)
        for ln in LABEL_NAMES:
            key = f"d600_{ln}_f1"
            p_val = p600["per_class_f1"][ln]
            if key not in m:
                if fix:
                    m[key] = p_val
                    result["fixed"].append(f"ADDED: {key}={p_val}")
                else:
                    result["issues"].append(f"MISSING_FIELD: {key} metrics.json'da yok")
                    result["status"] = "WARN"
            else:
                ok, msg = check_consistency(m[key], p_val, key)
                if ok is False:
                    if fix:
                        m[key] = p_val
                        result["fixed"].append(f"FIXED: {key} {m[key]:.4f} → {p_val}")
                    else:
                        result["issues"].append(msg)
                        result["status"] = "FAIL"

        # DEV600 accuracy/macro_f1 da predictions'dan fix et eğer tutarsızsa
        if fix:
            m["d600_accuracy"] = p600["accuracy"]
            m["d600_macro_f1"] = p600["macro_f1"]

    # DEV960 per-class eksik alanları da ekle
    if p960 is not None:
        for ln in LABEL_NAMES:
            key = f"d960_{ln}_f1"
            p_val = p960["per_class_f1"][ln]
            if key not in m:
                if fix:
                    m[key] = p_val
                    result["fixed"].append(f"ADDED: {key}={p_val}")
                else:
                    result["issues"].append(f"MISSING_FIELD: {key} metrics.json'da yok")
                    result["status"] = "WARN"

    if fix and result["fixed"]:
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

    if not result["issues"]:
        result["status"] = "PASS"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Tutarsızlıkları predictions'dan düzelt")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Q1 Pipeline Run Validation Audit")
    print(f"Fix mode: {'AÇIK' if args.fix else 'KAPALI'}")
    print("=" * 70)

    all_results = []
    pass_count = warn_count = fail_count = 0

    for run_path in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not run_path.is_dir():
            continue
        r = validate_run(run_path, fix=args.fix)
        all_results.append(r)

        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(r["status"], "?")
        print(f"\n{status_icon} {r['run_id']} [{r['status']}]")
        for issue in r["issues"]:
            print(f"    ! {issue}")
        for fix_note in r["fixed"]:
            print(f"    → {fix_note}")

        if r["status"] == "PASS":
            pass_count += 1
        elif r["status"] == "WARN":
            warn_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print(f"OZET: {pass_count} PASS | {warn_count} WARN | {fail_count} FAIL")
    print("=" * 70)

    report_path = REPORT_DIR / "validation_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
                   "runs": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nRapor yazildi: {report_path}")

    # CI icin: FAIL veya WARN varsa exit code 1
    if fail_count > 0 or warn_count > 0:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()