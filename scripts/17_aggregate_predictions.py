"""
scripts/17_aggregate_predictions.py

Protokol Section 15 — Predictions klasörünü doldurur.

Tüm başarılı koşuların predictions dosyalarını results/predictions/ altında
model+devset bazında birleştirip indekser.

Çıktılar (results/predictions/):
  predictions_index.json          — Hangi predictions hangi run'dan geldiği
  ensemble_voting_DEV960.jsonl    — Başarılı modellerin majority vote topluluğu
  ensemble_voting_DEV600.jsonl    — Başarılı modellerin majority vote topluluğu
  error_analysis_DEV960.jsonl     — Tüm başarılı modellerin yanlış tahmin ettiği örnekler
  error_analysis_DEV600.jsonl     — Tüm başarılı modellerin yanlış tahmin ettiği örnekler
"""
import sys, json, csv
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
PRED_DIR  = Q1_DIR / "results/predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]

SUCCESSFUL_RUNS = [
    "B1_M04_ELECTRA_ALLLINEAR_LORA16_S42",
    "B1_M03_CONVBERT_ALLLINEAR_LORA16_S42",
    "B1_M01_BERT32K_LORA32_S42",
    "B1_M01_BERT32K_FULL_FT_S42",
    "B1_M01_BERT32K_ALLLINEAR_LORA16_S42",
    "B1_M02_BERT128K_ALLLINEAR_LORA16_S42",
    "B1_M01_BERT32K_ALLLINEAR_LORA8_S42",
]


def load_predictions(run_id: str, suffix: str):
    pred_file = RUNS_DIR / run_id / f"predictions_{suffix}.jsonl"
    if not pred_file.exists():
        return {}
    preds = {}
    for line in open(pred_file, encoding="utf-8"):
        r = json.loads(line)
        ex_id = r.get("example_id")
        if ex_id:
            preds[ex_id] = r
    return preds


def majority_vote(votes: list):
    """En çok oy alan etiketi döndürür. Beraberlikte alfabetik ilk seçilir."""
    c = Counter(votes)
    return c.most_common(1)[0][0]


def process_dev_set(suffix: str, label: str):
    # Tüm modellerin predictions'larını yükle
    all_preds = {}  # run_id → {example_id: pred_record}
    for run_id in SUCCESSFUL_RUNS:
        preds = load_predictions(run_id, suffix)
        if preds:
            all_preds[run_id] = preds

    if not all_preds:
        print(f"  [UYARI] {suffix} için predictions bulunamadı.")
        return 0, 0

    # Ortak example_id seti
    all_ids = set.intersection(*[set(p.keys()) for p in all_preds.values()])
    print(f"  {label}: {len(all_ids)} ortak örnek, {len(all_preds)} model")

    # Ensemble majority voting
    ensemble_records = []
    for ex_id in sorted(all_ids):
        votes      = [all_preds[run_id][ex_id]["predicted_label"] for run_id in all_preds]
        gold       = all_preds[list(all_preds.keys())[0]][ex_id]["gold_label"]
        majority   = majority_vote(votes)
        vote_dist  = dict(Counter(votes))
        agreement  = round(max(vote_dist.values()) / len(votes), 4)
        ensemble_records.append({
            "example_id":       ex_id,
            "domain":           all_preds[list(all_preds.keys())[0]][ex_id].get("domain"),
            "gold_label":       gold,
            "ensemble_pred":    majority,
            "correct":          majority == gold,
            "agreement_ratio":  agreement,
            "vote_distribution": vote_dist,
            "per_model_preds":  {run_id: all_preds[run_id][ex_id]["predicted_label"] for run_id in all_preds},
        })

    ensemble_acc = sum(1 for r in ensemble_records if r["correct"]) / len(ensemble_records)
    print(f"  {label} ensemble accuracy: {ensemble_acc:.4f}")

    ens_file = PRED_DIR / f"ensemble_voting_{suffix.replace('.jsonl', '')}.jsonl"
    with open(ens_file, "w", encoding="utf-8") as f:
        for r in ensemble_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [OK] {ens_file.name}")

    # Error analysis — tüm modellerin yanlış yaptığı örnekler
    error_records = [r for r in ensemble_records if not r["correct"]]
    all_wrong = [r for r in error_records if all(
        all_preds[run_id][r["example_id"]]["predicted_label"] != r["gold_label"]
        for run_id in all_preds if r["example_id"] in all_preds[run_id]
    )]
    print(f"  {label} tüm modellerin yanlış yaptığı: {len(all_wrong)}/{len(all_ids)} örnek")

    err_file = PRED_DIR / f"error_analysis_{suffix.replace('.jsonl', '')}.jsonl"
    with open(err_file, "w", encoding="utf-8") as f:
        for r in all_wrong:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [OK] {err_file.name}")

    return len(ensemble_records), len(all_wrong)


def main():
    print("=" * 60)
    print("Q1 Predictions Aggregator")
    print("=" * 60)

    # Predictions index — every run and every discovered evaluation set.
    index = {}
    total_files = 0
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for pred_file in sorted(run_dir.glob("predictions_*.jsonl")):
            dev_set = pred_file.stem.removeprefix("predictions_")
            n = sum(1 for _ in open(pred_file, encoding="utf-8"))
            index.setdefault(run_id, {})[dev_set] = {
                "file": pred_file.relative_to(PROJECT_ROOT).as_posix(),
                "n": n,
            }
            total_files += 1

    with open(PRED_DIR / "predictions_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"  [OK] predictions_index.json ({len(index)} run, {total_files} files)")

    # DEV960 ensemble
    n960, err960 = process_dev_set("DEV960-GEN", "DEV960")

    # DEV600 ensemble
    n600, err600 = process_dev_set("DEV600-STR", "DEV600")

    print(f"\n[BAŞARILI] {len(list(PRED_DIR.iterdir()))} dosya {PRED_DIR} klasörüne yazıldı.")


if __name__ == "__main__":
    main()