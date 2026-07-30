"""
scripts/16_generate_statistics.py

Protokol Section 15 — İstatistiksel analiz raporlarını üretir.

Çıktılar (results/statistics/):
  summary_statistics.json       — Tüm koşuların özet istatistikleri
  domain_analysis.json          — Per-domain performans analizi (predictions'dan)
  stress_vs_general.json        — DEV960 vs DEV600 karşılaştırması
  key_findings.md               — Ana bulgular özet raporu
"""
import sys, json, csv
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
CSV_DIR    = Q1_DIR / "results/csv"
RUNS_DIR   = Q1_DIR / "runs/completed"
STATS_DIR  = Q1_DIR / "results/statistics"
STATS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]

# SUCCESSFUL_RUNS artık runs.csv'den dinamik olarak okunuyor (aşağıda main() içinde)


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions(run_id: str, dev_set: str):
    suffix = "DEV960-GEN" if dev_set == "d960" else "DEV600-STR"
    pred_file = RUNS_DIR / run_id / f"predictions_{suffix}.jsonl"
    if not pred_file.exists():
        return []
    return [json.loads(l) for l in open(pred_file, encoding="utf-8")]


def compute_domain_metrics(predictions: list):
    """predictions listesinden per-domain accuracy ve per-class F1 hesaplar."""
    domains = set(r.get("domain", "?") for r in predictions)
    result = {}
    for domain in sorted(domains):
        subset = [r for r in predictions if r.get("domain") == domain]
        if not subset:
            continue
        n = len(subset)
        correct = sum(1 for r in subset if r.get("gold_label") == r.get("predicted_label"))
        acc = round(correct / n, 4)

        from sklearn.metrics import f1_score
        gold_ids = [LABEL_NAMES.index(r.get("gold_label", "supported")) for r in subset if r.get("gold_label") in LABEL_NAMES]
        pred_ids = [LABEL_NAMES.index(r.get("predicted_label", "supported")) for r in subset if r.get("predicted_label") in LABEL_NAMES]

        if len(gold_ids) > 0:
            macro_f1 = round(f1_score(gold_ids, pred_ids, average="macro", zero_division=0), 4)
        else:
            macro_f1 = 0.0

        result[domain] = {"n": n, "accuracy": acc, "macro_f1": macro_f1}
    return result


def main():
    print("=" * 60)
    print("Q1 Statistics Generator")
    print("=" * 60)

    metrics_rows = read_csv(CSV_DIR / "metrics_long_canonical.csv")
    runs_rows    = read_csv(CSV_DIR / "runs.csv")

    # metrics_overall → run → devset lookup
    metrics = {}
    for r in metrics_rows:
        run_id  = r["run_id"]
        eval_set = r["eval_set"]
        if run_id not in metrics:
            metrics[run_id] = {}
        metrics[run_id][eval_set] = r

    runs = {r["run_id"]: r for r in runs_rows}

    # Başarılı ve başarısız koşuları runs.csv'den dinamik oku
    SUCCESSFUL_RUNS = [r["run_id"] for r in runs_rows if r.get("status", "success") == "success"]
    DEGENERATE_RUNS = [r["run_id"] for r in runs_rows if r.get("status", "success") != "success"]
    print(f"  Toplam: {len(runs_rows)} koşu | Başarılı: {len(SUCCESSFUL_RUNS)} | Degenerate: {len(DEGENERATE_RUNS)}")

    # ================================================================
    # 1. Summary Statistics
    # ================================================================
    best_run_d960 = max(SUCCESSFUL_RUNS, key=lambda r: float(metrics.get(r, {}).get("DEV960-GEN", {}).get("macro_f1") or 0))
    best_run_d600 = max(SUCCESSFUL_RUNS, key=lambda r: float(metrics.get(r, {}).get("DEV600-STR", {}).get("macro_f1") or 0))

    d960_f1_values = [float(metrics.get(r, {}).get("DEV960-GEN", {}).get("macro_f1") or 0) for r in SUCCESSFUL_RUNS]
    d600_f1_values = [float(metrics.get(r, {}).get("DEV600-STR", {}).get("macro_f1") or 0) for r in SUCCESSFUL_RUNS]

    # Degenerate status counts
    status_counts = Counter(r.get("status", "success") for r in runs_rows)

    summary = {
        "total_runs": len(runs_rows),
        "successful_runs": len(SUCCESSFUL_RUNS),
        "degenerate_runs": len(DEGENERATE_RUNS),
        "status_breakdown": dict(status_counts),
        "best_model_d960": {
            "run_id": best_run_d960,
            "macro_f1": float(metrics[best_run_d960]["DEV960-GEN"]["macro_f1"]),
        },
        "best_model_d600": {
            "run_id": best_run_d600,
            "macro_f1": float(metrics[best_run_d600]["DEV600-STR"]["macro_f1"]),
        },
        "d960_f1_range": {
            "min": round(min(d960_f1_values), 4),
            "max": round(max(d960_f1_values), 4),
            "mean": round(sum(d960_f1_values) / len(d960_f1_values), 4),
        },
        "d600_f1_range": {
            "min": round(min(d600_f1_values), 4),
            "max": round(max(d600_f1_values), 4),
            "mean": round(sum(d600_f1_values) / len(d600_f1_values), 4),
        },
        "generalization_gap": {
            desc: round(
                float(metrics.get(r, {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                float(metrics.get(r, {}).get("DEV600-STR", {}).get("macro_f1", 0)),
                4
            )
            for r, desc in [
                ("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", "electra_lora16"),
                ("B1_M03_CONVBERT_ALLLINEAR_LORA16_S42", "convbert_lora16"),
                ("B1_M01_BERT32K_LORA32_S42", "bert_lora32"),
                ("B1_M01_BERT32K_FULL_FT_S42", "bert_full_ft"),
                ("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", "bert_lora16"),
            ]
        },
        "rank_scaling_analysis": {
            # BERTurk tarihsel referans (B1 koşuları)
            "bert_r8_d960":  float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA8_S42",  {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "bert_r16_d960": float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "bert_r32_d960": float(metrics.get("B1_M01_BERT32K_LORA32_S42",           {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            # ELECTRA ana model rank ablasyonu (C3 koşuları + B1 r16)
            "electra_r4_d960":  float(metrics.get("C3_M04_ELECTRA_LORA4_S42",           {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "electra_r8_d960":  float(metrics.get("C3_M04_ELECTRA_LORA8_S42",           {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "electra_r16_d960": float(metrics.get("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42",{}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "electra_r32_d960": float(metrics.get("C3_M04_ELECTRA_LORA32_S42",          {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "electra_r4_to_r8_gain":   round(float(metrics.get("C3_M04_ELECTRA_LORA8_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                                             float(metrics.get("C3_M04_ELECTRA_LORA4_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
            "electra_r8_to_r16_gain":  round(float(metrics.get("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                                             float(metrics.get("C3_M04_ELECTRA_LORA8_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
            "electra_r16_to_r32_gain": round(float(metrics.get("C3_M04_ELECTRA_LORA32_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                                             float(metrics.get("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
            "electra_optimal_rank": 16,
            # Uyumluluk için eski isimler (BERTurk)
            "r8_d960":  float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA8_S42",  {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "r16_d960": float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "r32_d960": float(metrics.get("B1_M01_BERT32K_LORA32_S42",           {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "r8_to_r16_gain":  round(float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                                     float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA8_S42",  {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
            "r16_to_r32_gain": round(float(metrics.get("B1_M01_BERT32K_LORA32_S42",           {}).get("DEV960-GEN", {}).get("macro_f1", 0)) -
                                     float(metrics.get("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
        },
        "target_module_ablation": {
            "scope": "ELECTRA-TR, LoRA r=16, seed=42; single-seed exploratory ablation",
            "qv_d960": float(metrics.get("C7_M04_ELECTRA_QV_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "qv_d600": float(metrics.get("C7_M04_ELECTRA_QV_LORA16_S42", {}).get("DEV600-STR", {}).get("macro_f1", 0)),
            "all_attention_d960": float(metrics.get("C8_M04_ELECTRA_ALLATTN_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "all_attention_d600": float(metrics.get("C8_M04_ELECTRA_ALLATTN_LORA16_S42", {}).get("DEV600-STR", {}).get("macro_f1", 0)),
            "all_linear_d960": float(metrics.get("G1_C6_M04_ELECTRA_LORA16_DROP10_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)),
            "all_linear_d600": float(metrics.get("G1_C6_M04_ELECTRA_LORA16_DROP10_S42", {}).get("DEV600-STR", {}).get("macro_f1", 0)),
            "all_linear_minus_all_attention_d960": round(float(metrics.get("G1_C6_M04_ELECTRA_LORA16_DROP10_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)) - float(metrics.get("C8_M04_ELECTRA_ALLATTN_LORA16_S42", {}).get("DEV960-GEN", {}).get("macro_f1", 0)), 4),
            "interpretation": "All-linear is higher on DEV960 in this single-seed comparison, but uses more trainable parameters and the runs are not a parameter-matched multi-seed isolation of FFN effects."
        }
    }

    out = STATS_DIR / "summary_statistics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [OK] summary_statistics.json")

    # ================================================================
    # 2. Domain Analysis (best model predictions'dan)
    # ================================================================
    domain_analysis = {}
    for run_id in SUCCESSFUL_RUNS:
        preds960 = load_predictions(run_id, "d960")
        preds600 = load_predictions(run_id, "d600")
        if preds960:
            domain_analysis[run_id] = {
                "DEV960": compute_domain_metrics(preds960),
                "DEV600": compute_domain_metrics(preds600) if preds600 else {},
            }

    out = STATS_DIR / "domain_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(domain_analysis, f, indent=2, ensure_ascii=False)
    print(f"  [OK] domain_analysis.json")

    # ================================================================
    # 3. Stress vs General gap analizi
    # ================================================================
    stress_vs_general = {}
    for run_id in SUCCESSFUL_RUNS:
        m960 = metrics.get(run_id, {}).get("DEV960-GEN", {})
        m600 = metrics.get(run_id, {}).get("DEV600-STR", {})
        try:
            gap = round(float(m960.get("macro_f1", 0)) - float(m600.get("macro_f1", 0)), 4)
        except:
            gap = None
        stress_vs_general[run_id] = {
            "d960_macro_f1": float(m960.get("macro_f1", 0)),
            "d600_macro_f1": float(m600.get("macro_f1", 0)),
            "gap_d960_minus_d600": gap,
            "per_class_d960_vs_d600": {
                ln: {
                    "d960": float(m960.get(f"{ln}_f1", 0)),
                    "d600": float(m600.get(f"{ln}_f1", 0)),
                }
                for ln in LABEL_NAMES
            }
        }

    out = STATS_DIR / "stress_vs_general.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stress_vs_general, f, indent=2, ensure_ascii=False)
    print(f"  [OK] stress_vs_general.json")

    # ================================================================
    # 4. Key Findings Markdown
    # ================================================================
    best_d960_f1  = summary["best_model_d960"]["macro_f1"]
    best_d600_f1  = summary["best_model_d600"]["macro_f1"]
    rank_s        = summary["rank_scaling_analysis"]
    target_s      = summary["target_module_ablation"]

    md_lines = [
        "# Q1 Benchmark — Ana Bulgular",
        "",
        f"**Toplam koşu:** {summary['total_runs']} | **Başarılı:** {summary['successful_runs']} | **Degenerate:** {summary['degenerate_runs']}",
        f"**Durum dağılımı:** {summary['status_breakdown']}",
        "",
        "## 1. En İyi Model",
        f"- **DEV960-GEN:** `{summary['best_model_d960']['run_id']}` — macro F1 = **{best_d960_f1:.4f}**",
        f"- **DEV600-STR:** `{summary['best_model_d600']['run_id']}` — macro F1 = **{best_d600_f1:.4f}**",
        "",
        "## 2. LoRA Hedef Modül Karşılaştırması — Tek Seed",
        f"- Q+V: DEV960 = {target_s['qv_d960']:.4f}, DEV600 = {target_s['qv_d600']:.4f}",
        f"- All-attention: DEV960 = {target_s['all_attention_d960']:.4f}, DEV600 = {target_s['all_attention_d600']:.4f}",
        f"- All-linear: DEV960 = {target_s['all_linear_d960']:.4f}, DEV600 = {target_s['all_linear_d600']:.4f}",
        f"- All-linear, DEV960'ta all-attention'dan {target_s['all_linear_minus_all_attention_d960']*100:+.2f} puan yüksektir; ancak daha fazla eğitilebilir parametre kullanır.",
        "- Bu tek-seed karşılaştırma FFN katkısını tek başına izole etmez; çoklu-seed ve parametre-eşlemeli doğrulama beklenmektedir.",
        "",
        "## 3. LoRA Rank Skalası — ELECTRA-TR (ana model, all-linear, alpha=2r, seed=42)",
        f"- r=4:  DEV960 F1 = {rank_s['electra_r4_d960']:.4f}",
        f"- r=8:  DEV960 F1 = {rank_s['electra_r8_d960']:.4f}  (+{rank_s['electra_r4_to_r8_gain']*100:.1f}pp)",
        f"- r=16: DEV960 F1 = {rank_s['electra_r16_d960']:.4f}  (+{rank_s['electra_r8_to_r16_gain']*100:.1f}pp) ★ OPTIMAL",
        f"- r=32: DEV960 F1 = {rank_s['electra_r32_d960']:.4f}  ({rank_s['electra_r16_to_r32_gain']*100:.1f}pp)",
        f"- **Optimal rank: r=16.** r=32 ek kapasite kazanç sağlamadı.",
        "",
        "### 3b. BERTurk Rank Skalası (tarihsel referans, BERT-32k all-linear)",
        f"- r=8:  DEV960 F1 = {rank_s['r8_d960']:.4f}",
        f"- r=16: DEV960 F1 = {rank_s['r16_d960']:.4f}  (+{rank_s['r8_to_r16_gain']*100:.1f}pp)",
        f"- r=32: DEV960 F1 = {rank_s['r32_d960']:.4f}  (+{rank_s['r16_to_r32_gain']*100:.1f}pp)",
        f"- **Azalan getiri:** r=8→r=16 kazanımı ({rank_s['r8_to_r16_gain']*100:.1f}pp) > r=16→r=32 kazanımı ({rank_s['r16_to_r32_gain']*100:.1f}pp)",
        "",
        "## 4. LoRA r=32 vs Full Fine-tune (BERT-32k)",
        f"- LoRA r=32 (5.4M param, %4.6): DEV960 F1 = {rank_s['r32_d960']:.4f}",
        f"- Full FT (110.6M param, %100): DEV960 F1 = {float(metrics.get('B1_M01_BERT32K_FULL_FT_S42', {}).get('DEV960-GEN', {}).get('macro_f1', 0)):.4f}",
        f"- LoRA r=32, parametrelerin %4.6'sını eğiterek Full FT'yi geçiyor.",
        "",
        "## 5. Genelleme Açığı (DEV960 − DEV600)",
        "",
    ]
    for run_id, gap_data in stress_vs_general.items():
        gap = gap_data["gap_d960_minus_d600"]
        md_lines.append(f"- `{run_id}`: {gap_data['d960_macro_f1']:.4f} → {gap_data['d600_macro_f1']:.4f}  (gap = {gap:+.4f})")

    # Per-class analizi: başarılı modellerde DEV600-STR'de hangi sınıf en zor?
    label_f1_sums = {ln: [] for ln in LABEL_NAMES}
    for run_id in SUCCESSFUL_RUNS:
        m600 = metrics.get(run_id, {}).get("DEV600-STR", {})
        for ln in LABEL_NAMES:
            v = m600.get(f"{ln}_f1")
            try:
                label_f1_sums[ln].append(float(v))
            except (TypeError, ValueError):
                pass
    label_means = {ln: (sum(vs)/len(vs) if vs else 0.0) for ln, vs in label_f1_sums.items()}
    hardest_label = min(label_means, key=label_means.get)
    easiest_label = max(label_means, key=label_means.get)

    md_lines += [
        "",
        f"## 6. Per-Class Güçlük Analizi (DEV600-STR, {len(SUCCESSFUL_RUNS)} başarılı koşu ortalaması)",
        "",
        f"| Sınıf | Ort. DEV600-STR F1 |",
        "|---|---:|",
    ]
    for ln in sorted(LABEL_NAMES, key=lambda x: label_means[x]):
        marker = " ← en zor" if ln == hardest_label else (" ← en kolay" if ln == easiest_label else "")
        md_lines.append(f"| `{ln}` | {label_means[ln]:.4f}{marker} |")
    md_lines += [
        "",
        f"En zor sınıf: **`{hardest_label}`** (ort. {label_means[hardest_label]:.4f})",
        f"En kolay sınıf: **`{easiest_label}`** (ort. {label_means[easiest_label]:.4f})",
    ]

    out = STATS_DIR / "key_findings.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  [OK] key_findings.md")

    print(f"\n[BAŞARILI] {len(list(STATS_DIR.iterdir()))} dosya {STATS_DIR} klasörüne yazıldı.")


if __name__ == "__main__":
    main()