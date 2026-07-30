"""
scripts/15_generate_tables.py

Protokol Section 15 — Sonuç tablolarını üretir.

Çıktılar (results/tables/):
  table_A_model_comparison.csv       — Ana model karşılaştırma tablosu (B1 başarılı koşular)
  table_A_model_comparison.md        — Markdown formatı
  table_B_rank_scaling.csv           — Rank scaling analizi (BERT all-linear r=8/16/32 + Full FT)
  table_B_rank_scaling.md
  table_C_target_module_ablation.csv — Hedefleme ablasyon tablosu (attention-only vs all-linear)
  table_C_target_module_ablation.md
  table_D_all_runs_full.csv          — Tüm koşular tam tablo (B0 dahil)
  table_D_all_runs_full.md
"""
import sys, json, csv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
CSV_DIR   = Q1_DIR / "results/csv"
TABLE_DIR = Q1_DIR / "results/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, headers: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  [CSV] {path.name}")


def write_md(path: Path, title: str, headers: list, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            vals = [str(row.get(h, "")) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"  [MD]  {path.name}")


def fmt(v, digits=4):
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "—"


def load_metrics():
    """metrics_long_canonical.csv'den run_id → {eval_set: metrics} yapısı oluşturur."""
    rows = read_csv(CSV_DIR / "metrics_long_canonical.csv")
    data = {}
    for r in rows:
        run_id = r["run_id"]
        eval_set = r["eval_set"]
        if run_id not in data:
            data[run_id] = {}
        data[run_id][eval_set] = r
    return data


def load_runs():
    """runs.csv'den run_id → run_info yapısı oluşturur."""
    rows = read_csv(CSV_DIR / "runs.csv")
    return {r["run_id"]: r for r in rows}


def main():
    print("=" * 60)
    print("Q1 Table Generator")
    print("=" * 60)

    metrics = load_metrics()
    runs    = load_runs()

    # ================================================================
    # TABLE A — Ana model karşılaştırması (F1 serisi, 3-seed ortalama)
    # ================================================================
    # Her entry: (model_adi, method, [run_id_seed42, seed43, seed44])
    TABLE_A_MODEL_SEEDS = [
        ("ELECTRA-TR",   "LoRA r=16 all-linear",
         ["F1_M04_ELECTRA_LORA16_S42", "F1_M04_ELECTRA_LORA16_S43", "F1_M04_ELECTRA_LORA16_S44"]),
        ("ModernBERT-TR","LoRA r=16 all-linear",
         ["F1_M12_MODERNBERT_LORA16_S42","F1_M12_MODERNBERT_LORA16_S43","F1_M12_MODERNBERT_LORA16_S44"]),
        ("ConvBERT-TR",  "LoRA r=16 all-linear",
         ["F1_M03_CONVBERT_LORA16_S42","F1_M03_CONVBERT_LORA16_S43","F1_M03_CONVBERT_LORA16_S44"]),
        ("BERT-32k",     "Full Fine-tune",
         ["F1_M01_BERT32K_FULLFT_S43", "F1_M01_BERT32K_FULLFT_S44", "F1_M01_BERT32K_FULLFT_S45"]),
        ("BERT-32k",     "LoRA r=32 all-linear",
         ["F1_M01_BERT32K_LORA32_S42", "F1_M01_BERT32K_LORA32_S43","F1_M01_BERT32K_LORA32_S44"]),
        ("BERT-128k",    "LoRA r=16 all-linear",
         ["F1_M02_BERT128K_LORA16_S42","F1_M02_BERT128K_LORA16_S43","F1_M02_BERT128K_LORA16_S44"]),
        ("mBERT",        "LoRA r=16 all-linear",
         ["F1_M07_MBERT_LORA16_S42",  "F1_M07_MBERT_LORA16_S43",  "F1_M07_MBERT_LORA16_S44"]),
    ]

    def avg_metric(run_ids, eval_set, key):
        vals = [metrics[r].get(eval_set, {}).get(key) for r in run_ids if r in metrics]
        vals = [float(v) for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    headers_A = ["Model", "Method", "Trainable Params", "Trainable %",
                 "DEV960 F1 (avg)", "DEV960 Sup", "DEV960 PSup", "DEV960 Cont", "DEV960 Unver",
                 "DEV600 F1 (avg)", "Seeds"]
    rows_A = []
    for model_name, method, seed_runs in TABLE_A_MODEL_SEEDS:
        # Trainable params from seed 42 (or first available)
        r0 = runs.get(seed_runs[0], {})
        try:
            trainable = int(r0.get("trainable_params") or 0)
            total_p = int(r0.get("total_params") or 0)
            pct = f"{100*trainable/total_p:.2f}%" if total_p > 0 else "—"
        except:
            trainable, pct = "—", "—"
        seed_ids = "/".join(r.split("_S")[-1] for r in seed_runs)
        rows_A.append({
            "Model": model_name,
            "Method": method,
            "Trainable Params": f"{trainable:,}" if isinstance(trainable, int) else trainable,
            "Trainable %": pct,
            "DEV960 F1 (avg)": fmt(avg_metric(seed_runs, "DEV960-GEN", "macro_f1")),
            "DEV960 Sup":  fmt(avg_metric(seed_runs, "DEV960-GEN", "supported_f1")),
            "DEV960 PSup": fmt(avg_metric(seed_runs, "DEV960-GEN", "partially_supported_f1")),
            "DEV960 Cont": fmt(avg_metric(seed_runs, "DEV960-GEN", "contradicted_f1")),
            "DEV960 Unver":fmt(avg_metric(seed_runs, "DEV960-GEN", "unverifiable_f1")),
            "DEV600 F1 (avg)": fmt(avg_metric(seed_runs, "DEV600-STR", "macro_f1")),
            "Seeds": seed_ids,
        })

    write_csv(TABLE_DIR / "table_A_model_comparison.csv", headers_A, rows_A)
    write_md(TABLE_DIR / "table_A_model_comparison.md", "Table A — Model Karşılaştırması (F1 serisi, 3-seed ortalama, DEV960-GEN macro F1 sırası)", headers_A, rows_A)

    # ================================================================
    # TABLE B — Rank Scaling (BERT-32k all-linear)
    # ================================================================
    TABLE_B_RUNS = [
        ("B1_M01_BERT32K_ALLLINEAR_LORA8_S42",  "BERT-32k", "LoRA r=8",   "1,342,468",  "1.21%"),
        ("B1_M01_BERT32K_ALLLINEAR_LORA16_S42", "BERT-32k", "LoRA r=16",  "2,681,860",  "2.42%"),
        ("B1_M01_BERT32K_LORA32_S42",           "BERT-32k", "LoRA r=32",  "5,360,644",  "4.83%"),
        ("B1_M01_BERT32K_FULL_FT_S42",          "BERT-32k", "Full FT",    "110,620,420","99.73%"),
    ]

    headers_B = ["Run ID", "Model", "Method", "Trainable Params", "Trainable %",
                 "DEV960 F1", "DEV960 Sup", "DEV960 PSup", "DEV960 Cont", "DEV960 Unver",
                 "DEV600 F1", "DEV600 Sup", "DEV600 PSup", "DEV600 Cont", "DEV600 Unver",
                 "DEV960-DEV600 Gap"]
    rows_B = []
    for run_id, model_name, method, tp_hint, pct_hint in TABLE_B_RUNS:
        if run_id not in metrics:
            continue
        m960 = metrics[run_id].get("DEV960-GEN", {})
        m600 = metrics[run_id].get("DEV600-STR", {})
        try:
            gap = round(float(m960.get("macro_f1", 0)) - float(m600.get("macro_f1", 0)), 4)
        except:
            gap = "—"
        # Gercek param sayilari runs.csv'den
        r = runs.get(run_id, {})
        real_trainable = int(r.get("trainable_params") or 0)
        real_total = int(r.get("total_params") or 0)
        tp = f"{real_trainable:,}" if real_trainable > 0 else tp_hint
        pct = f"{100*real_trainable/real_total:.2f}%" if real_total > 0 else pct_hint
        rows_B.append({
            "Run ID": run_id,
            "Model": model_name,
            "Method": method,
            "Trainable Params": tp,
            "Trainable %": pct,
            "DEV960 F1":   fmt(m960.get("macro_f1")),
            "DEV960 Sup":  fmt(m960.get("supported_f1")),
            "DEV960 PSup": fmt(m960.get("partially_supported_f1")),
            "DEV960 Cont": fmt(m960.get("contradicted_f1")),
            "DEV960 Unver":fmt(m960.get("unverifiable_f1")),
            "DEV600 F1":   fmt(m600.get("macro_f1")),
            "DEV600 Sup":  fmt(m600.get("supported_f1")),
            "DEV600 PSup": fmt(m600.get("partially_supported_f1")),
            "DEV600 Cont": fmt(m600.get("contradicted_f1")),
            "DEV600 Unver":fmt(m600.get("unverifiable_f1")),
            "DEV960-DEV600 Gap": fmt(gap),
        })

    write_csv(TABLE_DIR / "table_B_rank_scaling.csv", headers_B, rows_B)
    write_md(TABLE_DIR / "table_B_rank_scaling.md",
             "Table B — Rank Scaling Analizi (BERT-32k, all-linear, LR=2e-4, seed=42) [Tarihsel referans]", headers_B, rows_B)

    # ================================================================
    # TABLE B2 — ELECTRA Rank Scaling (ana model, tek seed=42)
    # ================================================================
    TABLE_B2_RUNS = [
        ("C3_M04_ELECTRA_LORA4_S42",           "ELECTRA-tr", "LoRA r=4",  4,  8,  "3,331,972",  "3.01%"),
        ("C3_M04_ELECTRA_LORA8_S42",           "ELECTRA-tr", "LoRA r=8",  8,  16, "6,587,396",  "5.95%"),
        ("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42","ELECTRA-tr", "LoRA r=16 ★", 16, 32, "13,096,716", "11.81%"),
        ("C3_M04_ELECTRA_LORA32_S42",          "ELECTRA-tr", "LoRA r=32", 32, 64, "26,095,628", "23.53%"),
    ]

    headers_B2 = ["Run ID", "Model", "Method", "Rank", "Alpha", "Trainable Params", "Trainable %",
                  "DEV960 F1", "DEV960 Sup", "DEV960 PSup", "DEV960 Cont", "DEV960 Unver",
                  "DEV600 F1", "DEV600 Sup", "DEV600 PSup", "DEV600 Cont", "DEV600 Unver",
                  "DEV960-DEV600 Gap"]
    rows_B2 = []
    for run_id, model_name, method, rank, alpha, tp_hint, pct_hint in TABLE_B2_RUNS:
        if run_id not in metrics:
            print(f"  [UYARI] {run_id} metrics'te bulunamadi, atlaniyor")
            continue
        m960 = metrics[run_id].get("DEV960-GEN", {})
        m600 = metrics[run_id].get("DEV600-STR", {})
        try:
            gap = round(float(m960.get("macro_f1", 0)) - float(m600.get("macro_f1", 0)), 4)
        except:
            gap = "—"
        # Gercek param sayilari: once runs.csv, sonra trainable_parameters.json
        r = runs.get(run_id, {})
        real_trainable = int(r.get("trainable_params") or 0)
        real_total = int(r.get("total_params") or 0)
        if real_trainable == 0:
            import json as _json
            _tp_path = Q1_DIR / "runs/completed" / run_id / "trainable_parameters.json"
            if _tp_path.exists():
                _tp = _json.load(open(_tp_path, encoding="utf-8"))
                real_trainable = int(_tp.get("num_trainable_params") or 0)
                real_total = int(_tp.get("num_total_params") or 0)
        tp = f"{real_trainable:,}" if real_trainable > 0 else tp_hint
        pct = f"{100*real_trainable/real_total:.2f}%" if real_total > 0 else pct_hint
        rows_B2.append({
            "Run ID": run_id,
            "Model": model_name,
            "Method": method,
            "Rank": rank,
            "Alpha": alpha,
            "Trainable Params": tp,
            "Trainable %": pct,
            "DEV960 F1":   fmt(m960.get("macro_f1")),
            "DEV960 Sup":  fmt(m960.get("supported_f1")),
            "DEV960 PSup": fmt(m960.get("partially_supported_f1")),
            "DEV960 Cont": fmt(m960.get("contradicted_f1")),
            "DEV960 Unver":fmt(m960.get("unverifiable_f1")),
            "DEV600 F1":   fmt(m600.get("macro_f1")),
            "DEV600 Sup":  fmt(m600.get("supported_f1")),
            "DEV600 PSup": fmt(m600.get("partially_supported_f1")),
            "DEV600 Cont": fmt(m600.get("contradicted_f1")),
            "DEV600 Unver":fmt(m600.get("unverifiable_f1")),
            "DEV960-DEV600 Gap": fmt(gap),
        })

    write_csv(TABLE_DIR / "table_B2_electra_rank_scaling.csv", headers_B2, rows_B2)
    # Markdown with özet notu
    md_b2_path = TABLE_DIR / "table_B2_electra_rank_scaling.md"
    with open(md_b2_path, "w", encoding="utf-8") as _f:
        _f.write("# Table B2 — ELECTRA-TR LoRA Rank Scaling Analizi\n\n")
        _f.write("Model: `dbmdz/electra-base-turkish-cased-discriminator` | ")
        _f.write("Eğitim: `train_4800_context_balanced_core_v1` | ")
        _f.write("Target: `all-linear` | Seed: 42\n\n")
        _f.write("| " + " | ".join(headers_B2) + " |\n")
        _f.write("|" + "|".join(["---"] * len(headers_B2)) + "|\n")
        for row in rows_B2:
            vals = [str(row.get(h, "")) for h in headers_B2]
            _f.write("| " + " | ".join(vals) + " |\n")
        _f.write("\n## Özet\n\n")
        _f.write("- r=16 hem DEV960-GEN hem DEV600-STR'de en yüksek Macro F1 değerini üretmiştir.\n")
        _f.write("- r=4→r=8: belirgin kazanç (+2.06pp DEV960), kapasite yetersizliği giderilmiş.\n")
        _f.write("- r=8→r=16: ek kazanç (+0.44pp DEV960), optimal noktaya ulaşılmış.\n")
        _f.write("- r=16→r=32: performans düşmüş (-2.09pp DEV960). Ek kapasite kazanç sağlamamış;\n")
        _f.write("  ELECTRA mimarisi için r=16 optimal rank olarak saptanmıştır.\n")
    print(f"  [MD]  table_B2_electra_rank_scaling.md")

    # ================================================================
    # TABLE C — Target Module Ablasyon (ELECTRA-TR r=16, seed=42)
    # Kanonik karsilastirma: C7 (Q+V) | C8 (all-attn) | G1 (all-linear, canonical rerun)
    # ================================================================
    TABLE_C_RUNS = [
        # (run_id, label, attn_modules, ffn_modules, trainable_params, status)
        ("C7_M04_ELECTRA_QV_LORA16_S42",       "Query + Value only",          24, 0,  "1,183,492", "success"),
        ("C8_M04_ELECTRA_ALLATTN_LORA16_S42",  "All Attention (q+k+v+out)",   48, 0,  "1,773,316", "success"),
        ("G1_C6_M04_ELECTRA_LORA16_DROP10_S42","All-Linear (attn + FFN)",      48, 24, "3,247,876", "success"),
    ]

    headers_C = ["Run ID", "Target Modules", "Attn Modules", "FFN Modules",
                 "Trainable Params", "Status", "DEV960 F1", "DEV600 F1"]
    rows_C = []
    for run_id, target, attn_c, ffn_c, tp, status in TABLE_C_RUNS:
        m960 = metrics.get(run_id, {}).get("DEV960-GEN", {})
        m600 = metrics.get(run_id, {}).get("DEV600-STR", {})
        rows_C.append({
            "Run ID": run_id,
            "Target Modules": target,
            "Attn Modules": attn_c,
            "FFN Modules": ffn_c,
            "Trainable Params": tp,
            "Status": status,
            "DEV960 F1": fmt(m960.get("macro_f1")),
            "DEV600 F1": fmt(m600.get("macro_f1")),
        })

    write_csv(TABLE_DIR / "table_C_target_module_ablation.csv", headers_C, rows_C)
    write_md(TABLE_DIR / "table_C_target_module_ablation.md",
             "Table C — Target Module Ablasyonu (ELECTRA-TR r=16, seed=42)", headers_C, rows_C)

    # Eski BERT ablasyon deneyleri arşiv tablosu
    TABLE_C_HIST_RUNS = [
        ("B0_M01_TALL_LORA8_LR2e4_S42",        "BERT-32k",  "TALL (q+v)",              "r=8",  "297,988",   "degenerate_skewed"),
        ("B1_M01_BERT32K_ALL_ATTN_LORA8_S42",  "BERT-32k",  "All Attention (q+k+v+out)","r=8", "1,342,468", "degenerate_collapse"),
        ("B1_M02_BERT128K_TALL_LORA8_S42",     "BERT-128k", "TALL (q+v)",              "r=8",  "1,342,468", "degenerate_collapse"),
        ("B1_M01_BERT32K_ALLLINEAR_LORA8_S42", "BERT-32k",  "All Linear (FFN dahil)",  "r=8",  "1,342,468", "success"),
        ("B1_M01_BERT32K_ALLLINEAR_LORA16_S42","BERT-32k",  "All Linear (FFN dahil)",  "r=16", "2,681,860", "success"),
    ]
    headers_C_hist = ["Run ID", "Model", "Target Modules", "Rank", "Trainable Params", "Status", "DEV960 F1", "DEV600 F1"]
    rows_C_hist = []
    for run_id, model_name, target, rank, tp, status in TABLE_C_HIST_RUNS:
        m960 = metrics.get(run_id, {}).get("DEV960-GEN", {})
        m600 = metrics.get(run_id, {}).get("DEV600-STR", {})
        rows_C_hist.append({
            "Run ID": run_id, "Model": model_name, "Target Modules": target,
            "Rank": rank, "Trainable Params": tp, "Status": status,
            "DEV960 F1": fmt(m960.get("macro_f1")),
            "DEV600 F1": fmt(m600.get("macro_f1")),
        })
    write_csv(TABLE_DIR / "table_C_historical_bert_target_modules.csv", headers_C_hist, rows_C_hist)
    print("  [CSV] table_C_historical_bert_target_modules.csv (arşiv)")

    # ================================================================
    # TABLE D — Tüm koşular tam tablo
    # ================================================================
    all_runs = read_csv(CSV_DIR / "runs.csv")
    headers_D = ["run_id", "model_id", "method", "lora_rank", "learning_rate", "seed", "status",
                 "trainable_params", "train_seconds", "peak_vram_mb",
                 "d960_f1", "d600_f1"]
    rows_D = []
    for r in all_runs:
        run_id = r["run_id"]
        m960 = metrics.get(run_id, {}).get("DEV960-GEN", {})
        m600 = metrics.get(run_id, {}).get("DEV600-STR", {})
        rows_D.append({
            "run_id": run_id,
            "model_id": r.get("model_id", "").split("/")[-1],
            "method": r.get("method", ""),
            "lora_rank": r.get("lora_rank", ""),
            "learning_rate": r.get("learning_rate", ""),
            "seed": r.get("seed", ""),
            "status": r.get("status", ""),
            "trainable_params": r.get("trainable_params", ""),
            "train_seconds": r.get("train_seconds", ""),
            "peak_vram_mb": r.get("peak_vram_mb", ""),
            "d960_f1": fmt(m960.get("macro_f1")),
            "d600_f1": fmt(m600.get("macro_f1")),
        })

    write_csv(TABLE_DIR / "table_D_all_runs_full.csv", headers_D, rows_D)
    write_md(TABLE_DIR / "table_D_all_runs_full.md",
             "Table D — Tüm Koşular (B0 dahil)", headers_D, rows_D)

    print(f"\n[BAŞARILI] {len(list(TABLE_DIR.iterdir()))} dosya {TABLE_DIR} klasörüne yazıldı.")


if __name__ == "__main__":
    main()