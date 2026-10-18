"""
run_gold_llm_eval.py
====================
TR-FactBench gold_v1.0 (480 örnek, çift-annotator, held-out) üzerinde
LLM baseline değerlendirmesi.

dev960 ve dev600 zaten önceden test edildi — bkz. reports/LLM_BASELINES.md.
Bu script yalnızca henüz hiç LLM ile test edilmemiş gold_v1.0 içindir.

Mevcut run_baseline.py altyapısını doğrudan kullanır.
Gold dosyasındaki `release_example_id` → `example_id` dönüşümü burada yapılır.

Kullanım:
    # GPT-4.1 mini, zero-shot, dry-run (API çağrısı yapmaz)
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot --dry-run

    # GPT-4.1 mini, zero-shot (tam çalıştırma)
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot

    # GPT-4.1 mini, 8-shot
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode few_shot_8

    # Gemma, 8-shot
    python run_gold_llm_eval.py --model google/gemma-2-27b-it --prompt-mode few_shot_8

    # Llama, 8-shot
    python run_gold_llm_eval.py --model meta-llama/llama-3.3-70b-instruct --prompt-mode few_shot_8

    # Qwen, zero-shot
    python run_gold_llm_eval.py --model qwen/qwen2.5-72b-instruct --prompt-mode zero_shot

    # Yalnızca mevcut tahminleri değerlendir (API çağrısı yapma)
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot --eval-only

    # Kesilen çalıştırmayı devam ettir (run_baseline.py otomatik resume yapar)
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot

    # Status=failed örnekleri yeniden çalıştır
    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot --retry-failed

    # Tüm gold_v1.0 sonuçlarını karşılaştır
    python run_gold_llm_eval.py --compare

Gerekli: OPENROUTER_API_KEY ortam değişkeni veya .env dosyası.

Çıktı dizini: results/llm_baselines/gold_v1_0/<model_slug>/<prompt_mode>/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import LABELS, read_jsonl, safe_slug, utc_now_iso, write_jsonl_atomic  # noqa: E402

# ── Sabit: gold_v1.0 ──────────────────────────────────────────────────────────
# dev960 ve dev600 zaten önceden test edildi — bkz. reports/LLM_BASELINES.md

GOLD_DATASET_KEY = "gold_v1.0"
GOLD_DATASET_PATH = (
    REPO_ROOT / "data" / "evaluation" / "gold_v1.0" / "gold"
    / "TR-FactBench_controlled480_GOLD_v1.0.jsonl"
)
GOLD_ID_FIELD = "release_example_id"   # gold dosyasında farklı alan adı
GOLD_DESCRIPTION = "480 örnek, çift-annotator, held-out gold set"
GOLD_N_EXPECTED = 480

OUTPUT_ROOT = REPO_ROOT / "results" / "llm_baselines"


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def adapt_gold_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Gold dosyasında `release_example_id` kullanılıyor, run_baseline.py
    `example_id` bekliyor. Gerektiğinde alanı yeniden adlandır.
    """
    adapted = []
    for rec in records:
        new_rec = dict(rec)
        new_rec["example_id"] = rec[GOLD_ID_FIELD]
        adapted.append(new_rec)
    return adapted


def resolve_paths(model: str, prompt_mode: str, model_alias: str | None = None) -> dict[str, Path]:
    dataset_slug = safe_slug(GOLD_DATASET_KEY)
    model_slug = safe_slug(model_alias or model)
    run_dir = OUTPUT_ROOT / dataset_slug / model_slug / prompt_mode
    return {
        "run_dir": run_dir,
        "predictions": run_dir / "predictions.jsonl",
        "raw_responses": run_dir / "raw_responses.jsonl",
        "request_log": run_dir / "request_log.jsonl",
        "manifest": run_dir / "manifest.json",
        "metrics": run_dir / "metrics" / "metrics.json",
    }


def run_baseline(adapted_file: Path, args: argparse.Namespace) -> int:
    """run_baseline.py'yi subprocess olarak çalıştır."""
    cmd = [
        sys.executable,
        str(ROOT / "run_baseline.py"),
        "--input", str(adapted_file),
        "--dataset-name", GOLD_DATASET_KEY,
        "--model", args.model,
        "--prompt-mode", args.prompt_mode,
        "--output-root", str(OUTPUT_ROOT),
        "--sleep-between-requests", str(args.sleep),
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--timeout", str(args.timeout),
        "--max-retries", str(args.max_retries),
    ]
    if args.model_alias:
        cmd += ["--model-alias", args.model_alias]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.retry_failed:
        cmd.append("--retry-failed")
    if args.free_tier:
        cmd.append("--free-tier")
    if args.zdr:
        cmd.append("--zdr")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.api_key_env != "OPENROUTER_API_KEY":
        cmd += ["--api-key-env", args.api_key_env]

    print(f"\n[run_baseline.py] Komut:")
    print("  " + " ".join(str(c) for c in cmd))
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def run_evaluate(predictions_path: Path, model: str, prompt_mode: str, model_alias: str | None = None) -> int:
    """eval_gold.py'yi çalıştır. Gold dosyası example_id değil release_example_id kullanır."""
    if not predictions_path.exists():
        print(f"[WARN] Tahmin dosyası bulunamadı: {predictions_path}")
        return 1

    cmd = [
        sys.executable,
        str(ROOT / "eval_gold.py"),
        "--model", model,
        "--prompt-mode", prompt_mode,
    ]
    if model_alias:
        cmd += ["--model-alias", model_alias]
    print(f"\n[eval_gold.py] Komut:")
    print("  " + " ".join(str(c) for c in cmd))
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def _metric_value(metrics: dict[str, Any], nested_key: str, flat_key: str, nested_parent: str) -> Any:
    nested = metrics.get(nested_parent) or {}
    if nested and nested.get(nested_key) is not None:
        return nested.get(nested_key)
    return metrics.get(flat_key)


def print_summary(metrics_path: Path, model: str, prompt_mode: str) -> None:
    """Sonuç özetini terminale yaz."""
    if not metrics_path.exists():
        return
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    print("\n" + "=" * 60)
    print(f"  Sonuç: {GOLD_DATASET_KEY} | {model} | {prompt_mode}")
    print("=" * 60)
    acc = _metric_value(m, "accuracy", "accuracy", "overall") or 0
    macro = _metric_value(m, "macro_f1", "macro_f1", "overall") or 0
    weighted = _metric_value(m, "weighted_f1", "weighted_f1", "overall") or 0
    coverage = _metric_value(m, "coverage_rate", "coverage_rate", "coverage") or 0
    valid = _metric_value(m, "valid", "n_valid", "coverage") or 0
    total = _metric_value(m, "total", "n_total", "coverage") or 0
    print(f"  Accuracy    : {acc * 100:.2f}%")
    print(f"  Macro F1    : {macro * 100:.2f}%")
    print(f"  Weighted F1 : {weighted * 100:.2f}%")
    print(f"  Coverage    : {coverage * 100:.1f}%  ({valid}/{total})")
    per_class = m.get("per_class", {})
    if per_class:
        print("-" * 60)
        print("  Per-Class F1:")
        for lbl in LABELS:
            f1 = (per_class.get(lbl, {}).get("f1") or 0) * 100
            n = per_class.get(lbl, {}).get("support", 0)
            print(f"    {lbl:<25}: {f1:.2f}%  (n={n})")
    print("=" * 60)


def print_comparison_table() -> None:
    """Mevcut tüm gold_v1.0 sonuçlarını karşılaştırma tablosu olarak yazdır."""
    gold_slug = safe_slug(GOLD_DATASET_KEY)
    gold_dir = OUTPUT_ROOT / gold_slug
    print("\n" + "=" * 85)
    print(f"  TR-FactBench gold_v1.0 — LLM Karşılaştırma Tablosu")
    print("=" * 85)
    print(f"  {'Model':<45} {'Mode':<12} {'N':<6} {'Acc%':<8} {'MacroF1%':<10} {'Cov%'}")
    print("-" * 85)

    rows = []
    if gold_dir.exists():
        for model_dir in sorted(gold_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for mode_dir in sorted(model_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                metrics_path = mode_dir / "metrics" / "metrics.json"
                manifest_path = mode_dir / "manifest.json"
                if not metrics_path.exists():
                    continue
                with open(metrics_path, encoding="utf-8") as f:
                    m = json.load(f)
                model_name = model_dir.name
                if manifest_path.exists():
                    with open(manifest_path, encoding="utf-8") as f:
                        mf = json.load(f)
                    model_name = mf.get("requested_model", model_dir.name)
                rows.append({
                    "model": model_name,
                    "mode": mode_dir.name,
                    "n": _metric_value(m, "valid", "n_valid", "coverage") or 0,
                    "accuracy": _metric_value(m, "accuracy", "accuracy", "overall") or 0,
                    "macro_f1": _metric_value(m, "macro_f1", "macro_f1", "overall") or 0,
                    "coverage": _metric_value(m, "coverage_rate", "coverage_rate", "coverage") or 0,
                })

    if not rows:
        print("  Henüz değerlendirme sonucu yok.")
        print("  Başlamak için:")
        print("    python run_gold_llm_eval.py --model openai/gpt-4.1-mini --prompt-mode zero_shot")
    else:
        rows.sort(key=lambda r: -r["macro_f1"])
        for r in rows:
            print(
                f"  {r['model']:<45} {r['mode']:<12} {r['n']:<6} "
                f"{r['accuracy'] * 100:<8.2f} {r['macro_f1'] * 100:<10.2f} "
                f"{r['coverage'] * 100:.1f}%"
            )
    print("=" * 85 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TR-FactBench gold_v1.0 (480 örnek, held-out) üzerinde LLM baseline değerlendirmesi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-4.1-mini",
        help="OpenRouter model ID (örn: openai/gpt-4.1-mini, google/gemma-2-27b-it, meta-llama/llama-3.3-70b-instruct)",
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Çıktı klasörü adı için model kısaltması (varsayılan: model slug'ından türetilir)",
    )
    parser.add_argument(
        "--prompt-mode", "-p",
        choices=["zero_shot", "few_shot_8"],
        default="zero_shot",
        help="Prompt modu: zero_shot veya few_shot_8 (varsayılan: zero_shot)",
    )

    # API ayarları
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="API anahtarını içeren ortam değişkeni (varsayılan: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float, default=0.0,
        help="Örnekleme sıcaklığı (varsayılan: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int, default=1000,
        help="Model çıktısı için maksimum token (varsayılan: 1000)",
    )
    parser.add_argument(
        "--timeout",
        type=int, default=240,
        help="HTTP istek zaman aşımı saniye cinsinden (varsayılan: 240)",
    )
    parser.add_argument(
        "--max-retries",
        type=int, default=5,
        help="Başarısız istek için maksimum yeniden deneme (varsayılan: 5)",
    )
    parser.add_argument(
        "--sleep", "-s",
        type=float, default=0.25,
        help="API istekleri arasındaki bekleme süresi saniye cinsinden (varsayılan: 0.25)",
    )

    # Çalışma modları
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="İlk örnek için prompt önizlemesi yazdır, API çağrısı yapma",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="API çağrısı yapmadan mevcut tahminleri değerlendir",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Status=failed olan örnekleri yeniden çalıştır",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int, default=None,
        help="Test/dry-run için işlenecek maksimum örnek sayısı",
    )
    parser.add_argument(
        "--free-tier",
        action="store_true",
        help="OpenRouter :free model rotası için provider routing bloğunu atla",
    )
    parser.add_argument(
        "--zdr",
        action="store_true",
        help="OpenRouter Zero Data Retention (ZDR) zorunlu kıl",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Çalıştırma tamamlandıktan sonra evaluate_baseline.py'yi çalıştırma",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Mevcut tüm gold_v1.0 değerlendirme sonuçlarını karşılaştır",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Karşılaştırma modu
    if args.compare:
        print_comparison_table()
        return

    if not GOLD_DATASET_PATH.exists():
        print(f"[ERROR] Dataset dosyası bulunamadı: {GOLD_DATASET_PATH}")
        sys.exit(1)

    records = read_jsonl(GOLD_DATASET_PATH)
    print(f"[INFO] Dataset: {GOLD_DATASET_KEY} — {GOLD_DESCRIPTION}")
    print(f"[INFO] Yüklenen örnek sayısı: {len(records)} (beklenen: {GOLD_N_EXPECTED})")

    paths = resolve_paths(args.model, args.prompt_mode, args.model_alias)
    print(f"[INFO] Çıktı dizini: {paths['run_dir']}")

    # release_example_id → example_id dönüşümü + geçici dosya
    adapted = adapt_gold_records(records)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix="tfb_gold_v1_0_adapted_",
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as tmp:
        tmp_path = Path(tmp.name)
        for rec in adapted:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Geçici adapt edilmiş dosya: {tmp_path} ({len(adapted)} örnek)")

    try:
        # Çalıştırma aşaması
        if not args.eval_only:
            exit_code = run_baseline(tmp_path, args)
            if exit_code != 0 and not args.dry_run:
                print(f"\n[WARN] run_baseline.py exit_code={exit_code} — kısmi çalıştırma mümkün.")

        # Değerlendirme aşaması
        if not args.dry_run and not args.skip_eval:
            if paths["predictions"].exists():
                print("\n[INFO] Değerlendirme başlıyor...")
                eval_exit = run_evaluate(paths["predictions"], args.model, args.prompt_mode, args.model_alias)
                if eval_exit == 0:
                    print_summary(paths["metrics"], args.model, args.prompt_mode)
            else:
                print("\n[INFO] Tahmin dosyası henüz yok, değerlendirme atlandı.")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()