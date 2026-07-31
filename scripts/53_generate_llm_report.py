"""Generate canonical LLM-baseline tables, reports, and figures.

The report is built from run manifests and evaluation metrics already stored under
``results/llm_baselines``. It does not call any model API.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results/llm_baselines"
TABLE_DIR = ROOT / "results/tables"
REPORT_DIR = ROOT / "reports"
FIG_EN = ROOT / "results/figures/en"
FIG_TR = ROOT / "results/figures/tr"

MODELS = [
    ("gemma_4_26b_moe_free", "Gemma 4 26B A4B IT"),
    ("gpt_4_1_mini", "GPT-4.1 mini"),
    ("llama_3_3_70b", "Llama 3.3 70B Instruct"),
    ("qwen2_5_72b", "Qwen2.5 72B Instruct"),
]
SPLITS = [
    ("dev_general", "DEV960-GEN", 960, 0.9014),
    ("dev_stress", "DEV600-STR", 600, 0.8675),
]
MODES = ["zero_shot", "few_shot_8"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows() -> list[dict]:
    out: list[dict] = []
    for split, split_label, examples, encoder_f1 in SPLITS:
        for alias, display in MODELS:
            for mode in MODES:
                run_dir = RESULT_ROOT / split / alias / mode
                manifest = load_json(run_dir / "manifest.json")
                metrics = load_json(run_dir / "metrics/metrics.json")
                strict = metrics["strict_metrics"]
                out.append(
                    {
                        "dataset": split,
                        "dataset_label": split_label,
                        "examples": examples,
                        "model": display,
                        "model_alias": alias,
                        "requested_model": manifest.get("requested_model", ""),
                        "resolved_model": metrics.get("resolved_model") or manifest.get("requested_model", ""),
                        "prompt_mode": mode,
                        "prompt_label": "Zero-shot" if mode == "zero_shot" else "8-shot",
                        "strict_macro_f1": float(strict["macro_f1"]),
                        "strict_accuracy": float(strict["accuracy"]),
                        "coverage": float(metrics["coverage"]),
                        "invalid_or_missing": int(metrics["invalid_or_missing_predictions"]),
                        "encoder_selected_f1": encoder_f1,
                        "system_prompt_sha256": manifest.get("system_prompt_sha256", ""),
                        "fewshot_sha256": manifest.get("fewshot_sha256") or "",
                    }
                )
    return out


def table_for(data: list[dict], split: str) -> str:
    selected = [r for r in data if r["dataset"] == split]
    selected.sort(key=lambda r: r["strict_macro_f1"], reverse=True)
    lines = [
        "| Model | Prompt | Macro-F1 | Accuracy | Coverage | Invalid/missing |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in selected:
        lines.append(
            f"| {r['model']} | {r['prompt_label']} | {r['strict_macro_f1']:.4f} | "
            f"{r['strict_accuracy']:.4f} | {r['coverage']:.4f} | {r['invalid_or_missing']} |"
        )
    return "\n".join(lines)


def write_reports(data: list[dict]) -> None:
    general = table_for(data, "dev_general")
    stress = table_for(data, "dev_stress")
    common_note = (
        "Archived scores were produced on the original pre-review evaluation files. The "
        "single-annotator preview changes 2/960 DEV960 labels and 3/600 Dev600 labels, "
        "with one additional Dev600 claim correction. Reported values may therefore shift "
        "slightly on the reviewed preview files; API calls and model training were not repeated."
    )
    en = f"""# TR-FactBench LLM Baselines

Four general-purpose LLMs were evaluated in two prompting conditions: instruction-only
zero-shot and fixed 8-shot. Each request contains exactly one context-claim pair. Gold
labels, candidate labels, example IDs, and context IDs are not exposed to the model.
Strict Macro-F1 counts missing or invalid outputs as errors.

## DEV960-GEN

{general}

Selected ELECTRA checkpoint (`H1_DROP10_S45`): **0.9014 Macro-F1**.

## DEV600-STR

{stress}

Selected ELECTRA checkpoint (`H1_DROP10_S45`): **0.8675 Macro-F1**.

## Interpretation

- Few-shot prompting strongly improves GPT-4.1 mini and Llama 3.3 70B on both sets.
- Qwen2.5 72B improves on DEV960 but changes little on DEV600; one output is invalid or
  missing in each DEV600 prompting condition.
- Gemma is strongest overall in these runs. On DEV600 its zero-shot score exceeds its
  8-shot score, so demonstrations are not uniformly beneficial.
- The selected local ELECTRA checkpoint remains competitive with several API baselines,
  especially on DEV600, while avoiding per-example API inference.

## Protocol notes

- The exact requested and resolved model IDs are preserved in each run manifest and in
  `results/llm_baselines/experiment_matrix.csv`.
- DEV600 Gemma runs requested the OpenRouter `:free` route, whereas DEV960 requested the
  standard model route. These are reported transparently rather than treated as an
  identical provider-routing condition.
- All few-shot runs used the same fixed demonstration file with SHA-256
  `2b30bc4360ea5922d3b969a1fcc65528cf10df481afecdd7ace968142d78e8ba`.
- {common_note}

Machine-readable results are available in `results/tables/table_I_llm_baselines.csv` and
`results/llm_baselines/leaderboard.csv`.
"""
    tr = f"""# TR-FactBench LLM Baseline Sonuçları

Dört genel amaçlı LLM, yalnız yönerge içeren zero-shot ve sabit 8-shot olmak üzere iki
prompt koşulunda değerlendirilmiştir. Her istek yalnızca tek bir bağlam-iddia çifti içerir.
Gold etiket, aday etiket, örnek kimliği ve bağlam kimliği modele gösterilmez. Strict
Macro-F1 hesabında eksik veya geçersiz çıktılar hata kabul edilir.

## DEV960-GEN

{general}

Seçilen ELECTRA checkpointi (`H1_DROP10_S45`): **0.9014 Macro-F1**.

## DEV600-STR

{stress}

Seçilen ELECTRA checkpointi (`H1_DROP10_S45`): **0.8675 Macro-F1**.

## Yorum

- Few-shot, GPT-4.1 mini ve Llama 3.3 70B sonuçlarını iki kümede de belirgin biçimde artırmıştır.
- Qwen2.5 72B, DEV960'ta iyileşmiş; DEV600'de ise çok sınırlı değişmiştir. DEV600'de her
  iki prompt koşulunda birer geçersiz veya eksik çıktı vardır.
- Bu koşularda en yüksek sonuçları Gemma vermiştir. DEV600'de zero-shot sonucu 8-shot
  sonucundan yüksektir; dolayısıyla örnek eklemek her model ve kümede yarar sağlamamıştır.
- Seçilen yerel ELECTRA checkpointi, özellikle DEV600'de bazı API tabanlı LLM koşullarıyla
  rekabetçi kalırken örnek başına API çıkarımı gerektirmez.

## Protokol notları

- İstenen ve çözümlenen tam model kimlikleri run manifestlerinde ve
  `results/llm_baselines/experiment_matrix.csv` dosyasında korunur.
- DEV600 Gemma koşuları OpenRouter `:free` rotasını, DEV960 koşuları ise standart model
  rotasını istemiştir. Bu fark gizlenmemiş ve aynı sağlayıcı yönlendirme koşuluymuş gibi
  sunulmamıştır.
- Bütün few-shot koşularında SHA-256 değeri
  `2b30bc4360ea5922d3b969a1fcc65528cf10df481afecdd7ace968142d78e8ba` olan sabit örnek dosyası kullanılmıştır.
- Arşivdeki skorlar özgün, inceleme öncesi değerlendirme dosyalarında üretilmiştir. Tek
  anotatörlü preview sürümünde DEV960'ta 2/960, Dev600'de 3/600 etiket değişmiş ve Dev600'de
  bir iddia düzeltmesi yapılmıştır. Güncel preview dosyalarında skorlar küçük ölçüde değişebilir;
  bu sınırlı revizyonlar için API çağrıları ve model eğitimleri tekrarlanmamıştır.

Makine-okunur sonuçlar `results/tables/table_I_llm_baselines.csv` ve
`results/llm_baselines/leaderboard.csv` dosyalarındadır.
"""
    (REPORT_DIR / "LLM_BASELINES.md").write_text(en, encoding="utf-8")
    (REPORT_DIR / "LLM_BASELINES_TR.md").write_text(tr, encoding="utf-8")


def write_csv_and_md(data: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "dataset_label", "examples", "model", "model_alias",
        "requested_model", "resolved_model", "prompt_mode", "strict_macro_f1",
        "strict_accuracy", "coverage", "invalid_or_missing",
        "system_prompt_sha256", "fewshot_sha256",
    ]
    with (TABLE_DIR / "table_I_llm_baselines.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in data:
            writer.writerow({k: r[k] for k in fields})
    with (RESULT_ROOT / "experiment_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in data:
            writer.writerow({k: r[k] for k in fields})
    md = "# Table I — LLM baselines\n\n## DEV960-GEN\n\n" + table_for(data, "dev_general")
    md += "\n\n## DEV600-STR\n\n" + table_for(data, "dev_stress") + "\n"
    (TABLE_DIR / "table_I_llm_baselines.md").write_text(md, encoding="utf-8")


def write_figures(data: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    for split, split_label, _, encoder_f1 in SPLITS:
        subset = [r for r in data if r["dataset"] == split]
        labels = [display for _, display in MODELS]
        zero = [next(r["strict_macro_f1"] for r in subset if r["model"] == label and r["prompt_mode"] == "zero_shot") for label in labels]
        few = [next(r["strict_macro_f1"] for r in subset if r["model"] == label and r["prompt_mode"] == "few_shot_8") for label in labels]
        x = np.arange(len(labels))
        width = 0.36
        for lang, outdir in [("en", FIG_EN), ("tr", FIG_TR)]:
            outdir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            b1 = ax.bar(x - width / 2, zero, width, label="Zero-shot")
            b2 = ax.bar(x + width / 2, few, width, label="8-shot")
            ax.axhline(encoder_f1, linestyle="--", linewidth=1.5, label=f"ELECTRA S45 ({encoder_f1:.4f})")
            ax.set_ylim(0.5, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=12, ha="right")
            ax.set_ylabel("Macro-F1")
            title = (
                f"LLM baselines on {split_label}"
                if lang == "en"
                else f"{split_label} üzerinde LLM baseline sonuçları"
            )
            ax.set_title(title)
            ax.legend()
            ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
            ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            suffix = "dev960" if split == "dev_general" else "dev600"
            fig.savefig(outdir / f"fig_10_llm_baselines_{suffix}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    data = rows()
    write_csv_and_md(data)
    write_reports(data)
    write_figures(data)
    print(f"[OK] generated LLM reports for {len(data)} experiment conditions")


if __name__ == "__main__":
    main()
