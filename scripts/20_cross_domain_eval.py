"""
scripts/20_cross_domain_eval.py

A1 alan bazlı eğitim — Çapraz Domain Değerlendirmesi
Her domain modeli, tüm domain dev setlerinde (medical, finance, legal) test edilir.

Çıktı:
  results/tables/table_E_cross_domain_matrix.csv
  results/tables/table_E_cross_domain_matrix.md
  results/figures/fig_07_cross_domain_heatmap.png
"""
import sys, json, torch, csv
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR  = Q1_DIR / "runs/completed"
DATA_DIR  = Q1_DIR / "data/derived/dev_domain"
TABLE_DIR = Q1_DIR / "results/tables"
FIG_DIR   = Q1_DIR / "results/figures/supplementary"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID    = {"supported": 0, "partially_supported": 1, "contradicted": 2, "unverifiable": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
BATCH_SIZE  = 32
MAX_LEN     = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# A1 modelleri: (run_id, label, train_domain)
A1_MODELS = [
    ("A1_M04_ELECTRA_TMED_LORA16_S42", "TMED",  "medical"),
    ("A1_M04_ELECTRA_TFIN_LORA16_S42", "TFIN",  "finance"),
    ("A1_M04_ELECTRA_TLEG_LORA16_S42", "TLEG",  "legal"),
    ("A1_M04_ELECTRA_TMIX_LORA16_S42", "TMIX",  "mixed"),
    ("B1_M04_ELECTRA_ALLLINEAR_LORA16_S42", "TALL", "multidomain"),
]

# Eval setleri: (label, dev960_file)
EVAL_SETS = [
    ("Medical",  DATA_DIR / "dev960_general_medical.jsonl"),
    ("Finance",  DATA_DIR / "dev960_general_finance.jsonl"),
    ("Legal",    DATA_DIR / "dev960_general_legal.jsonl"),
]

def macro_f1(preds, labels):
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average="macro", zero_division=0)

def infer_on_file(model, tokenizer, data_file: Path):
    records = [json.loads(l) for l in open(data_file, encoding="utf-8")]
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(records), BATCH_SIZE):
            batch   = records[i:i+BATCH_SIZE]
            texts_a = [f"[CONTEXT] {r.get('context', '')}" for r in batch]
            texts_b = [f"[QUESTION] {r.get('question', '')} [CLAIM] {r.get('claim', '')}" for r in batch]
            enc     = tokenizer(texts_a, texts_b, max_length=MAX_LEN, truncation=True,
                                padding=True, return_tensors="pt")
            if "token_type_ids" in enc:
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
            else:
                enc = {k: v.to(DEVICE) for k, v in enc.items() if k != "token_type_ids"}
            out     = model(**enc)
            preds   = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            for r in records[i:i+BATCH_SIZE]:
                lbl = r.get("gold_label", "")
                if lbl not in LABEL2ID:
                    raise ValueError(f"Record {r.get('example_id','?')}: unknown gold_label '{lbl}'")
                all_labels.append(LABEL2ID[lbl])
    return macro_f1(all_preds, all_labels)


def load_model_and_tokenizer(run_id: str):
    run_dir = RUNS_DIR / run_id
    config = json.load(open(run_dir / "config.json", encoding="utf-8"))
    model_name = config["model_name"]
    method = config.get("method", "lora")

    # Find checkpoint dir
    ckpt_base = run_dir / "ckpt" / model_name.split("/")[-1] / "multidomain"
    seed_dirs = list(ckpt_base.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dir found in {ckpt_base}")
    seed_dir = seed_dirs[0]
    best_model_dir = seed_dir / "best_model"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)

    if method == "lora":
        model = PeftModel.from_pretrained(base_model, best_model_dir)
        model = model.merge_and_unload()
    else:
        from safetensors.torch import load_file
        state = load_file(str(best_model_dir / "model.safetensors"))
        base_model.load_state_dict(state, strict=False)
        model = base_model

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def main():
    print("=" * 60)
    print("Cross-Domain Evaluation Matrix (A1)")
    print("=" * 60)

    # Results matrix: {model_label: {eval_label: f1}}
    matrix = {}

    for run_id, model_label, train_domain in A1_MODELS:
        print(f"\nLoading: {run_id} ({model_label})")
        try:
            model, tokenizer = load_model_and_tokenizer(run_id)
        except Exception as e:
            print(f"  [ERROR] {e}")
            matrix[model_label] = {el: None for el, _ in EVAL_SETS}
            continue

        matrix[model_label] = {}
        for eval_label, eval_file in EVAL_SETS:
            f1 = infer_on_file(model, tokenizer, eval_file)
            matrix[model_label][eval_label] = round(f1, 4)
            marker = " ← in-domain" if train_domain.startswith(eval_label.lower()[:3]) else ""
            print(f"  {eval_label}: {f1:.4f}{marker}")

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV
    eval_labels = [el for el, _ in EVAL_SETS]
    csv_path = TABLE_DIR / "table_E_cross_domain_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Train\\Eval"] + eval_labels)
        for _, model_label, _ in A1_MODELS:
            row = [model_label] + [matrix[model_label].get(el, "") for el in eval_labels]
            w.writerow(row)
    print(f"\n  [CSV] {csv_path.name}")

    # Write MD
    md_path = TABLE_DIR / "table_E_cross_domain_matrix.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table E — Çapraz Domain Değerlendirme Matrisi (ELECTRA-TR, all-linear LoRA r=16, seed=42)\n\n")
        f.write("Her satır = eğitim domain'i. Her sütun = değerlendirme domain'i. **Kalın** = in-domain.\n\n")
        headers = ["Eğitim \\ Değerlendirme"] + eval_labels
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        domain_map = {"TMED": "Medical", "TFIN": "Finance", "TLEG": "Legal", "TMIX": None, "TALL": None}
        for _, model_label, train_domain in A1_MODELS:
            in_domain_col = domain_map.get(model_label)
            cells = []
            for el in eval_labels:
                val = matrix[model_label].get(el)
                s = f"{val:.4f}" if val is not None else "—"
                if el == in_domain_col:
                    s = f"**{s}**"
                cells.append(s)
            f.write(f"| {model_label} | " + " | ".join(cells) + " |\n")
    print(f"  [MD]  {md_path.name}")

    # Write heatmap figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = []
        model_labels = [ml for _, ml, _ in A1_MODELS]
        for ml in model_labels:
            row = [matrix[ml].get(el, 0.0) or 0.0 for el in eval_labels]
            data.append(row)
        data_arr = np.array(data)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data_arr, cmap="YlOrRd", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(eval_labels)))
        ax.set_xticklabels(eval_labels, fontsize=11)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=11)
        ax.set_title("Çapraz Domain Macro F1 (Eğitim Alan × Değerlendirme Alan)", fontsize=12, pad=12)
        plt.colorbar(im, ax=ax, label="Macro F1")
        for i in range(len(model_labels)):
            for j in range(len(eval_labels)):
                val = data_arr[i, j]
                color = "white" if val < 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")
        plt.tight_layout()
        fig_path = FIG_DIR / "fig_07_cross_domain_heatmap.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  [PNG] {fig_path.name}")
    except Exception as e:
        print(f"  [WARN] Figure skipped: {e}")

    print("\n[TAMAMLANDI]")


if __name__ == "__main__":
    main()
