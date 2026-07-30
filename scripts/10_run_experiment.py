"""
scripts/10_run_experiment.py
Q1 Reproducibility Plan v2.0 Resmi Eğitim ve Değerlendirme Giriş Noktası
Protokol Bölüm 7 & 13

Her bir koşu runs/completed/<run_id>/ altına aşağıdaki paketleri üretir:
  - config.json & resolved_config.json
  - dataset_manifest.json (64-karakter SHA-256)
  - model_manifest.json
  - trainable_parameters.json (Parametre isimleri, sayıları)
  - trainer_state.json
  - metrics.json
  - validation_report.json (Sınıf çökmesi ve QC kalite denetimi)
  - predictions_DEV960-GEN.jsonl
  - predictions_DEV600-STR.jsonl
"""
import os, sys, json, time, hashlib, argparse, subprocess, torch
from pathlib import Path
from collections import Counter
import numpy as np
import transformers, peft as _peft_mod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
Q1_DIR       = PROJECT_ROOT  # repository root
RUNS_DIR     = Q1_DIR / "runs/completed"
TRAIN_PY = PROJECT_ROOT / "src" / "train.py"  # kanonik eğitim kodu (repo içinde)

LABEL2ID    = {"supported": 0, "partially_supported": 1, "contradicted": 2, "unverifiable": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["supported", "partially_supported", "contradicted", "unverifiable"]
BATCH_SIZE  = 32
MAX_LEN     = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

sys.stdout.reconfigure(encoding="utf-8")


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class NLIDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text_a = f"[CONTEXT] {r.get('context', '')}"
        text_b = f"[QUESTION] {r.get('question', '')} [CLAIM] {r.get('claim', '')}"
        enc = self.tok(
            text_a, text_b,
            max_length=MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        _gold = r.get("gold_label", "")
        if _gold not in LABEL2ID:
            raise ValueError(
                f"Record {r.get('example_id', idx)}: unknown gold_label '{_gold}'. "
                f"Expected one of {list(LABEL2ID.keys())}."
            )
        label = LABEL2ID[_gold]
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "example_id":     r.get("example_id", f"example_{idx}"),
            "domain":         r.get("domain", "?"),
            "gold_label":     r.get("gold_label", "supported"),
        }
        # Include token_type_ids when available (e.g. mBERT, BERT).
        # Without these, models that rely on segment embeddings (sentence A vs B)
        # will receive all-zero type ids and may collapse to a single class.
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def eval_and_predict(model, tokenizer, eval_file: Path):
    records = [json.loads(line) for line in open(eval_file, encoding="utf-8")]
    loader  = DataLoader(NLIDataset(records, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds, all_labels, all_probs = [], [], []
    t0 = time.time()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            # Pass token_type_ids when present (BERT-family needs them for sentence-pair tasks).
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
            out   = model(**kwargs)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_probs.extend(probs.tolist())
            
    inf_time = time.time() - t0
    examples_per_sec = len(records) / inf_time if inf_time > 0 else 0
    
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm     = confusion_matrix(all_labels, all_preds).tolist()
    
    predictions_out = []
    for r, p, prob in zip(records, all_preds, all_probs):
        predictions_out.append({
            "example_id": r.get("example_id"),
            "domain": r.get("domain"),
            "gold_label": r.get("gold_label"),
            "predicted_label": ID2LABEL[p],
            "probabilities": {ID2LABEL[i]: float(prob[i]) for i in range(4)}
        })
        
    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "supported_f1": report["supported"]["f1-score"],
        "partially_supported_f1": report["partially_supported"]["f1-score"],
        "contradicted_f1": report["contradicted"]["f1-score"],
        "unverifiable_f1": report["unverifiable"]["f1-score"],
        "inference_examples_per_second": round(examples_per_sec, 2),
        "confusion_matrix": cm,
        "predictions": predictions_out,
        "pred_class_counts": dict(Counter(all_preds))
    }


def run_q1_experiment(run_id: str, model_name: str, train_file: Path, dev960_file: Path, dev600_file: Path,
                      method: str, lr: float, freeze_layers: int, lora_r: int, lora_alpha: int, seed: int,
                      lora_dropout: float = 0.05, lora_target_modules: str = "all-linear"):
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Q1 EXPERIMENT EXECUTION: {run_id}")
    print(f"Model: {model_name} | Method: {method} | LR: {lr} | Seed: {seed}")
    print(f"Run Output Directory: {run_dir}")
    print("=" * 80)
    
    # 1. Dataset Manifest & SHA-256
    train_file_abs = train_file.resolve()
    dev960_file_abs = dev960_file.resolve()
    dev600_file_abs = dev600_file.resolve()
    
    train_sha256 = get_file_sha256(train_file_abs)
    dev960_sha256 = get_file_sha256(dev960_file_abs)
    dev600_sha256 = get_file_sha256(dev600_file_abs)
    
    train_records = [json.loads(line) for line in open(train_file_abs, encoding="utf-8")]
    dev960_records = [json.loads(line) for line in open(dev960_file_abs, encoding="utf-8")]
    dev600_records = [json.loads(line) for line in open(dev600_file_abs, encoding="utf-8")]
    
    dataset_manifest = {
        "train_file": str(train_file_abs.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in train_file_abs.parents else str(train_file_abs),
        "train_sha256": train_sha256,
        "train_num_records": len(train_records),
        "train_label_dist": dict(Counter(r.get("gold_label") for r in train_records)),
        "dev960_file": str(dev960_file_abs.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in dev960_file_abs.parents else str(dev960_file_abs),
        "dev960_sha256": dev960_sha256,
        "dev960_num_records": len(dev960_records),
        "dev600_file": str(dev600_file_abs.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in dev600_file_abs.parents else str(dev600_file_abs),
        "dev600_sha256": dev600_sha256,
        "dev600_num_records": len(dev600_records)
    }
    with open(run_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2, ensure_ascii=False)
        
    # 2. Config & Model Manifest
    config_dict = {
        "run_id": run_id,
        "model_name": model_name,
        "method": method,
        "learning_rate": lr,
        "freeze_layers": freeze_layers,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        "seed": seed,
        "batch_size": 8,
        "epochs": 5.0,
        "max_length": 512,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 2b. Model manifest — software versions & reproducibility metadata
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "unknown"
    try:
        import platform
        python_ver = platform.python_version()
    except Exception:
        python_ver = "unknown"
    try:
        import sklearn
        sklearn_ver = sklearn.__version__
    except Exception:
        sklearn_ver = "unknown"
    # Fetch model config to get revision SHA
    try:
        from transformers import AutoConfig
        _mcfg = AutoConfig.from_pretrained(model_name)
        model_revision = getattr(_mcfg, "_commit_hash", "unknown") or "unknown"
    except Exception:
        model_revision = "unknown"
    model_manifest = {
        "run_id": run_id,
        "base_model": model_name,
        "model_revision_sha": model_revision,
        "python_version": python_ver,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "transformers_version": transformers.__version__,
        "peft_version": _peft_mod.__version__,
        "sklearn_version": sklearn_ver,
        "git_commit": git_commit,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    with open(run_dir / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2, ensure_ascii=False)

    # 3. Model inspection & Trainable parameters logging
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
    
    if method == "lora":
        from peft import LoraConfig, get_peft_model

        # Use FEATURE_EXTRACTION to fully control modules_to_save.
        # SEQ_CLS auto-adds ["classifier","score"] which conflicts on DistilBERT
        # where "pre_classifier" is already LoRA-wrapped → TypeError.
        # FEATURE_EXTRACTION lets us explicitly exclude & save all head modules cleanly.
        _top_children = {name for name, _ in base_model.named_children()}
        _head_candidates = ("classifier", "pre_classifier", "score")
        _head_names = sorted(n for n in _head_candidates if n in _top_children)

        # Parse target_modules: comma-separated list or special string "all-linear"
        _target_mods = lora_target_modules.strip()
        if "," in _target_mods:
            _target_mods = [m.strip() for m in _target_mods.split(",") if m.strip()]

        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_target_mods,
            exclude_modules=_head_names if _head_names else None,
            modules_to_save=_head_names if _head_names else None,
        )
        peft_model = get_peft_model(base_model, peft_config)
        trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in peft_model.parameters())
    else:
        trainable_names = [n for n, p in base_model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in base_model.parameters())
        
    trainable_manifest = {
        "num_trainable_params": trainable_count,
        "num_total_params": total_count,
        "trainable_percentage": round(100 * trainable_count / total_count, 4),
        "trainable_param_names": trainable_names
    }
    with open(run_dir / "trainable_parameters.json", "w", encoding="utf-8") as f:
        json.dump(trainable_manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Trainable Parameters: {trainable_count:,} / {total_count:,} ({trainable_manifest['trainable_percentage']}%)")

    # 4. Training Call via verifier/train.py
    peft_flag = "lora" if method == "lora" else "none"
    cmd_train = [
        sys.executable,
        str(TRAIN_PY),
        "--domain", "multidomain",
        "--model_name", model_name,
        "--seed", str(seed),
        "--epochs", "5.0",
        "--batch_size", "8",
        "--learning_rate", str(lr),
        "--peft_type", peft_flag,
        "--freeze_layers", str(freeze_layers),
        "--train_file", str(train_file_abs),
        "--dev_file", str(dev960_file_abs),
        "--test_file", str(dev960_file_abs),
        "--output_dir", str(run_dir / "ckpt"),
        "--label_set", "4class"
    ]
    if method == "lora":
        cmd_train.extend(["--lora_r", str(lora_r), "--lora_alpha", str(lora_alpha),
                          "--lora_target_modules", lora_target_modules,
                          "--lora_dropout", str(lora_dropout)])
        
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    t0_train = time.time()
    res = subprocess.run(cmd_train, cwd=str(PROJECT_ROOT))
    train_seconds = round(time.time() - t0_train, 2)
    
    # VRAM: find peak_memory.json anywhere under ckpt/ (src/train.py writes it
    # to the nested experiment_dir, not directly to ckpt/)
    peak_vram_mb = "NA"
    peak_memory_candidates = sorted((run_dir / "ckpt").rglob("peak_memory.json")) if (run_dir / "ckpt").exists() else []
    peak_memory_file = peak_memory_candidates[0] if peak_memory_candidates else (run_dir / "ckpt" / "peak_memory.json")
    if peak_memory_file.exists():
        try:
            pm = json.loads(peak_memory_file.read_text(encoding="utf-8"))
            # src/train.py writes "peak_gpu_memory_mb"
            v = pm.get("peak_gpu_memory_mb") or pm.get("peak_vram_mb")
            if v is not None and v != 0:
                peak_vram_mb = v
        except Exception:
            pass
    if peak_vram_mb == "NA" and torch.cuda.is_available():
        # Last-resort fallback: parent process (unreliable)
        v = torch.cuda.max_memory_allocated()
        if v > 0:
            peak_vram_mb = round(v / (1024 * 1024), 2)
        
    if res.returncode != 0:
        print(f"[ERROR] Training failed for {run_id}")
        sys.exit(1)

    # 5. Extract trainer_state.json if available
    # Search under ckpt/ recursively — path varies by model name and seed
    ckpt_base = run_dir / "ckpt"
    state_candidates = sorted(ckpt_base.rglob("trainer_state.json")) if ckpt_base.exists() else []
    ckpt_dir = None
    best_epoch = None
    best_metric = None
    if state_candidates:
        # Sort by integer step number extracted from the parent directory name.
        # Lexicographic sort is incorrect: "checkpoint-900" > "checkpoint-1000" alphabetically.
        def _ckpt_step(p: Path) -> int:
            name = p.parent.name
            try:
                return int(name.split("-")[-1])
            except (ValueError, IndexError):
                return 0
        # trainer_state.json files directly in a checkpoint-N directory come first;
        # the final trainer_state (in best_model/) has step 0 and sorts last.
        state_file = max(state_candidates, key=_ckpt_step)
        ckpt_dir = state_file.parent
        state_data = json.loads(open(state_file, encoding="utf-8").read())
        best_metric = state_data.get("best_metric")
        log_hist = state_data.get("log_history", [])
        for entry in log_hist:
            if "eval_macro_f1" in entry and entry.get("eval_macro_f1") == best_metric:
                best_epoch = entry.get("epoch")
                break
        with open(run_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    else:
        # Fallback: look for best_model dir (legacy path)
        _legacy = run_dir / "ckpt" / model_name.split("/")[-1] / "multidomain" / f"seed_{seed}" / "best_model"
        ckpt_dir = _legacy if _legacy.exists() else ckpt_base
            
    # 6. Load Trained Model for Rigorous Evaluation on DEV960 & DEV600
    # Find best_model dir — prefer best_model/, fall back to last checkpoint/
    best_model_candidates = sorted(ckpt_base.rglob("best_model")) if ckpt_base.exists() else []
    if best_model_candidates:
        ckpt_dir = best_model_candidates[-1]
    print(f"Loading trained best model from: {ckpt_dir}...")
    if method == "lora":
        raw_base = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
        model = PeftModel.from_pretrained(raw_base, str(ckpt_dir))
        # merge_and_unload() handles both LoRA weights and modules_to_save correctly.
        # The manual unwrap loop above was buggy — it corrupted modules_to_save for
        # non-BERT architectures (RoBERTa, ELECTRA, etc.) by replacing trained modules
        # with untrained base modules before merge.
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
        
    model.to(DEVICE)
    model.eval()
    
    res_d960 = eval_and_predict(model, tokenizer, dev960_file_abs)
    res_d600 = eval_and_predict(model, tokenizer, dev600_file_abs)
    
    # Save predictions
    with open(run_dir / "predictions_DEV960-GEN.jsonl", "w", encoding="utf-8") as f:
        for item in res_d960["predictions"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(run_dir / "predictions_DEV600-STR.jsonl", "w", encoding="utf-8") as f:
        for item in res_d600["predictions"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # 7. Quality Control Check (QC)
    qc_status = "success"
    qc_failures = []
    
    if res_d960["macro_f1"] < 0.30:
        qc_status = "degenerate_class_collapse"
        qc_failures.append(f"DEV960 Macro F1 ({res_d960['macro_f1']:.4f}) is below 0.30 threshold.")
        
    for label_name in LABEL_NAMES:
        key_f1 = f"{label_name}_f1"
        if res_d960[key_f1] == 0.0:
            qc_status = "degenerate_class_collapse"
            qc_failures.append(f"Class '{label_name}' has 0.0 F1 score in DEV960.")
            
    val_report = {
        "qc_status": qc_status,
        "qc_failures": qc_failures,
        "d960_pred_class_counts": res_d960["pred_class_counts"],
        "d600_pred_class_counts": res_d600["pred_class_counts"],
        "d960_confusion_matrix": res_d960["confusion_matrix"],
        "d600_confusion_matrix": res_d600["confusion_matrix"]
    }
    with open(run_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, ensure_ascii=False)
        
    # Save Final Metrics JSON
    metrics_data = {
        "run_id": run_id,
        "model_id": model_name,
        "train_dataset": train_file_abs.stem,
        "train_sha256": train_sha256,
        "method": method,
        "freeze_layers": freeze_layers,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": lr,
        "seed": seed,
        "best_epoch": best_epoch if best_epoch is not None else 5,
        "best_metric": best_metric,
        "d960_accuracy": round(res_d960["accuracy"], 4),
        "d960_macro_f1": round(res_d960["macro_f1"], 4),
        "d960_supported_f1": round(res_d960["supported_f1"], 4),
        "d960_partially_supported_f1": round(res_d960["partially_supported_f1"], 4),
        "d960_contradicted_f1": round(res_d960["contradicted_f1"], 4),
        "d960_unverifiable_f1": round(res_d960["unverifiable_f1"], 4),
        "d600_accuracy": round(res_d600["accuracy"], 4),
        "d600_macro_f1": round(res_d600["macro_f1"], 4),
        "d600_supported_f1": round(res_d600["supported_f1"], 4),
        "d600_partially_supported_f1": round(res_d600["partially_supported_f1"], 4),
        "d600_contradicted_f1": round(res_d600["contradicted_f1"], 4),
        "d600_unverifiable_f1": round(res_d600["unverifiable_f1"], 4),
        "trainable_params": trainable_count,
        "peak_vram_mb": peak_vram_mb,
        "train_seconds": train_seconds,
        "inference_examples_per_second": res_d960["inference_examples_per_second"],
        "qc_status": qc_status
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print(f"Q1 RUN COMPLETE: {run_id}")
    print(f"DEV960 Macro F1: {res_d960['macro_f1']*100:.2f}% | DEV600 Macro F1: {res_d600['macro_f1']*100:.2f}%")
    print(f"QC Status: {qc_status}")
    print("=" * 80)
    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--train_file", type=str, default=str(Q1_DIR / "data/primary/train.jsonl"))
    parser.add_argument("--dev960_file", type=str, default=str(Q1_DIR / "data/evaluation/legacy_experiment_version/dev_general.jsonl"))
    parser.add_argument("--dev600_file", type=str, default=str(Q1_DIR / "data/evaluation/legacy_experiment_version/dev_stress.jsonl"))
    parser.add_argument("--method", type=str, default="lora")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="LoRA target modules. 'all-linear' or comma-sep list e.g. 'query,key,value,output.dense'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_q1_experiment(
        run_id=args.run_id,
        model_name=args.model_name,
        train_file=Path(args.train_file),
        dev960_file=Path(args.dev960_file),
        dev600_file=Path(args.dev600_file),
        method=args.method,
        lr=args.learning_rate,
        freeze_layers=args.freeze_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        lora_target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout
    )

