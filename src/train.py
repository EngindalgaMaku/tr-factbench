#!/usr/bin/env python3
"""Unified training and evaluation script for hallucination verification."""
import os
import sys
import json
import argparse
import random
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Varsayilan 5-sinif (v16 ve oncesi)
LABEL2ID_5CLASS = {
    "supported": 0,
    "partially_supported": 1,
    "unsupported": 2,
    "contradicted": 3,
    "insufficient_information": 4,
}

# Kanonik 4-sinif (v17+)
LABEL2ID_4CLASS = {
    "supported": 0,
    "partially_supported": 1,
    "contradicted": 2,
    "unverifiable": 3,
}

# Aktif etiket haritasi — main() icinde args.label_set'e gore atanir
LABEL2ID: dict[str, int] = LABEL2ID_5CLASS
ID2LABEL:  dict[int, str] = {v: k for k, v in LABEL2ID.items()}

REQUIRED_FIELDS = {
    "example_id",
    "context_id",
    "context",
    "question",
    "claim",
    "gold_label"
}

def set_deterministic_seed(seed: int):
    """Sets seed for PyTorch, NumPy, random and configures CUDA determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic seed set to: {seed}")

def load_jsonl_data(path: Path) -> list[dict]:
    """Loads a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error(f"Failed parsing JSON on line {line_no} of {path}: {exc}")
                sys.exit(1)
    return data

def validate_dataset(data: list[dict], name: str, is_multidomain: bool = False):
    """Performs rigorous checks on loaded datasets before training."""
    logger.info(f"Running data validation checks on split: {name} ({len(data)} records)")
    
    seen_ids = set()
    label_counts = Counter()
    
    for idx, r in enumerate(data, 1):
        # 1. Required fields present
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            logger.error(f"Row {idx} in {name} is missing mandatory fields: {missing}")
            sys.exit(1)
            
        if is_multidomain and "domain" not in r:
            # Eski format finance verisi domain field'i icermeyebilir — varsayilan ata
            logger.warning(f"Row {idx} in multidomain split {name} is missing 'domain'; inferring from example_id.")
            if "example_id" in r:
                eid = r["example_id"].lower()
                if "finance" in eid:
                    r["domain"] = "finance"
                elif "legal" in eid or "law" in eid:
                    r["domain"] = "legal"
                elif "medical" in eid or "med" in eid:
                    r["domain"] = "medical"
                else:
                    r["domain"] = "finance"  # v4'te eksik satirlar finance'tan
            else:
                r["domain"] = "finance"
            
        # 2. Fields must not be empty
        for field in REQUIRED_FIELDS:
            if not str(r[field]).strip():
                logger.error(f"Row {idx} in {name} has an empty field: '{field}'")
                sys.exit(1)
                
        # 3. Gold label must be valid
        label = r["gold_label"]
        if label not in LABEL2ID:
            logger.error(f"Row {idx} in {name} has an invalid gold label: '{label}'")
            sys.exit(1)
        label_counts[label] += 1
        
        # 4. example_id must be unique
        eid = r["example_id"]
        if eid in seen_ids:
            logger.error(f"Row {idx} in {name} has duplicate example_id: '{eid}'")
            sys.exit(1)
        seen_ids.add(eid)
        
    # Print label distribution
    logger.info(f"{name} label distribution:")
    for label in sorted(LABEL2ID.keys()):
        logger.info(f"  {label}: {label_counts[label]}")
        
    return set(r["context_id"] for r in data)

def detect_leakage(train_ctxs: set[str], dev_ctxs: set[str]):
    """Asserts that there is no context_id leakage between train and dev splits."""
    overlap = train_ctxs & dev_ctxs
    if overlap:
        logger.error(f"DATA LEAKAGE DETECTED! {len(overlap)} context_ids overlap between train and dev splits: {overlap}")
        sys.exit(1)
    logger.info("Validation PASS: No train/dev context_id leakage detected.")

def apply_shortcut_ablation(data: list[dict], view: str, seed: int) -> list[dict]:
    """Applies the selected shortcut or ablation view to the dataset inputs."""
    processed = []
    
    if view == "shuffled_context_claim":
        # Extract and shuffle all contexts.
        # Always use seed=0 for consistency with the published ablation protocol
        # (README: "fixed seed of 0") regardless of the training seed argument.
        # Also guarantee no self-matches: a record must not receive its own context.
        contexts = [r["context"] for r in data]
        rng = random.Random(0)
        indices = list(range(len(contexts)))
        rng.shuffle(indices)
        # Repair any position where a record would still get its own context.
        for i in range(len(indices)):
            if indices[i] == i:
                # Swap with the next position (wrapping around)
                swap_target = (i + 1) % len(indices)
                indices[i], indices[swap_target] = indices[swap_target], indices[i]
        shuffled_contexts = [contexts[j] for j in indices]

        for r, shuffled_ctx in zip(data, shuffled_contexts):
            processed.append({
                "example_id": r["example_id"],
                "context_id": r["context_id"],
                "context": shuffled_ctx,
                "question": r["question"],
                "claim": r["claim"],
                "gold_label": r["gold_label"]
            })
    else:
        # Standard views
        for r in data:
            ctx = r["context"]
            q = r["question"]
            cl = r["claim"]
            
            # Format inputs based on selected view
            if view == "full_input":
                text_a = f"[CONTEXT] {ctx}"
                text_b = f"[QUESTION] {q} [CLAIM] {cl}"
            elif view == "question_only":
                text_a = f"[QUESTION] {q}"
                text_b = None
            elif view == "claim_only":
                text_a = f"[CLAIM] {cl}"
                text_b = None
            elif view == "question_claim":
                text_a = f"[QUESTION] {q} [CLAIM] {cl}"
                text_b = None
            elif view == "context_only":
                text_a = f"[CONTEXT] {ctx}"
                text_b = None
            elif view == "context_claim":
                text_a = f"[CONTEXT] {ctx}"
                text_b = f"[CLAIM] {cl}"
            else:
                logger.error(f"Unknown input view setting: {view}")
                sys.exit(1)
                
            processed.append({
                "example_id": r["example_id"],
                "context_id": r["context_id"],
                "context": text_a,
                "claim": text_b,  # Store text_b in 'claim' field so get_verifier_dataset works out-of-the-box
                "gold_label": r["gold_label"]
            })
            
    return processed

def compute_metrics(eval_pred):
    """Computes Macro F1, accuracy, weighted F1 and class-wise metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
    }
    
    # Add class-wise precision, recall, and f1
    for label_name, idx in LABEL2ID.items():
        metrics[f"{label_name}_precision"] = precision_per[idx] if idx < len(precision_per) else 0.0
        metrics[f"{label_name}_recall"] = recall_per[idx] if idx < len(recall_per) else 0.0
        metrics[f"{label_name}_f1"] = f1_per[idx] if idx < len(f1_per) else 0.0
        
    return metrics

def freeze_encoder_layers(model, num_frozen_layers: int):
    """Freezes embeddings and bottom num_frozen_layers encoder layers.
    Supports both BERT-family (model.bert) and ModernBERT-family (model.model) architectures.
    """
    model_type = model.config.model_type  # e.g. 'bert' or 'modernbert'
    
    if hasattr(model, 'bert'):
        # BERT-family: bert.embeddings, bert.encoder.layer
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        encoder_layers = model.bert.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"  Layer {i}: trainable")
        # Keep pooler and classifier trainable
        for param in model.bert.pooler.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # ModernBERT-family: model.embeddings, model.layers
        for param in model.model.embeddings.parameters():
            param.requires_grad = False
        encoder_layers = model.model.layers
        for i, layer in enumerate(encoder_layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"  Layer {i}: trainable")
        # Keep head and classifier trainable
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'deberta'):
        # DeBERTa / mDeBERTa-v3: deberta.embeddings, deberta.encoder.layer
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
        encoder_layers = model.deberta.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"  Layer {i}: trainable")
        # Keep pooler and classifier trainable
        if hasattr(model, 'pooler'):
            for param in model.pooler.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'electra'):
        # ELECTRA discriminator: electra.embeddings, electra.encoder.layer
        for param in model.electra.embeddings.parameters():
            param.requires_grad = False
        encoder_layers = model.electra.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"  Layer {i}: trainable")
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'convbert'):
        # ConvBERT: convbert.embeddings, convbert.encoder.layer
        for param in model.convbert.embeddings.parameters():
            param.requires_grad = False
        encoder_layers = model.convbert.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                logger.info(f"  Layer {i}: trainable")
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        logger.warning(f"Unknown model architecture ({model_type}), attempting generic layer freeze...")
        # Fallback: freeze first N children modules
        children = list(model.children())
        for i, child in enumerate(children):
            if i < num_frozen_layers:
                for param in child.parameters():
                    param.requires_grad = False
                logger.info(f"  Module {i} ({child.__class__.__name__}): FROZEN")
            else:
                logger.info(f"  Module {i} ({child.__class__.__name__}): trainable")

def main():
    parser = argparse.ArgumentParser(description="Unified Supervised Training script for Hallucination Verifier.")
    parser.add_argument("--domain", type=str, required=True, choices=["medical", "finance", "legal", "multidomain"],
                        help="Select training domain split.")
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-cased",
                        help="Pre-trained model name or path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model training and deterministic setup.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU/CPU.")
    parser.add_argument("--epochs", type=float, default=4.0, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument("--use_train_dev_final", action="store_true", help="Merge train and dev splits for final training.")
    parser.add_argument("--input_view", type=str, default="full_input",
                        choices=["full_input", "question_only", "claim_only", "question_claim", "context_only", "context_claim", "shuffled_context_claim"],
                        help="Input view configuration for shortcut ablation experiments.")
    parser.add_argument("--train_file", type=str, default=None, help="Explicit path override for training file.")
    parser.add_argument("--dev_file", type=str, default=None, help="Explicit path override for development file.")
    parser.add_argument("--test_file", type=str, default=None, help="Explicit path override for evaluation test file.")
    parser.add_argument("--peft_type", type=str, default="none", choices=["none", "lora"],
                        help="Select PEFT strategy (none, lora).")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="LoRA target modules. 'all-linear' (default) or explicit list like "
                             "'query,key,value,output.dense' for attention-only ablation.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate (default 0.05). Must be passed explicitly.")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of bottom BERT layers to freeze.")
    parser.add_argument("--label_set", type=str, default="4class",
                        choices=["5class", "4class"],
                        help="Label set: '4class' (current primary release, default) or '5class' (legacy v16).")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="Sinif sayisi (label_set'ten otomatik belirlenir, override icin kullanin).")
    parser.add_argument("--unv_loss_weight", type=float, default=1.0,
                        help="Loss weight multiplier for unverifiable class.")
    args = parser.parse_args()

    # ── Dinamik etiket haritasi ────────────────────────────────────────────────
    global LABEL2ID, ID2LABEL
    if args.label_set == "4class":
        LABEL2ID = LABEL2ID_4CLASS
    else:
        LABEL2ID = LABEL2ID_5CLASS
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    logger.info(f"Label set: {args.label_set}  ({len(LABEL2ID)} sinif: {list(LABEL2ID.keys())})")

    set_deterministic_seed(args.seed)

    # 1. Resolve dataset directory paths dynamically
    # Training data stays under data/primary/. New multidomain evaluations default to
    # data/evaluation/preview_v0.1/, while archived experiment runners explicitly use
    # data/evaluation/legacy_experiment_version/. Explicit overrides always take priority.
    primary = Path("data/primary")
    preview_eval = Path("data/evaluation/preview_v0.1")
    derived = Path("data/derived/train_domain")
    preview_domain = preview_eval / "domain"

    if args.train_file:
        train_file = args.train_file
    elif args.domain == "multidomain":
        train_file = primary / "train.jsonl"
    else:
        # Single-domain derived splits
        candidate = derived / f"train_1600_{args.domain}.jsonl"
        if candidate.exists():
            train_file = candidate
        else:
            # Fallback: warn and use multidomain train
            logger.warning(
                f"Domain split '{candidate}' not found. Falling back to multidomain "
                f"train.jsonl. Pass --train_file to specify an explicit path."
            )
            train_file = primary / "train.jsonl"

    if args.dev_file:
        dev_file = args.dev_file
    elif args.domain == "multidomain":
        dev_file = preview_eval / "dev_general.jsonl"
    else:
        candidate = preview_domain / f"dev960_general_{args.domain}.jsonl"
        if candidate.exists():
            dev_file = candidate
        else:
            logger.warning(
                f"Preview domain split '{candidate}' not found. Falling back to the full reviewed preview dev_general.jsonl."
            )
            dev_file = preview_eval / "dev_general.jsonl"

    logger.info(f"train_file resolved to: {Path(train_file).resolve()}")
    logger.info(f"dev_file   resolved to: {Path(dev_file).resolve()}")
    
    train_path = Path(train_file)
    dev_path = Path(dev_file)
    
    if not train_path.exists():
        logger.error(f"Training split file does not exist: {train_path}")
        sys.exit(1)
    if not dev_path.exists():
        logger.error(f"Development split file does not exist: {dev_path}")
        sys.exit(1)

    # Resolve test set paths to evaluate on
    # --test_file ile tek bir dosya verilmisse sadece "multidomain" olarak degerlendir
    if args.test_file:
        test_path = Path(args.test_file)
        if not test_path.exists():
            logger.error(f"Test file does not exist: {test_path}")
            sys.exit(1)
        test_domains = ["multidomain"]
        test_files = {"multidomain": test_path}
        logger.info(f"Test file override: {test_path}")
    else:
        test_domains = ["medical", "finance", "legal", "multidomain"]
        test_files = {
            "medical": preview_domain / "dev600_stress_medical.jsonl",
            "finance": preview_domain / "dev600_stress_finance.jsonl",
            "legal": preview_domain / "dev600_stress_legal.jsonl",
            "multidomain": preview_eval / "dev_stress.jsonl",
        }
        for test_domain, test_path in test_files.items():
            if not test_path.exists():
                logger.error(f"Test split file does not exist for {test_domain}: {test_path}")
                sys.exit(1)
            
    # 2. Load and Validate Raw Data
    is_multi = (args.domain == "multidomain")
    raw_train = load_jsonl_data(train_path)
    raw_dev = load_jsonl_data(dev_path)
    
    train_ctxs = validate_dataset(raw_train, "train", is_multidomain=is_multi)
    dev_ctxs = validate_dataset(raw_dev, "dev", is_multidomain=is_multi)
    
    detect_leakage(train_ctxs, dev_ctxs)
    
    # 3. Apply Shortcut Ablation / View configurations
    logger.info(f"Applying input view formatting: {args.input_view}")
    proc_train = apply_shortcut_ablation(raw_train, args.input_view, args.seed)
    proc_dev = apply_shortcut_ablation(raw_dev, args.input_view, args.seed)
    
    if args.use_train_dev_final:
        logger.info("Combining Train and Dev datasets for final training run (--use_train_dev_final).")
        proc_train = proc_train + proc_dev

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # LABEL_MAP must be defined before tokenize_dataset is called.
    # Previously this was assigned after the function definition (line 533), which
    # worked only due to Python closure late-binding — fragile and misleading.
    LABEL_MAP = LABEL2ID

    def tokenize_dataset(data: list[dict]) -> Dataset:
        # Form sequence inputs
        first_texts = [r["context"] for r in data]
        second_texts = [r["claim"] for r in data]
        labels = [LABEL_MAP[r["gold_label"]] for r in data] if "gold_label" in data[0] else None
        
        # Hugging Face tokenizers handle sequence pairs
        if second_texts[0] is None:
            tokenized = tokenizer(
                first_texts,
                truncation=True,
                max_length=args.max_length,
                padding="max_length"
            )
        else:
            tokenized = tokenizer(
                first_texts,
                second_texts,
                truncation=True,
                max_length=args.max_length,
                padding="max_length"
            )
            
        if labels is not None:
            tokenized["labels"] = labels
            
        return Dataset.from_dict(tokenized)
        
    logger.info("Tokenizing datasets...")
    train_ds = tokenize_dataset(proc_train)
    dev_ds = tokenize_dataset(proc_dev)
    
    # 4. Load Model
    logger.info(f"Loading model config and model: {args.model_name}")
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    # mDeBERTa-v3 (deberta-v2) must be loaded in FP32 — BF16 cast is handled by Trainer via bf16=True.
    # Loading directly in BF16/FP16 causes gradient instability with fresh classifier/pooler heads.
    _load_dtype = torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=model_config, torch_dtype=_load_dtype
    )
    
    if model.config.vocab_size < len(tokenizer):
        logger.warning(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)} to match tokenizer.")
        model.resize_token_embeddings(len(tokenizer))

    # Apply layer freezing if requested
    if args.freeze_layers > 0:
        model_type = model.config.model_type
        logger.info(f"Freezing bottom {args.freeze_layers} layers of {model_type} encoder...")
        freeze_encoder_layers(model, args.freeze_layers)

    # Apply LoRA if requested
    if args.peft_type == "lora":
        logger.info(f"Wrapping model with LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        from peft import LoraConfig, get_peft_model

        # LoRA task_type strategy:
        # - SEQ_CLS: correct for BERT, ELECTRA, ConvBERT, RoBERTa, etc.
        #   PEFT auto-registers ["classifier","score"] as modules_to_save.
        # - FEATURE_EXTRACTION + explicit exclude_modules: needed ONLY for DistilBERT
        #   because DistilBERT has a "pre_classifier" Linear that "all-linear" would
        #   also pick up as a LoRA target, causing a TypeError with SEQ_CLS auto-wrapping.
        # Using FEATURE_EXTRACTION with "all-linear" on BERT-family models causes the
        # classifier to appear in both target_modules and exclude_modules simultaneously,
        # which PEFT 0.19.1 resolves inconsistently → full class collapse at inference.
        _is_distilbert = "distilbert" in args.model_name.lower()
        _top_children = {name for name, _ in model.named_children()}
        _head_candidates = ("classifier", "pre_classifier", "score")

        # Parse target_modules: comma-separated list or special string "all-linear"
        _target_mods = args.lora_target_modules.strip()
        if "," in _target_mods:
            _target_mods = [m.strip() for m in _target_mods.split(",") if m.strip()]

        if _is_distilbert:
            # DistilBERT: FEATURE_EXTRACTION to avoid pre_classifier conflict
            _head_names = sorted(n for n in _head_candidates if n in _top_children)
            logger.info(f"LoRA (DistilBERT): task=FEATURE_EXTRACTION, exclude_modules={_head_names}, modules_to_save={_head_names}")
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=_target_mods,
                exclude_modules=_head_names if _head_names else None,
                modules_to_save=_head_names if _head_names else None,
            )
        else:
            # BERT, ELECTRA, ConvBERT, RoBERTa, mBERT, etc.: use SEQ_CLS
            # PEFT will automatically add ["classifier","score"] to modules_to_save.
            logger.info(f"LoRA (SEQ_CLS): target_modules={_target_mods}")
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=_target_mods,
            )
        model = get_peft_model(model, peft_config)

    # Calculate and log trainable parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / all_params
    logger.info(f"Trainable params: {trainable_params:,} / {all_params:,} ({pct:.2f}%)")

    # Setup directories
    model_base_name = args.model_name.split("/")[-1]
    experiment_dir = Path(args.output_dir) / model_base_name / args.domain / f"seed_{args.seed}"
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Outputs will be saved in: {experiment_dir}")

    # 5. Training Arguments
    # mDeBERTa-v3 (deberta-v2) does not support FP16 — use BF16 instead.
    # BF16 requires Ampere+ GPU (RTX 30xx / 40xx). FP16 is used for all other models.
    _model_type = model.config.model_type if not hasattr(model, 'base_model') else model.base_model.config.model_type
    _use_bf16 = _model_type in ("deberta-v2", "deberta") and torch.cuda.is_bf16_supported()
    _use_fp16 = torch.cuda.is_available() and not _use_bf16
    logger.info(f"Precision: fp16={_use_fp16}, bf16={_use_bf16} (model_type={_model_type})")

    training_args = TrainingArguments(
        output_dir=str(experiment_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.batch_size >= 8 else 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        fp16=_use_fp16,
        bf16=_use_bf16,
        optim="adamw_torch_fused",  # matches B1 training args; faster & same results as adamw_torch
        logging_dir=str(experiment_dir / "logs"),
        logging_steps=10,
        eval_strategy="no" if args.use_train_dev_final else "epoch",
        save_strategy="no" if args.use_train_dev_final else "epoch",
        save_total_limit=1,
        load_best_model_at_end=False if args.use_train_dev_final else True,
        metric_for_best_model="macro_f1" if not args.use_train_dev_final else None,
        greater_is_better=True if not args.use_train_dev_final else None,
        # Explicitly pass seed and data_seed so the Trainer's internal RNG
        # (data shuffling, dropout, etc.) is tied to the same seed as the
        # model initialisation and set_deterministic_seed() above.
        seed=args.seed,
        data_seed=args.seed,
        report_to="none"
    )

    class WeightedTrainer(Trainer):
        def __init__(self, *args_t, class_weights=None, **kwargs_t):
            super().__init__(*args_t, **kwargs_t)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    if args.unv_loss_weight != 1.0:
        c_weights = [1.0] * len(LABEL2ID)
        if "unverifiable" in LABEL2ID:
            c_weights[LABEL2ID["unverifiable"]] = args.unv_loss_weight
        logger.info(f"Using WeightedTrainer with class weights: {c_weights}")
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=None if args.use_train_dev_final else dev_ds,
            compute_metrics=compute_metrics,
            class_weights=c_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=None if args.use_train_dev_final else dev_ds,
            compute_metrics=compute_metrics
        )

    # 6. Run Training
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Track peak VRAM usage
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        logger.info(f"Peak GPU VRAM usage: {peak_mem:.2f} MB")
    else:
        peak_mem = 0.0
        logger.info("CUDA not available. Peak GPU VRAM tracking skipped.")
        
    # Save best model
    logger.info(f"Saving best verifier checkpoint/model to: {experiment_dir / 'best_model'}")
    trainer.save_model(str(experiment_dir / "best_model"))
    tokenizer.save_pretrained(str(experiment_dir / "best_model"))
    
    # Write configuration details
    config_details = {
        "domain": args.domain,
        "model_name": args.model_name,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "input_view": args.input_view,
        "use_train_dev_final": args.use_train_dev_final,
        "peft_type": args.peft_type,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "freeze_layers": args.freeze_layers,
        "trainable_pct": pct,
        "peak_gpu_memory_mb": peak_mem
    }
    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_details, f, indent=2, ensure_ascii=False)
        
    # Also write a separate peak memory metrics file
    with open(experiment_dir / "peak_memory.json", "w", encoding="utf-8") as f:
        json.dump({"peak_gpu_memory_mb": peak_mem}, f, indent=2)

    # 7. Evaluate on Dev
    if not args.use_train_dev_final:
        logger.info("Evaluating best model on development split...")
        dev_eval = trainer.evaluate()
        with open(experiment_dir / "dev_metrics.json", "w", encoding="utf-8") as f:
            json.dump(dev_eval, f, indent=2, ensure_ascii=False)
        logger.info(f"Dev Macro F1: {dev_eval.get('eval_macro_f1', 0.0):.4f}")
    else:
        # Dev metrics are skipped since dev was used in training
        with open(experiment_dir / "dev_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"info": "Skipped. Combined train+dev training was executed."}, f, indent=2)

    # 8. Evaluate on all Test splits
    logger.info("Starting evaluations on all Test splits...")
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Track classifications for prediction logs
    predictions_log = []
    
    for t_domain in test_domains:
        logger.info(f"Evaluating on {t_domain} test split...")
        raw_test = load_jsonl_data(test_files[t_domain])
        # Apply the exact same shortcut view configuration to the test dataset!
        proc_test = apply_shortcut_ablation(raw_test, args.input_view, args.seed)
        test_ds = tokenize_dataset(proc_test)
        
        # Predict
        preds_output = trainer.predict(test_ds)
        test_metrics = preds_output.metrics
        logits = preds_output.predictions
        preds = np.argmax(logits, axis=-1)
        labels = preds_output.label_ids
        
        # Format metrics and write to test_metrics_{domain}.json
        with open(experiment_dir / f"test_metrics_{t_domain}.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Test ({t_domain}) Macro F1: {test_metrics.get('test_macro_f1', test_metrics.get('eval_macro_f1', 0.0)):.4f}")
        
        # If it matches the training domain split, compute details like classification report and confusion matrix
        if t_domain == args.domain or (args.domain == "multidomain" and t_domain == "multidomain"):
            # Detailed classification report
            report = classification_report(
                labels,
                preds,
                target_names=list(LABEL2ID.keys()),
                output_dict=True,
                zero_division=0
            )
            with open(experiment_dir / "classification_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            # Confusion matrix csv
            cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL2ID))))
            cm_df = pd.DataFrame(cm, index=list(LABEL2ID.keys()), columns=list(LABEL2ID.keys()))
            cm_df.to_csv(experiment_dir / "confusion_matrix.csv")
            
        # Log prediction details
        for r, pred, label in zip(raw_test, preds, labels):
            predictions_log.append({
                "example_id": r["example_id"],
                "test_domain": t_domain,
                "context_id": r["context_id"],
                "question": r["question"],
                "claim": r["claim"],
                "gold_label": ID2LABEL[label],
                "pred_label": ID2LABEL[pred]
            })
            
    # Write predictions.jsonl
    with open(experiment_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for entry in predictions_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    # Save best checkpoint reference
    best_checkpoint_path = str((experiment_dir / "best_model").resolve())
    with open(experiment_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(best_checkpoint_path + "\n")
        
    logger.info(f"All evaluations complete. Unified results saved to: {experiment_dir}")

if __name__ == "__main__":
    main()