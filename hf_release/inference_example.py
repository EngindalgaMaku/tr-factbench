"""
Inference example for engin-dalga/tr-factbench-electra-lora

Loads the LoRA adapter on top of the exact base-model revision used during training
and runs factuality verification on a short example.

Requirements:
    pip install torch transformers peft

Usage:
    python inference_example.py
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
BASE_MODEL_REVISION = "44ba696463e4d853078f3094cec76ba4e924cfd1"
ADAPTER_REPO = "engin-dalga/tr-factbench-electra-lora"   # Hugging Face Hub ID

LABEL_MAPPING = {
    0: "supported",
    1: "partially_supported",
    2: "contradicted",
    3: "unverifiable",
}

MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Example input ──────────────────────────────────────────────────────────────
EXAMPLE = {
    "context": (
        "Türkiye'de lisanslı hekimlerin tıp fakültesinden mezun olması zorunludur. "
        "Tıp fakülteleri en az altı yıllık eğitim vermektedir."
    ),
    "question": "Türkiye'de hekim olmak için ne kadar eğitim gereklidir?",
    "claim": "Türkiye'de hekim olmak için en az altı yıllık tıp eğitimi zorunludur.",
}


def load_model(adapter_path: str | None = None):
    """
    Load base model + LoRA adapter.

    Args:
        adapter_path: Local path to adapter directory (optional).
                      If None, loads from Hugging Face Hub.
    """
    print(f"Loading tokenizer from: {adapter_path or ADAPTER_REPO}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or ADAPTER_REPO,
        revision=None,  # adapter repo, not base model
    )

    print(f"Loading base model: {BASE_MODEL}@{BASE_MODEL_REVISION}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        num_labels=len(LABEL_MAPPING),
        ignore_mismatched_sizes=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path or ADAPTER_REPO}")
    model = PeftModel.from_pretrained(base_model, adapter_path or ADAPTER_REPO)
    model.eval().to(DEVICE)
    return model, tokenizer


def format_input(context: str, question: str, claim: str) -> tuple[str, str]:
    """Format context and question+claim into the two text_a / text_b strings used during training."""
    text_a = f"[CONTEXT] {context}"
    text_b = f"[QUESTION] {question} [CLAIM] {claim}"
    return text_a, text_b


def predict(model, tokenizer, context: str, question: str, claim: str) -> dict:
    text_a, text_b = format_input(context, question, claim)
    enc = tokenizer(
        text_a, text_b,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return {
        "predicted_label": LABEL_MAPPING[pred_id],
        "probabilities": {LABEL_MAPPING[i]: round(p, 4) for i, p in enumerate(probs)},
    }


def verify_classifier_head(model) -> None:
    """Quick sanity check: confirm the classifier head is inside the adapter (modules_to_save)."""
    classifier_params = [
        n for n, _ in model.named_parameters()
        if "classifier" in n or "modules_to_save" in n
    ]
    if not classifier_params:
        raise RuntimeError(
            "Classifier head not found in adapter parameters. "
            "The adapter may be incomplete — retrain with modules_to_save=['classifier', 'score']."
        )
    print(f"[OK] Classifier head found in adapter ({len(classifier_params)} parameters).")


if __name__ == "__main__":
    model, tokenizer = load_model()
    verify_classifier_head(model)

    result = predict(model, tokenizer, **EXAMPLE)

    print("\n── Inference result ──────────────────────────────────────────────────")
    print(f"Context : {EXAMPLE['context'][:80]}...")
    print(f"Question: {EXAMPLE['question']}")
    print(f"Claim   : {EXAMPLE['claim']}")
    print(f"\nPredicted label : {result['predicted_label']}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:<22}: {prob:.4f}")
