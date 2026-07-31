"""Upload adapter to Hugging Face Hub."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
from huggingface_hub import HfApi
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "runs/ablations/H1_DROP10_S45_ABLATION_run/ckpt/electra-base-turkish-cased-discriminator/multidomain/seed_45/best_model"
HF_DIR = ROOT / "hf_release"
REPO_ID = "angeldust007/tr-factbench-electra-lora"

TOKEN = (os.environ.get("HF_WRITE_TOKEN") or "").strip() or None
api = HfApi(token=TOKEN)

print(f"Logged in as: {api.whoami()['name']}")

print(f"Creating repo: {REPO_ID}")
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
print(f"Repo ready: https://huggingface.co/{REPO_ID}")

uploads = [
    (ADAPTER_DIR / "adapter_model.safetensors", "adapter_model.safetensors"),
    (ADAPTER_DIR / "adapter_config.json",        "adapter_config.json"),
    (ADAPTER_DIR / "tokenizer.json",             "tokenizer.json"),
    (ADAPTER_DIR / "tokenizer_config.json",      "tokenizer_config.json"),
    (HF_DIR / "label_mapping.json",              "label_mapping.json"),
    (HF_DIR / "inference_example.py",            "inference_example.py"),
    (HF_DIR / "requirements.txt",               "requirements.txt"),
    (HF_DIR / "README.md",                       "README.md"),
]

for local, remote in uploads:
    print(f"Uploading {remote} ...", end="", flush=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(" done")

print(f"\nAll files uploaded to https://huggingface.co/{REPO_ID}")