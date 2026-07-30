"""Generate model_manifest.json for run directories that are missing one.

Reads software versions from the current environment. Only adds the manifest
to runs that don't already have one — never overwrites.
"""
from __future__ import annotations
import json, sys, subprocess, platform
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch, transformers
import peft as _peft_mod

try:
    import sklearn
    sklearn_ver = sklearn.__version__
except Exception:
    sklearn_ver = "unknown"

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "completed"

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    git_commit = "unknown"

def get_model_revision(model_name: str) -> str:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        return getattr(cfg, "_commit_hash", "unknown") or "unknown"
    except Exception:
        return "unknown"

# Cache revision lookups
_rev_cache: dict[str, str] = {}

added = 0
skipped = 0
for run_dir in sorted(RUNS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    manifest_path = run_dir / "model_manifest.json"
    if manifest_path.exists():
        skipped += 1
        continue
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"  [SKIP] no config.json: {run_dir.name}")
        continue
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    model_name = config.get("model_name", "unknown")
    if model_name not in _rev_cache:
        _rev_cache[model_name] = get_model_revision(model_name)
    manifest = {
        "run_id": run_dir.name,
        "base_model": model_name,
        "model_revision_sha": _rev_cache[model_name],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "transformers_version": transformers.__version__,
        "peft_version": _peft_mod.__version__,
        "sklearn_version": sklearn_ver,
        "git_commit": git_commit,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "note": "Retroactively generated — reflects current env, not necessarily the env used for training.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    added += 1
    print(f"  [ADD] {run_dir.name}")

print(f"\n[OK] Added {added} model_manifest.json files, skipped {skipped} (already had one).")