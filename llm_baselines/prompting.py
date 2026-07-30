from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import AMBIGUITY_FLAGS, LABELS, RELATIONS, read_jsonl

PROMPT_VERSION = "tr_factbench_llm_baseline_tr_v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = ROOT / "prompts" / "system_tr_v1.txt"
DEFAULT_FEWSHOT_FILE = ROOT / "prompts" / "fewshot_examples_tr_v1.jsonl"


def load_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT) -> str:
    return path.read_text(encoding="utf-8").strip()


def target_payload(row: dict[str, Any]) -> dict[str, str]:
    """Build the only user-visible target fields. IDs and labels are intentionally omitted."""
    return {
        "domain": str(row.get("domain", "")),
        "context": str(row.get("context", "")),
        "question": str(row.get("question", "")),
        "claim": str(row.get("claim", "")),
    }


def build_user_message(row: dict[str, Any]) -> str:
    return (
        "Aşağıdaki tek iddiayı yalnızca verilen bağlama göre değerlendir. "
        "İstenen JSON nesnesi dışında metin üretme.\n\n"
        + json.dumps(target_payload(row), ensure_ascii=False, indent=2)
    )


def canonical_assistant_message(demo: dict[str, Any]) -> str:
    output = {
        "label": demo["label"],
        "atomic_propositions": demo["atomic_propositions"],
        "brief_rationale": demo["brief_rationale"],
        "confidence": demo["confidence"],
        "ambiguity_flags": demo["ambiguity_flags"],
    }
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def validate_fewshot_examples(demos: list[dict[str, Any]], expected_count: int = 8) -> None:
    if len(demos) != expected_count:
        raise ValueError(f"Expected {expected_count} few-shot examples, found {len(demos)}")
    ids: set[str] = set()
    context_ids: set[str] = set()
    label_counts = {label: 0 for label in LABELS}
    for index, demo in enumerate(demos, start=1):
        for key in [
            "demo_id",
            "context_id",
            "domain",
            "context",
            "question",
            "claim",
            "label",
            "atomic_propositions",
            "brief_rationale",
            "confidence",
            "ambiguity_flags",
        ]:
            if key not in demo:
                raise ValueError(f"Few-shot example {index} is missing {key}")
        if demo["demo_id"] in ids:
            raise ValueError(f"Duplicate demo_id: {demo['demo_id']}")
        if demo["context_id"] in context_ids:
            raise ValueError(f"Few-shot examples must use distinct contexts: {demo['context_id']}")
        ids.add(demo["demo_id"])
        context_ids.add(demo["context_id"])
        label = demo["label"]
        if label not in LABELS:
            raise ValueError(f"Invalid few-shot label: {label}")
        label_counts[label] += 1
        confidence = demo["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid confidence in {demo['demo_id']}")
        atoms = demo["atomic_propositions"]
        if not isinstance(atoms, list) or not atoms:
            raise ValueError(f"Missing atoms in {demo['demo_id']}")
        for atom in atoms:
            if atom.get("relation") not in RELATIONS:
                raise ValueError(f"Invalid atom relation in {demo['demo_id']}")
        flags = demo["ambiguity_flags"]
        if not isinstance(flags, list) or any(flag not in AMBIGUITY_FLAGS for flag in flags):
            raise ValueError(f"Invalid ambiguity flags in {demo['demo_id']}")
    if any(count != 2 for count in label_counts.values()):
        raise ValueError(f"Few-shot set must contain exactly two examples per label: {label_counts}")


def load_fewshot_examples(path: Path = DEFAULT_FEWSHOT_FILE) -> list[dict[str, Any]]:
    demos = read_jsonl(path)
    validate_fewshot_examples(demos)
    return demos


def build_messages(
    row: dict[str, Any],
    prompt_mode: str,
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
    fewshot_path: Path = DEFAULT_FEWSHOT_FILE,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if prompt_mode not in {"zero_shot", "few_shot_8"}:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": load_system_prompt(system_prompt_path)}
    ]
    demos: list[dict[str, Any]] = []
    if prompt_mode == "few_shot_8":
        demos = load_fewshot_examples(fewshot_path)
        for demo in demos:
            messages.append({"role": "user", "content": build_user_message(demo)})
            messages.append({"role": "assistant", "content": canonical_assistant_message(demo)})
    messages.append({"role": "user", "content": build_user_message(row)})
    return messages, demos
