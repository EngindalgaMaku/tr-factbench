from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from annotation.common import LABELS

PROMPT_VERSION = "dev600_openrouter_llama33_v1_1"
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "prompts" / "dev600_system_tr_v1.txt"


def load_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT) -> str:
    return path.read_text(encoding="utf-8").strip()


def anonymized_items(group: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return shuffled opaque item IDs so neither ID suffix nor source order leaks labels."""
    examples = list(group["examples"])
    seed_bytes = hashlib.sha256(
        f"{PROMPT_VERSION}|{group['context_id']}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(seed_bytes[:8], "big")).shuffle(examples)
    items: list[dict[str, Any]] = []
    item_to_example: dict[str, str] = {}
    for index, row in enumerate(examples, start=1):
        item_id = f"item_{index}"
        item_to_example[item_id] = str(row["example_id"])
        items.append(
            {
                "item_id": item_id,
                "question": row.get("question", ""),
                "claim": row.get("claim", ""),
            }
        )
    return items, item_to_example


def build_user_prompt(group: dict[str, Any]) -> tuple[str, dict[str, str]]:
    claims, item_to_example = anonymized_items(group)
    payload = {
        "context_id": group["context_id"],
        "domain": group.get("domain", ""),
        "context": group["context"],
        "items": claims,
    }
    return (
        "Aşağıdaki tek bağlam grubundaki bütün iddiaları değerlendir. "
        "Her item_id için tam bir anotasyon döndür.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    ), item_to_example


def annotation_json_schema(expected_item_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "tr_factbench_context_annotations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "context_id": {"type": "string"},
                "annotations": {
                    "type": "array",
                    "minItems": len(expected_item_ids),
                    "maxItems": len(expected_item_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string", "enum": expected_item_ids},
                            "label": {"type": "string", "enum": LABELS},
                            "atomic_propositions": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "relation": {
                                            "type": "string",
                                            "enum": ["entailed", "contradicted", "not_in_context"],
                                        },
                                        "evidence_quote": {"type": "string"},
                                    },
                                    "required": ["text", "relation", "evidence_quote"],
                                    "additionalProperties": False,
                                },
                            },
                            "brief_rationale": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "ambiguity_flags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "multi_interpretation",
                                        "weak_contradiction",
                                        "scope_mismatch",
                                        "temporal_ambiguity",
                                        "numerical_ambiguity",
                                        "terminology_ambiguity",
                                        "none",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "item_id",
                            "label",
                            "atomic_propositions",
                            "brief_rationale",
                            "confidence",
                            "ambiguity_flags",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["context_id", "annotations"],
            "additionalProperties": False,
        },
    }
