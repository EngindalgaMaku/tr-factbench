from __future__ import annotations

from typing import Any

from common import AMBIGUITY_FLAGS, LABELS, RELATIONS


def response_json_schema() -> dict[str, Any]:
    return {
        "name": "tr_factbench_single_claim_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": LABELS},
                "atomic_propositions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "relation": {"type": "string", "enum": RELATIONS},
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
                    "items": {"type": "string", "enum": AMBIGUITY_FLAGS},
                },
            },
            "required": [
                "label",
                "atomic_propositions",
                "brief_rationale",
                "confidence",
                "ambiguity_flags",
            ],
            "additionalProperties": False,
        },
    }
