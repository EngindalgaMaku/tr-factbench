from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_baseline import parse_json_content, validate_model_output  # noqa: E402


def valid_output() -> dict:
    return {
        "label": "supported",
        "atomic_propositions": [
            {"text": "Bir önerme.", "relation": "entailed", "evidence_quote": "kanıt"}
        ],
        "brief_rationale": "Bağlam destekliyor.",
        "confidence": 0.9,
        "ambiguity_flags": ["none"],
    }


def test_validate_model_output_accepts_valid_object() -> None:
    assert validate_model_output(valid_output())["label"] == "supported"


def test_validate_model_output_rejects_unknown_label() -> None:
    output = valid_output()
    output["label"] = "unknown"
    with pytest.raises(ValueError):
        validate_model_output(output)


def test_parser_accepts_fenced_json() -> None:
    content = '```json\n{"label":"supported"}\n```'
    assert parse_json_content(content)["label"] == "supported"
