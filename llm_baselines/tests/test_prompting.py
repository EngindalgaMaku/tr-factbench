from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import LABELS, read_jsonl  # noqa: E402
from prompting import DEFAULT_FEWSHOT_FILE, build_messages, build_user_message, validate_fewshot_examples  # noqa: E402


def test_target_prompt_does_not_leak_ids_or_gold() -> None:
    row = {
        "example_id": "LEAK_SENTINEL_EXAMPLE",
        "context_id": "LEAK_SENTINEL_CONTEXT",
        "domain": "legal",
        "context": "Bağlam metni.",
        "question": "Soru?",
        "claim": "İddia.",
        "gold_label": "LEAK_SENTINEL_GOLD",
    }
    prompt = build_user_message(row)
    assert "LEAK_SENTINEL_EXAMPLE" not in prompt
    assert "LEAK_SENTINEL_CONTEXT" not in prompt
    assert "LEAK_SENTINEL_GOLD" not in prompt
    assert "Bağlam metni." in prompt
    assert "İddia." in prompt


def test_fewshot_set_is_balanced_and_context_unique() -> None:
    demos = read_jsonl(DEFAULT_FEWSHOT_FILE)
    validate_fewshot_examples(demos)
    assert Counter(demo["label"] for demo in demos) == Counter({label: 2 for label in LABELS})
    assert len({demo["context_id"] for demo in demos}) == 8


def test_zero_shot_contains_no_assistant_demonstrations() -> None:
    row = {"domain": "medical", "context": "C", "question": "Q", "claim": "K"}
    messages, demos = build_messages(row, "zero_shot")
    assert demos == []
    assert [message["role"] for message in messages] == ["system", "user"]


def test_few_shot_contains_eight_pairs() -> None:
    row = {"domain": "medical", "context": "C", "question": "Q", "claim": "K"}
    messages, demos = build_messages(row, "few_shot_8")
    assert len(demos) == 8
    assert len(messages) == 18
    for index in range(1, 17, 2):
        assert messages[index]["role"] == "user"
        assert messages[index + 1]["role"] == "assistant"
        json.loads(messages[index + 1]["content"])
    assert messages[-1]["role"] == "user"
