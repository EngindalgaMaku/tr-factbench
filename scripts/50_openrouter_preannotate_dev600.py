"""Generate raw LLM pre-annotations for Dev600 through OpenRouter.

The script groups the 600 examples into 150 contexts and sends one request per
context. It never exposes the existing gold_label to the model and supports
resume, dry-run, strict JSON schema, retries, and append-only request logging.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import requests
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.common import (  # noqa: E402
    LABELS,
    append_jsonl,
    group_by_context,
    load_keyed_jsonl,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now_iso,
    validate_dev600,
    write_jsonl_atomic,
)
from annotation.prompting import (  # noqa: E402
    PROMPT_VERSION,
    annotation_json_schema,
    build_user_prompt,
    load_system_prompt,
)

DEFAULT_INPUT = ROOT / "data" / "evaluation" / "legacy_experiment_version" / "dev_stress.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "annotation" / "dev600" / "llm_preannotations_raw.jsonl"
DEFAULT_LOG = ROOT / "results" / "annotation" / "dev600" / "llm_request_log.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "annotation" / "dev600" / "llm_preannotation_manifest.json"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--request-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--http-referer", default="https://github.com/EngindalgaMaku/tr-factbench")
    parser.add_argument("--app-title", default="TR-FactBench Dev600 Annotation")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.25)
    parser.add_argument("--limit-contexts", type=int)
    parser.add_argument("--context-id", action="append", default=[])
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--zdr", action="store_true", help="Require zero-data-retention endpoints")
    parser.add_argument("--no-structured-fallback", action="store_true")
    parser.add_argument("--overwrite-context", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def request_json(payload: dict[str, Any], api_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": args.http_referer,
        "X-OpenRouter-Title": args.app_title,
    }
    try:
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON content: {response.text[:2000]}") from exc
    return data, response.headers.get("x-request-id", "")


def extract_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenRouter response: {json.dumps(response)[:2000]}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(texts)
    raise ValueError(f"Unsupported message content type: {type(content)!r}")


def validate_model_output(
    data: dict[str, Any], group: dict[str, Any], item_to_example: dict[str, str]
) -> list[dict[str, Any]]:
    if data.get("context_id") != group["context_id"]:
        raise ValueError(
            f"context_id mismatch: expected {group['context_id']!r}, got {data.get('context_id')!r}"
        )
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    expected_ids = set(item_to_example)
    actual_ids = {str(x.get("item_id", "")) for x in annotations if isinstance(x, dict)}
    if actual_ids != expected_ids or len(annotations) != len(expected_ids):
        raise ValueError(f"item_id mismatch: expected={sorted(expected_ids)}, got={sorted(actual_ids)}")

    validated: list[dict[str, Any]] = []
    for item in annotations:
        if not isinstance(item, dict):
            raise ValueError("Each annotation must be an object")
        label = item.get("label")
        if label not in LABELS:
            raise ValueError(f"Invalid label for {item.get('item_id')}: {label!r}")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid confidence for {item.get('item_id')}: {confidence!r}")
        atoms = item.get("atomic_propositions")
        if not isinstance(atoms, list) or not atoms:
            raise ValueError(f"Missing atomic_propositions for {item.get('item_id')}")
        for atom in atoms:
            if atom.get("relation") not in {"entailed", "contradicted", "not_in_context"}:
                raise ValueError(f"Invalid atom relation for {item.get('item_id')}")
        normalized = dict(item)
        normalized["example_id"] = item_to_example[item["item_id"]]
        normalized.pop("item_id", None)
        validated.append(normalized)
    return validated


def make_payload(
    group: dict[str, Any], args: argparse.Namespace, structured: bool
) -> tuple[dict[str, Any], str, str, dict[str, str]]:
    system_prompt = load_system_prompt()
    user_prompt, item_to_example = build_user_prompt(group)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
        "provider": {
            "require_parameters": structured,
            "data_collection": "deny",
        },
    }
    if args.zdr:
        payload["provider"]["zdr"] = True
    if structured:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": annotation_json_schema(list(item_to_example)),
        }
    return payload, system_prompt, user_prompt, item_to_example


def call_with_retries(group: dict[str, Any], api_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    structured_modes = [True] if args.no_structured_fallback else [True, False]
    for structured in structured_modes:
        payload, system_prompt, user_prompt, item_to_example = make_payload(group, args, structured)
        for attempt in range(1, args.max_retries + 1):
            started = time.monotonic()
            try:
                response, header_request_id = request_json(payload, api_key, args)
                content = extract_content(response)
                parsed = json.loads(content)
                annotations = validate_model_output(parsed, group, item_to_example)
                meta = {
                    "structured_output": structured,
                    "attempt": attempt,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "request_id": response.get("id") or header_request_id,
                    "provider": response.get("provider"),
                    "usage": response.get("usage", {}),
                    "system_prompt_sha256": sha256_text(system_prompt),
                    "user_prompt_sha256": sha256_text(user_prompt),
                    "raw_content_sha256": sha256_text(content),
                }
                return {"context_id": parsed["context_id"], "annotations": annotations}, meta
            except Exception as exc:  # retry network, provider, parsing, and schema failures
                last_error = exc
                if attempt < args.max_retries:
                    delay = args.retry_base_seconds * (2 ** (attempt - 1)) + random.random()
                    print(
                        f"  [retry] context={group['context_id']} structured={structured} "
                        f"attempt={attempt} error={exc} sleep={delay:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
        print(
            f"  [fallback] structured={structured} failed for context={group['context_id']}: {last_error}",
            file=sys.stderr,
        )
    raise RuntimeError(f"All attempts failed for context={group['context_id']}: {last_error}")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    counts = validate_dev600(rows)
    groups = group_by_context(rows)
    if args.context_id:
        wanted = set(args.context_id)
        groups = [g for g in groups if g["context_id"] in wanted]
        missing = wanted - {g["context_id"] for g in groups}
        if missing:
            raise SystemExit(f"Unknown context_id values: {sorted(missing)}")
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(groups)
    if args.limit_contexts is not None:
        groups = groups[: args.limit_contexts]

    existing_by_example = load_keyed_jsonl(args.output)
    overwrite_contexts = set(args.overwrite_context)
    if overwrite_contexts:
        existing_by_example = {
            eid: row for eid, row in existing_by_example.items() if row.get("context_id") not in overwrite_contexts
        }
        write_jsonl_atomic(args.output, existing_by_example.values())

    completed_contexts = {row.get("context_id") for row in existing_by_example.values()}
    pending = [g for g in groups if g["context_id"] not in completed_contexts]

    system_prompt = load_system_prompt()
    manifest = {
        "created_at_utc": utc_now_iso(),
        "input_file": str(args.input),
        "input_sha256": sha256_file(args.input),
        "dataset_examples": counts["examples"],
        "dataset_contexts": counts["contexts"],
        "selected_contexts": len(groups),
        "already_completed_contexts": len(groups) - len(pending),
        "pending_contexts": len(pending),
        "model": args.model,
        "endpoint": ENDPOINT,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": sha256_text(system_prompt),
        "structured_output_preferred": True,
        "structured_fallback_enabled": not args.no_structured_fallback,
        "provider_data_collection": "deny",
        "zdr_required": args.zdr,
        "gold_label_exposed_to_model": False,
        "status": "dry_run" if args.dry_run else "running",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in ["dataset_examples", "dataset_contexts", "selected_contexts", "pending_contexts", "model"]}, ensure_ascii=False, indent=2))

    if args.dry_run:
        preview = pending[0] if pending else groups[0]
        payload, _, _, _ = make_payload(preview, args, structured=True)
        print("\n--- DRY-RUN REQUEST PREVIEW ---")
        printable = dict(payload)
        print(json.dumps(printable, ensure_ascii=False, indent=2)[:12000])
        return

    load_dotenv(ROOT / ".env")
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env} in the environment; do not put it in source files."
        )

    output_by_example = dict(existing_by_example)
    failed_contexts: list[dict[str, str]] = []
    for index, group in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {group['context_id']} ({group['domain']})")
        request_started = utc_now_iso()
        try:
            parsed, meta = call_with_retries(group, api_key, args)
            created_at = utc_now_iso()
            for item in parsed["annotations"]:
                record = {
                    "example_id": item["example_id"],
                    "context_id": group["context_id"],
                    "domain": group["domain"],
                    "llm_label": item["label"],
                    "atomic_propositions": item["atomic_propositions"],
                    "brief_rationale": item["brief_rationale"],
                    "confidence": float(item["confidence"]),
                    "ambiguity_flags": item["ambiguity_flags"],
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "request_id": meta.get("request_id"),
                    "provider": meta.get("provider"),
                    "structured_output": meta.get("structured_output"),
                    "created_at_utc": created_at,
                }
                output_by_example[item["example_id"]] = record
            write_jsonl_atomic(args.output, sorted(output_by_example.values(), key=lambda x: x["example_id"]))
            append_jsonl(
                args.request_log,
                {
                    "context_id": group["context_id"],
                    "domain": group["domain"],
                    "started_at_utc": request_started,
                    "completed_at_utc": created_at,
                    "status": "success",
                    **meta,
                },
            )
        except Exception as exc:
            failed_contexts.append({"context_id": group["context_id"], "error": str(exc)})
            append_jsonl(
                args.request_log,
                {
                    "context_id": group["context_id"],
                    "domain": group["domain"],
                    "started_at_utc": request_started,
                    "completed_at_utc": utc_now_iso(),
                    "status": "failed",
                    "error": str(exc),
                },
            )
            print(f"  [FAILED] {exc}", file=sys.stderr)
        time.sleep(args.sleep_between_requests)

    completed_contexts_final = {row.get("context_id") for row in output_by_example.values()}
    manifest.update(
        {
            "completed_at_utc": utc_now_iso(),
            "completed_examples": len(output_by_example),
            "completed_contexts": len(completed_contexts_final),
            "failed_contexts": failed_contexts,
            "status": "complete" if len(completed_contexts_final) == counts["contexts"] and not failed_contexts else "partial",
            "output_file": str(args.output),
            "output_sha256": sha256_file(args.output) if args.output.exists() else None,
        }
    )
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "completed_examples", "completed_contexts", "failed_contexts"]}, ensure_ascii=False, indent=2))
    if failed_contexts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
