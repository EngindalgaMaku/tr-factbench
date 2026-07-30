from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    AMBIGUITY_FLAGS,
    LABELS,
    RELATIONS,
    append_jsonl,
    load_keyed_jsonl,
    read_jsonl,
    safe_slug,
    sha256_file,
    sha256_text,
    utc_now_iso,
    validate_dataset,
    write_jsonl_atomic,
)
from prompting import (  # noqa: E402
    DEFAULT_FEWSHOT_FILE,
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_messages,
)
from schemas import response_json_schema  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"



def repo_relative(path: Path) -> str:
    """Return a portable POSIX path when the file is inside the repository."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an independent single-claim LLM baseline through OpenRouter.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "evaluation" / "preview_v0.1" / "dev_stress.jsonl")
    parser.add_argument("--dataset-name")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-alias")
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot_8"], default="zero_shot")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--fewshot-file", type=Path, default=DEFAULT_FEWSHOT_FILE)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "llm_baselines")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--http-referer", default="https://github.com/EngindalgaMaku/tr-factbench")
    parser.add_argument("--app-title", default="TR-FactBench LLM Baselines")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example-id", action="append", default=[])
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--overwrite-example", action="append", default=[])
    parser.add_argument("--retry-failed", action="store_true", help="Re-run examples with status='failed'")
    parser.add_argument("--zdr", action="store_true")
    parser.add_argument("--no-structured-fallback", action="store_true")
    parser.add_argument("--free-tier", action="store_true",
                        help="Skip provider routing block (required for :free models on OpenRouter)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_json_content(content: str) -> dict[str, Any]:
    import re as _re
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object")
        return parsed
    except json.JSONDecodeError:
        pass
    # Try raw_decode (partial JSON recovery)
    start = stripped.find("{")
    if start >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(stripped[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Last resort: regex-extract label from truncated JSON
    m = _re.search(r'"label"\s*:\s*"([^"]+)"', stripped)
    if m and m.group(1) in LABELS:
        # Build a minimal valid object with just the label
        evidence = ""
        eq = _re.search(r'"evidence_quote"\s*:\s*"((?:[^"\\]|\\.)*)', stripped)
        if eq:
            evidence = eq.group(1)
        reason = ""
        for key in ("brief_rationale", "reason"):
            rq = _re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)', stripped)
            if rq:
                reason = rq.group(1)
                break
        if not reason:
            reason = f"[truncated response, label={m.group(1)}]"
        conf_m = _re.search(r'"confidence"\s*:\s*([0-9.]+)', stripped)
        conf = float(conf_m.group(1)) if conf_m else 0.5
        return {
            "label": m.group(1),
            "evidence_quote": evidence,
            "brief_rationale": reason,
            "confidence": conf,
            "ambiguity_flags": ["none"],
        }
    raise json.JSONDecodeError("Cannot parse JSON content", stripped, 0)


def validate_model_output(data: dict[str, Any]) -> dict[str, Any]:
    label = data.get("label")
    if label not in LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    # --- atomic_propositions: optional fallback for simplified schemas (e.g. Qwen) ---
    atoms = data.get("atomic_propositions")
    normalized_atoms: list[dict[str, str]] = []
    if isinstance(atoms, list) and atoms:
        for atom in atoms:
            if not isinstance(atom, dict):
                raise ValueError("Each atomic proposition must be an object")
            relation = atom.get("relation")
            if relation not in RELATIONS:
                raise ValueError(f"Invalid atom relation: {relation!r}")
            text = str(atom.get("text", "")).strip()
            if not text:
                raise ValueError("Atomic proposition text cannot be empty")
            normalized_atoms.append(
                {
                    "text": text,
                    "relation": relation,
                    "evidence_quote": str(atom.get("evidence_quote", "")).strip(),
                }
            )
    else:
        # Simplified schema: synthesize a single atom from evidence_quote
        evidence = str(data.get("evidence_quote", "")).strip()
        normalized_atoms = [{"text": evidence or label, "relation": label, "evidence_quote": evidence}]

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError(f"Invalid confidence: {confidence!r}")

    # --- ambiguity_flags: accept string "none" or list ---
    flags = data.get("ambiguity_flags")
    if isinstance(flags, str):
        flags = [flags] if flags in AMBIGUITY_FLAGS else ["none"]
    if not isinstance(flags, list) or any(flag not in AMBIGUITY_FLAGS for flag in flags):
        flags = ["none"]

    # --- brief_rationale: accept "reason" as fallback ---
    rationale = str(data.get("brief_rationale") or data.get("reason", "")).strip()
    if not rationale:
        raise ValueError("brief_rationale cannot be empty")

    return {
        "label": label,
        "atomic_propositions": normalized_atoms,
        "brief_rationale": rationale,
        "confidence": float(confidence),
        "ambiguity_flags": flags,
    }


def extract_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenRouter response: {json.dumps(response)[:2000]}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError(f"Unsupported message content type: {type(content)!r}")


def request_json(payload: dict[str, Any], api_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": args.http_referer,
        "X-OpenRouter-Title": args.app_title,
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"OpenRouter returned non-JSON content: {response.text[:2000]}") from exc
    return body, response.headers.get("x-request-id", "")


def make_payload(row: dict[str, Any], args: argparse.Namespace, structured: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    messages, demos = build_messages(row, args.prompt_mode, args.system_prompt, args.fewshot_file)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
        **(
            {}
            if args.free_tier
            else {
                "provider": {
                    # Always require chat/completions-capable providers.
                    # When structured=True we also send json_schema; when False we omit it
                    # but still need a chat endpoint (not text-completion only).
                    "require_parameters": True,
                    "data_collection": "deny",
                }
            }
        ),
    }
    if args.zdr and not args.free_tier:
        payload["provider"]["zdr"] = True
    if structured:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": response_json_schema(),
        }
    hashes = {
        "system_prompt_sha256": sha256_text(messages[0]["content"]),
        "target_user_prompt_sha256": sha256_text(messages[-1]["content"]),
        "messages_sha256": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "fewshot_count": len(demos),
    }
    return payload, hashes


def call_with_retries(row: dict[str, Any], api_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    structured_modes = [True] if args.no_structured_fallback else [True, False]
    for structured in structured_modes:
        payload, hashes = make_payload(row, args, structured)
        for attempt in range(1, args.max_retries + 1):
            started = time.monotonic()
            try:
                response, header_request_id = request_json(payload, api_key, args)
                content = extract_content(response)
                parsed = validate_model_output(parse_json_content(content))
                meta = {
                    "structured_output": structured,
                    "attempt": attempt,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "request_id": response.get("id") or header_request_id,
                    "provider": response.get("provider"),
                    "resolved_model": response.get("model"),
                    "usage": response.get("usage", {}),
                    "raw_content_sha256": sha256_text(content),
                    **hashes,
                }
                return parsed, meta, {"response": response, "raw_content": content}
            except json.JSONDecodeError as exc:
                # JSON parse errors are deterministic (same truncated output) — skip retries, go to fallback
                last_error = exc
                print(
                    f"  [json-fallback] structured={structured} JSON parse failed for example={row['example_id']}: {exc}",
                    file=sys.stderr,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    delay = args.retry_base_seconds * (2 ** (attempt - 1)) + random.random()
                    print(
                        f"  [retry] example={row['example_id']} structured={structured} "
                        f"attempt={attempt} error={exc} sleep={delay:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
        print(
            f"  [fallback] structured={structured} failed for example={row['example_id']}: {last_error}",
            file=sys.stderr,
        )
    raise RuntimeError(f"All attempts failed for example={row['example_id']}: {last_error}")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    validate_dataset(rows, require_gold=False)
    if args.example_id:
        wanted = set(args.example_id)
        rows = [row for row in rows if str(row["example_id"]) in wanted]
        missing = wanted - {str(row["example_id"]) for row in rows}
        if missing:
            raise SystemExit(f"Unknown example_id values: {sorted(missing)}")
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    dataset_name = args.dataset_name or args.input.stem
    model_alias = args.model_alias or safe_slug(args.model)
    run_dir = args.output_root / safe_slug(dataset_name) / safe_slug(model_alias) / args.prompt_mode
    predictions_path = run_dir / "predictions.jsonl"
    raw_path = run_dir / "raw_responses.jsonl"
    request_log_path = run_dir / "request_log.jsonl"
    manifest_path = run_dir / "manifest.json"

    existing = load_keyed_jsonl(predictions_path)
    overwrite = set(args.overwrite_example)
    if overwrite:
        existing = {key: value for key, value in existing.items() if key not in overwrite}
        write_jsonl_atomic(predictions_path, existing.values())
    if args.retry_failed:
        failed_ids = {key for key, val in existing.items() if val.get("status") == "failed"}
        existing = {key: value for key, value in existing.items() if key not in failed_ids}
        write_jsonl_atomic(predictions_path, existing.values())
    pending = [row for row in rows if str(row["example_id"]) not in existing]

    manifest: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "status": "dry_run" if args.dry_run else "running",
        "input_file": repo_relative(args.input),
        "input_sha256": sha256_file(args.input),
        "dataset_name": dataset_name,
        "selected_examples": len(rows),
        "already_completed_examples": len(rows) - len(pending),
        "pending_examples": len(pending),
        "requested_model": args.model,
        "model_alias": model_alias,
        "endpoint": ENDPOINT,
        "prompt_mode": args.prompt_mode,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_file": repo_relative(args.system_prompt),
        "system_prompt_sha256": sha256_file(args.system_prompt),
        "fewshot_file": str(args.fewshot_file) if args.prompt_mode == "few_shot_8" else None,
        "fewshot_sha256": sha256_file(args.fewshot_file) if args.prompt_mode == "few_shot_8" else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "one_claim_per_request": True,
        "gold_label_exposed_to_model": False,
        "example_id_exposed_to_model": False,
        "context_id_exposed_to_model": False,
        "provider_data_collection": "not_sent" if args.free_tier else "deny",
        "free_tier_mode": args.free_tier,
        "zdr_required": args.zdr,
        "structured_output_preferred": True,
        "structured_fallback_enabled": not args.no_structured_fallback,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["dataset_name", "selected_examples", "pending_examples", "requested_model", "prompt_mode"]}, ensure_ascii=False, indent=2))

    if args.dry_run:
        if not rows:
            raise SystemExit("No examples selected")
        payload, hashes = make_payload(rows[0], args, structured=True)
        print("\n--- DRY-RUN REQUEST PREVIEW ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:30000])
        print("\n--- HASHES ---")
        print(json.dumps(hashes, ensure_ascii=False, indent=2))
        return

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(ROOT / ".env")
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} in the environment or .env file.")

    output_by_id = dict(existing)
    failures: list[dict[str, str]] = []
    for index, row in enumerate(pending, start=1):
        example_id = str(row["example_id"])
        print(f"[{index}/{len(pending)}] {example_id} ({row.get('domain', '')})")
        started_at = utc_now_iso()
        try:
            parsed, meta, raw = call_with_retries(row, api_key, args)
            completed_at = utc_now_iso()
            usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
            record = {
                "example_id": example_id,
                "context_id": row.get("context_id"),
                "domain": row.get("domain"),
                "status": "success",
                "parse_success": True,
                "predicted_label": parsed["label"],
                "atomic_propositions": parsed["atomic_propositions"],
                "brief_rationale": parsed["brief_rationale"],
                "confidence": parsed["confidence"],
                "ambiguity_flags": parsed["ambiguity_flags"],
                "requested_model": args.model,
                "resolved_model": meta.get("resolved_model"),
                "model_alias": model_alias,
                "provider": meta.get("provider"),
                "prompt_mode": args.prompt_mode,
                "prompt_version": PROMPT_VERSION,
                "structured_output": meta.get("structured_output"),
                "request_id": meta.get("request_id"),
                "latency_seconds": meta.get("latency_seconds"),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reported_cost": usage.get("cost"),
                "created_at_utc": completed_at,
            }
            output_by_id[example_id] = record
            write_jsonl_atomic(predictions_path, sorted(output_by_id.values(), key=lambda x: x["example_id"]))
            append_jsonl(
                raw_path,
                {
                    "example_id": example_id,
                    "created_at_utc": completed_at,
                    "requested_model": args.model,
                    "prompt_mode": args.prompt_mode,
                    "request_id": meta.get("request_id"),
                    "raw_content": raw["raw_content"],
                    "response": raw["response"],
                },
            )
            append_jsonl(
                request_log_path,
                {
                    "example_id": example_id,
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                    "status": "success",
                    **meta,
                },
            )
        except Exception as exc:
            completed_at = utc_now_iso()
            error = str(exc)
            failures.append({"example_id": example_id, "error": error})
            output_by_id[example_id] = {
                "example_id": example_id,
                "context_id": row.get("context_id"),
                "domain": row.get("domain"),
                "status": "failed",
                "parse_success": False,
                "predicted_label": None,
                "error": error,
                "requested_model": args.model,
                "model_alias": model_alias,
                "prompt_mode": args.prompt_mode,
                "prompt_version": PROMPT_VERSION,
                "created_at_utc": completed_at,
            }
            write_jsonl_atomic(predictions_path, sorted(output_by_id.values(), key=lambda x: x["example_id"]))
            append_jsonl(
                request_log_path,
                {
                    "example_id": example_id,
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                    "status": "failed",
                    "error": error,
                },
            )
            print(f"  [FAILED] {error}", file=sys.stderr)
        time.sleep(args.sleep_between_requests)

    selected_ids = {str(row["example_id"]) for row in rows}
    selected_outputs = [value for key, value in output_by_id.items() if key in selected_ids]
    success_count = sum(row.get("status") == "success" for row in selected_outputs)
    manifest.update(
        {
            "completed_at_utc": utc_now_iso(),
            "completed_examples": len(selected_outputs),
            "successful_examples": success_count,
            "failed_examples": len(selected_outputs) - success_count,
            "failures": failures,
            "status": "complete" if len(selected_outputs) == len(rows) and success_count == len(rows) else "partial",
            "predictions_file": repo_relative(predictions_path),
            "predictions_sha256": sha256_file(predictions_path) if predictions_path.exists() else None,
            "raw_responses_file": repo_relative(raw_path),
            "request_log_file": repo_relative(request_log_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "completed_examples", "successful_examples", "failed_examples"]}, ensure_ascii=False, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
