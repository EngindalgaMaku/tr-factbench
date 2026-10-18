"""Paired context-group bootstrap for gold-out gold-480 Gate 2.

Resamples the 120 source contexts with replacement, keeping the four sibling
claims together. Reports:

- percentile 95% CI for Macro-F1 of K1 and each frozen LLM condition
- paired Δ Macro-F1 (system − K1) on the same resampled groups
- paired Δ Macro-F1 (few_shot_8 − zero_shot) within each LLM

This is the LRE Gate 2 statistical comparison. It is not a seed-level test
and it is not the IAA bootstrap in scripts/42_compute_iaa.py.

Gold v1.0 has no context_id field; groups are formed from identical context
text. Expected: 120 groups × 4 claims = 480.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data/evaluation/gold_v1.0/gold/TR-FactBench_controlled480_GOLD_v1.0.jsonl"
K1_PATH = ROOT / "data/evaluation/gold_v1.0/verifier_hold/PREDICTIONS_v1.0.jsonl"
LLM_ROOT = ROOT / "results/llm_baselines/gold_v1.0"
OUT_DIR = LLM_ROOT / "bootstrap"
JSON_OUT = OUT_DIR / "gate2_paired_bootstrap.json"
CSV_OUT = OUT_DIR / "gate2_paired_bootstrap.csv"
MD_OUT = OUT_DIR / "gate2_paired_bootstrap.md"

LABELS = [
    "supported",
    "partially_supported",
    "contradicted",
    "unverifiable",
]
LABEL_TO_I = {label: i for i, label in enumerate(LABELS)}
N_LABELS = len(LABELS)
EXPECTED_EXAMPLES = 480
EXPECTED_GROUPS = 120
N_BOOTSTRAP = 20_000
SEED = 42
ALPHA = 0.05

SYSTEMS = [
    {
        "key": "k1_electra",
        "display": "K1 ELECTRA-TR LoRA (H1_DROP10_S45)",
        "kind": "encoder",
        "model": "H1_DROP10_S45",
        "prompt_mode": None,
        "path": K1_PATH,
        "id_field": "release_example_id",
    },
    {
        "key": "gpt_zs",
        "display": "GPT-4.1-mini zero-shot",
        "kind": "llm",
        "model": "openai/gpt-4.1-mini",
        "prompt_mode": "zero_shot",
        "path": LLM_ROOT / "openai_gpt-4.1-mini/zero_shot/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "gpt_8s",
        "display": "GPT-4.1-mini 8-shot",
        "kind": "llm",
        "model": "openai/gpt-4.1-mini",
        "prompt_mode": "few_shot_8",
        "path": LLM_ROOT / "openai_gpt-4.1-mini/few_shot_8/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "gemma_zs",
        "display": "Gemma-4-26B zero-shot",
        "kind": "llm",
        "model": "google/gemma-4-26b-a4b-it",
        "prompt_mode": "zero_shot",
        "path": LLM_ROOT / "google_gemma-4-26b-a4b-it/zero_shot/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "gemma_8s",
        "display": "Gemma-4-26B 8-shot",
        "kind": "llm",
        "model": "google/gemma-4-26b-a4b-it",
        "prompt_mode": "few_shot_8",
        "path": LLM_ROOT / "google_gemma-4-26b-a4b-it/few_shot_8/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "llama_zs",
        "display": "Llama-3.3-70B zero-shot",
        "kind": "llm",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "prompt_mode": "zero_shot",
        "path": LLM_ROOT / "meta-llama_llama-3.3-70b-instruct/zero_shot/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "llama_8s",
        "display": "Llama-3.3-70B 8-shot",
        "kind": "llm",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "prompt_mode": "few_shot_8",
        "path": LLM_ROOT / "meta-llama_llama-3.3-70b-instruct/few_shot_8/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "qwen_zs",
        "display": "Qwen-2.5-72B zero-shot",
        "kind": "llm",
        "model": "qwen/qwen-2.5-72b-instruct",
        "prompt_mode": "zero_shot",
        "path": LLM_ROOT / "qwen_qwen-2.5-72b-instruct/zero_shot/predictions.jsonl",
        "id_field": "example_id",
    },
    {
        "key": "qwen_8s",
        "display": "Qwen-2.5-72B 8-shot",
        "kind": "llm",
        "model": "qwen/qwen-2.5-72b-instruct",
        "prompt_mode": "few_shot_8",
        "path": LLM_ROOT / "qwen_qwen-2.5-72b-instruct/few_shot_8/predictions.jsonl",
        "id_field": "example_id",
    },
]

SHOT_PAIRS = [
    ("gpt_zs", "gpt_8s"),
    ("gemma_zs", "gemma_8s"),
    ("llama_zs", "llama_8s"),
    ("qwen_zs", "qwen_8s"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pred_map(spec: dict) -> dict[str, int]:
    rows = read_jsonl(spec["path"])
    out: dict[str, int] = {}
    for row in rows:
        eid = str(row.get(spec["id_field"]) or "")
        if not eid:
            raise SystemExit(f"Missing id in {spec['path']}")
        label = row.get("predicted_label")
        status = row.get("status")
        if spec["kind"] == "llm":
            if status != "success" or label not in LABEL_TO_I:
                raise SystemExit(
                    f"{spec['key']}: invalid/missing prediction for {eid} "
                    f"(status={status!r}, label={label!r})"
                )
        elif label not in LABEL_TO_I:
            raise SystemExit(f"{spec['key']}: invalid prediction for {eid}: {label!r}")
        if eid in out:
            raise SystemExit(f"{spec['key']}: duplicate id {eid}")
        out[eid] = LABEL_TO_I[str(label)]
    if len(out) != EXPECTED_EXAMPLES:
        raise SystemExit(f"{spec['key']}: expected {EXPECTED_EXAMPLES} preds, got {len(out)}")
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = y_true.astype(np.int64) * N_LABELS + y_pred.astype(np.int64)
    cm = np.bincount(counts, minlength=N_LABELS * N_LABELS).reshape(N_LABELS, N_LABELS)
    f1s = np.zeros(N_LABELS, dtype=np.float64)
    for i in range(N_LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[i] = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec / (prec + rec))
    return float(f1s.mean())


def percentile_ci(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    lo = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return hi < 0.0 or lo > 0.0


def main() -> None:
    gold_rows = read_jsonl(GOLD_PATH)
    if len(gold_rows) != EXPECTED_EXAMPLES:
        raise SystemExit(f"Expected {EXPECTED_EXAMPLES} gold rows, found {len(gold_rows)}")

    gold_ids: list[str] = []
    gold_y: list[int] = []
    gold_ctx: list[str] = []
    for row in gold_rows:
        eid = str(row.get("release_example_id") or row.get("example_id") or "")
        label = row.get("gold_label")
        context = row.get("context")
        if not eid or label not in LABEL_TO_I or not isinstance(context, str) or not context:
            raise SystemExit(f"Malformed gold row: {eid!r}")
        gold_ids.append(eid)
        gold_y.append(LABEL_TO_I[str(label)])
        gold_ctx.append(context)

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, context in enumerate(gold_ctx):
        grouped[context].append(idx)
    if len(grouped) != EXPECTED_GROUPS:
        raise SystemExit(f"Expected {EXPECTED_GROUPS} context groups, found {len(grouped)}")
    bad = {ctx[:40]: len(idxs) for ctx, idxs in grouped.items() if len(idxs) != 4}
    if bad:
        raise SystemExit(f"Non-4 claim groups: {bad}")

    group_order = sorted(grouped, key=lambda ctx: gold_ids[grouped[ctx][0]])
    group_indices = np.array([grouped[ctx] for ctx in group_order], dtype=np.int64)
    if group_indices.shape != (EXPECTED_GROUPS, 4):
        raise SystemExit(f"Unexpected group index shape: {group_indices.shape}")

    y_true = np.array(gold_y, dtype=np.int64)
    pred_by_system: dict[str, np.ndarray] = {}
    point_f1: dict[str, float] = {}
    for spec in SYSTEMS:
        pred_map = load_pred_map(spec)
        missing = [eid for eid in gold_ids if eid not in pred_map]
        extra = [eid for eid in pred_map if eid not in set(gold_ids)]
        if missing or extra:
            raise SystemExit(f"{spec['key']}: id mismatch missing={len(missing)} extra={len(extra)}")
        y_pred = np.array([pred_map[eid] for eid in gold_ids], dtype=np.int64)
        pred_by_system[spec["key"]] = y_pred
        point_f1[spec["key"]] = macro_f1(y_true, y_pred)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, EXPECTED_GROUPS, size=(N_BOOTSTRAP, EXPECTED_GROUPS), dtype=np.int64)
    example_draws = group_indices[draws].reshape(N_BOOTSTRAP, EXPECTED_EXAMPLES)

    boot_f1: dict[str, np.ndarray] = {}
    for key, y_pred in pred_by_system.items():
        scores = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for b in range(N_BOOTSTRAP):
            idx = example_draws[b]
            scores[b] = macro_f1(y_true[idx], y_pred[idx])
        boot_f1[key] = scores

    systems_out = []
    for spec in SYSTEMS:
        key = spec["key"]
        lo, hi = percentile_ci(boot_f1[key])
        systems_out.append(
            {
                "key": key,
                "display": spec["display"],
                "kind": spec["kind"],
                "model": spec["model"],
                "prompt_mode": spec["prompt_mode"],
                "n_valid": EXPECTED_EXAMPLES,
                "coverage": 1.0,
                "macro_f1": point_f1[key],
                "macro_f1_ci95": [lo, hi],
                "bootstrap_mean": float(boot_f1[key].mean()),
                "source": spec["path"].relative_to(ROOT).as_posix(),
                "source_sha256": sha256_file(spec["path"]),
            }
        )

    k1_scores = boot_f1["k1_electra"]
    k1_point = point_f1["k1_electra"]
    vs_k1 = []
    for spec in SYSTEMS:
        if spec["key"] == "k1_electra":
            continue
        key = spec["key"]
        delta_point = point_f1[key] - k1_point
        delta_boot = boot_f1[key] - k1_scores
        lo, hi = percentile_ci(delta_boot)
        vs_k1.append(
            {
                "system": key,
                "display": spec["display"],
                "delta_macro_f1": delta_point,
                "delta_ci95": [lo, hi],
                "bootstrap_mean_delta": float(delta_boot.mean()),
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
                "better_than_k1_if_ci_excludes_zero_and_delta_positive": bool(
                    ci_excludes_zero(lo, hi) and delta_point > 0
                ),
                "worse_than_k1_if_ci_excludes_zero_and_delta_negative": bool(
                    ci_excludes_zero(lo, hi) and delta_point < 0
                ),
            }
        )

    shot_deltas = []
    display_by_key = {spec["key"]: spec["display"] for spec in SYSTEMS}
    for zs_key, fs_key in SHOT_PAIRS:
        delta_point = point_f1[fs_key] - point_f1[zs_key]
        delta_boot = boot_f1[fs_key] - boot_f1[zs_key]
        lo, hi = percentile_ci(delta_boot)
        shot_deltas.append(
            {
                "zero_shot": zs_key,
                "few_shot_8": fs_key,
                "display": f"{display_by_key[fs_key]} − {display_by_key[zs_key]}",
                "delta_macro_f1": delta_point,
                "delta_ci95": [lo, hi],
                "bootstrap_mean_delta": float(delta_boot.mean()),
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
            }
        )

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "gold_v1.0",
        "gold_file": GOLD_PATH.relative_to(ROOT).as_posix(),
        "gold_sha256": sha256_file(GOLD_PATH),
        "n_examples": EXPECTED_EXAMPLES,
        "n_groups": EXPECTED_GROUPS,
        "group_size": 4,
        "grouping": "identical context text (gold_v1.0 has no context_id)",
        "metric": "macro_f1",
        "labels": LABELS,
        "resampling_unit": "context_group",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "ci": "percentile",
        "alpha": ALPHA,
        "delta_definition": "system_macro_f1 - k1_macro_f1",
        "reference_encoder": "k1_electra",
        "systems": systems_out,
        "paired_vs_k1": vs_k1,
        "paired_fewshot_minus_zeroshot": shot_deltas,
        "notes": [
            "Four sibling claims sharing a context are always resampled together.",
            "Invalid/missing LLM outputs are treated as fatal; all eight Gate 2 conditions have 480/480 coverage.",
            "K1 gold CI in v23 used the same grouping and 20,000 replicates; this run freezes seed=42.",
            "A CI that includes 0 is compatible with no paired Macro-F1 difference at 95%.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison",
                "system",
                "macro_f1",
                "delta_vs_k1",
                "ci95_lo",
                "ci95_hi",
                "ci_excludes_zero",
            ]
        )
        for row in systems_out:
            writer.writerow(
                [
                    "absolute_macro_f1",
                    row["key"],
                    f"{row['macro_f1']:.10f}",
                    "",
                    f"{row['macro_f1_ci95'][0]:.10f}",
                    f"{row['macro_f1_ci95'][1]:.10f}",
                    "",
                ]
            )
        for row in vs_k1:
            writer.writerow(
                [
                    "paired_delta_vs_k1",
                    row["system"],
                    "",
                    f"{row['delta_macro_f1']:.10f}",
                    f"{row['delta_ci95'][0]:.10f}",
                    f"{row['delta_ci95'][1]:.10f}",
                    str(row["ci_excludes_zero"]).lower(),
                ]
            )
        for row in shot_deltas:
            writer.writerow(
                [
                    "paired_fewshot_minus_zeroshot",
                    row["few_shot_8"],
                    "",
                    f"{row['delta_macro_f1']:.10f}",
                    f"{row['delta_ci95'][0]:.10f}",
                    f"{row['delta_ci95'][1]:.10f}",
                    str(row["ci_excludes_zero"]).lower(),
                ]
            )

    lines = [
        "# Gate 2 paired context-group bootstrap",
        "",
        f"- Dataset: gold_v1.0 (N={EXPECTED_EXAMPLES}, {EXPECTED_GROUPS} context groups × 4)",
        f"- Resampling unit: context group, with replacement",
        f"- Replicates: {N_BOOTSTRAP:,}; seed={SEED}; percentile 95% CI",
        "- Δ = system Macro-F1 − K1 Macro-F1",
        "",
        "## Absolute Macro-F1",
        "",
        "| System | Macro-F1 | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in systems_out:
        lines.append(
            f"| {row['display']} | {fmt(row['macro_f1'])} | "
            f"[{fmt(row['macro_f1_ci95'][0])}, {fmt(row['macro_f1_ci95'][1])}] |"
        )
    lines.extend(
        [
            "",
            "## Paired Δ vs K1",
            "",
            "| System | Δ Macro-F1 | 95% CI | 0 excluded |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in vs_k1:
        mark = "yes" if row["ci_excludes_zero"] else "no"
        lines.append(
            f"| {row['display']} | {fmt(row['delta_macro_f1'])} | "
            f"[{fmt(row['delta_ci95'][0])}, {fmt(row['delta_ci95'][1])}] | {mark} |"
        )
    lines.extend(
        [
            "",
            "## Paired 8-shot − zero-shot",
            "",
            "| Contrast | Δ Macro-F1 | 95% CI | 0 excluded |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in shot_deltas:
        mark = "yes" if row["ci_excludes_zero"] else "no"
        lines.append(
            f"| {row['display']} | {fmt(row['delta_macro_f1'])} | "
            f"[{fmt(row['delta_ci95'][0])}, {fmt(row['delta_ci95'][1])}] | {mark} |"
        )
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"K1 Macro-F1 = {point_f1['k1_electra']:.10f}")
    k1_lo, k1_hi = percentile_ci(k1_scores)
    print(f"K1 95% CI   = [{k1_lo:.10f}, {k1_hi:.10f}]")
    print()
    print(f"{'system':<12} {'F1':>8} {'CI_lo':>8} {'CI_hi':>8} {'dK1':>8} {'d_lo':>8} {'d_hi':>8} {'excl0':>5}")
    for spec, row in zip((s for s in SYSTEMS if s["key"] != "k1_electra"), vs_k1):
        key = spec["key"]
        lo, hi = percentile_ci(boot_f1[key])
        print(
            f"{key:<12} {point_f1[key]:8.4f} {lo:8.4f} {hi:8.4f} "
            f"{row['delta_macro_f1']:8.4f} {row['delta_ci95'][0]:8.4f} "
            f"{row['delta_ci95'][1]:8.4f} {str(row['ci_excludes_zero']):>5}"
        )
    print()
    print(f"Wrote {JSON_OUT.relative_to(ROOT).as_posix()}")
    print(f"Wrote {CSV_OUT.relative_to(ROOT).as_posix()}")
    print(f"Wrote {MD_OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
