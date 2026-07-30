"""Compare Dev600 candidate labels, LLM pre-annotations, and human review files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from annotation.common import LABELS, load_keyed_jsonl, read_jsonl, utc_now_iso, write_jsonl_atomic  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "evaluation" / "legacy_experiment_version" / "dev_stress.jsonl"
DEFAULT_PRE = ROOT / "results" / "annotation" / "dev600" / "llm_preannotations_normalized.jsonl"
DEFAULT_OUT = ROOT / "results" / "annotation" / "dev600" / "annotation_report.json"
DEFAULT_DISAGREEMENTS = ROOT / "results" / "annotation" / "dev600" / "annotation_disagreements.jsonl"


def args_parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--llm-preannotations", type=Path, default=DEFAULT_PRE)
    ap.add_argument("--candidate-review", type=Path, dest="candidate_review", help="Human-reviewed candidate-label refinement output")
    ap.add_argument("--assisted", type=Path, dest="assisted_legacy", help=argparse.SUPPRESS)
    ap.add_argument("--blind", type=Path, action="append", default=[], help="Blind human output; may be passed twice")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--disagreements", type=Path, default=DEFAULT_DISAGREEMENTS)
    return ap.parse_args()


def completed_labels(rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        eid: str(row["final_label"])
        for eid, row in rows.items()
        if row.get("review_status") == "final" and row.get("final_label") in LABELS
    }


def pair_stats(name: str, a: dict[str, str], b: dict[str, str], domains: dict[str, str]) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    y1 = [a[eid] for eid in ids]
    y2 = [b[eid] for eid in ids]
    result: dict[str, Any] = {
        "comparison": name,
        "n_common": len(ids),
        "n_a": len(a),
        "n_b": len(b),
        "percent_agreement": None,
        "cohen_kappa": None,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": None,
        "per_domain": {},
    }
    if not ids:
        return result
    result["percent_agreement"] = sum(x == y for x, y in zip(y1, y2)) / len(ids)
    result["cohen_kappa"] = cohen_kappa_score(y1, y2, labels=LABELS)
    result["confusion_matrix"] = confusion_matrix(y1, y2, labels=LABELS).tolist()
    result["classification_report_a_as_reference"] = classification_report(
        y1, y2, labels=LABELS, output_dict=True, zero_division=0
    )
    by_domain: dict[str, list[str]] = defaultdict(list)
    for eid in ids:
        by_domain[domains.get(eid, "unknown")].append(eid)
    for domain, domain_ids in sorted(by_domain.items()):
        da = [a[eid] for eid in domain_ids]
        db = [b[eid] for eid in domain_ids]
        result["per_domain"][domain] = {
            "n": len(domain_ids),
            "percent_agreement": sum(x == y for x, y in zip(da, db)) / len(domain_ids),
            "cohen_kappa": cohen_kappa_score(da, db, labels=LABELS),
        }
    return result


def annotator_ids(rows: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({str(r.get("annotator_id", "")) for r in rows.values() if r.get("annotator_id")})


def main() -> None:
    args = args_parse()
    dataset_rows = read_jsonl(args.dataset)
    candidate = {str(r["example_id"]): str(r["gold_label"]) for r in dataset_rows}
    domains = {str(r["example_id"]): str(r.get("domain", "")) for r in dataset_rows}
    context_ids = {str(r["example_id"]): str(r.get("context_id", "")) for r in dataset_rows}

    llm_rows = load_keyed_jsonl(args.llm_preannotations) if args.llm_preannotations.exists() else {}
    llm = {eid: str(r["llm_label"]) for eid, r in llm_rows.items() if r.get("llm_label") in LABELS}

    candidate_review_path = args.candidate_review or args.assisted_legacy
    review_rows = load_keyed_jsonl(candidate_review_path) if candidate_review_path else {}
    reviewed_candidate = completed_labels(review_rows)

    blind_row_sets = [load_keyed_jsonl(path) for path in args.blind]
    blind_sets = [completed_labels(rows) for rows in blind_row_sets]

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "dataset_examples": len(candidate),
        "labels": LABELS,
        "files": {
            "dataset": str(args.dataset),
            "llm_preannotations": str(args.llm_preannotations) if args.llm_preannotations.exists() else None,
            "candidate_review": str(candidate_review_path) if candidate_review_path else None,
            "blind": [str(x) for x in args.blind],
        },
        "completion": {
            "llm": len(llm),
            "candidate_review_final": len(reviewed_candidate),
            "candidate_review_total": len(review_rows),
            "blind_final": [len(x) for x in blind_sets],
            "blind_total_reviewed": [len(x) for x in blind_row_sets],
        },
        "comparisons": [],
        "methodological_warnings": [],
    }

    if llm:
        report["comparisons"].append(pair_stats("candidate_vs_llm", candidate, llm, domains))
    if reviewed_candidate:
        report["comparisons"].append(pair_stats("candidate_before_vs_candidate_after_review", candidate, reviewed_candidate, domains))
        if llm:
            report["comparisons"].append(pair_stats("llm_vs_candidate_after_review", llm, reviewed_candidate, domains))
            common = sorted(set(llm) & set(reviewed_candidate))
            report["candidate_refinement"] = {
                "independent_annotation": False,
                "n_common": len(common),
                "llm_suggestion_accepted": sum(llm[eid] == reviewed_candidate[eid] for eid in common),
                "llm_suggestion_corrected": sum(llm[eid] != reviewed_candidate[eid] for eid in common),
                "llm_acceptance_rate": (
                    sum(llm[eid] == reviewed_candidate[eid] for eid in common) / len(common) if common else None
                ),
                "correction_pairs": dict(Counter(f"{llm[eid]} -> {reviewed_candidate[eid]}" for eid in common if llm[eid] != reviewed_candidate[eid])),
            }

    for idx, blind in enumerate(blind_sets, start=1):
        report["comparisons"].append(pair_stats(f"candidate_vs_blind_{idx}", candidate, blind, domains))
        if reviewed_candidate:
            report["comparisons"].append(pair_stats(f"candidate_after_review_vs_blind_{idx}", reviewed_candidate, blind, domains))

    if len(blind_sets) >= 2:
        report["comparisons"].append(pair_stats("blind_1_vs_blind_2", blind_sets[0], blind_sets[1], domains))
        ids_1 = annotator_ids(blind_row_sets[0])
        ids_2 = annotator_ids(blind_row_sets[1])
        if set(ids_1) & set(ids_2):
            report["methodological_warnings"].append(
                "The two blind files share an annotator_id. This is not independent two-human IAA."
            )
        else:
            report["blind_human_iaa_status"] = "independent_annotator_ids_detected"

    if review_rows and blind_row_sets:
        review_ids = set(annotator_ids(review_rows))
        for idx, blind_rows in enumerate(blind_row_sets, start=1):
            overlap = review_ids & set(annotator_ids(blind_rows))
            if overlap:
                report["methodological_warnings"].append(
                    f"Candidate review and blind_{idx} share annotator_id(s) {sorted(overlap)}. "
                    "This comparison is descriptive and is not independent human-human IAA."
                )

    all_sources: dict[str, dict[str, str]] = {"candidate": candidate}
    if llm:
        all_sources["llm"] = llm
    if reviewed_candidate:
        all_sources["candidate_after_review"] = reviewed_candidate
    for idx, blind in enumerate(blind_sets, start=1):
        all_sources[f"blind_{idx}"] = blind

    disagreements = []
    common_ids = sorted(set.intersection(*(set(x) for x in all_sources.values()))) if all_sources else []
    for eid in common_ids:
        values = {name: labels[eid] for name, labels in all_sources.items()}
        if len(set(values.values())) > 1:
            disagreements.append(
                {
                    "example_id": eid,
                    "context_id": context_ids.get(eid),
                    "domain": domains.get(eid),
                    "labels": values,
                }
            )
    report["all_source_common_examples"] = len(common_ids)
    report["all_source_disagreement_count"] = len(disagreements)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl_atomic(args.disagreements, disagreements)
    print(json.dumps({"output": str(args.output), "completion": report["completion"], "warnings": report["methodological_warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
