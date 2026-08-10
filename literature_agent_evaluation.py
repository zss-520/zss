"""Deterministic evaluation of literature retrieval and meeting screening quality."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import deep_research_literature_agent as literature


ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY = ROOT / "data" / "literature_deep_research_memory.json"
DEFAULT_GOLD = ROOT / "data" / "literature_agent_gold_labels.json"
DEFAULT_JSON = ROOT / "data" / "literature_agent_evaluation.json"
DEFAULT_MD = ROOT / "data" / "literature_agent_evaluation.md"
DEFAULT_CSV = ROOT / "data" / "literature_meeting_screening_decisions.csv"
EVALUATION_VERSION = "1.0"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _percent(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float | None:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return round(((tp * tn) - (fp * fn)) / denominator, 6) if denominator else None


def _key(value: Any) -> str:
    return literature.normalize_key(literature.canonicalize_model_name(value))


def _index_memory_models(memory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in ["all_candidate_models", "models", "benchmark_ready_models"]:
        rows.extend(row for row in literature.ensure_list(memory.get(section)) if isinstance(row, dict))
    return {_key(row.get("model_name") or row.get("canonical_name")): row for row in literature.dedupe_models_by_name(rows)}


def _same_value(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None or literature.is_missing_value(actual)
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except Exception:
            return False
    return literature.normalize_key(actual) == literature.normalize_key(expected)


def _decision_reason(row: dict[str, Any], accepted: bool) -> str:
    if accepted:
        return "passed_strict_main_amp_deployment_gate"
    if row.get("benchmark_candidate") is False:
        return "benchmark_candidate_false"
    if row.get("deployment_eligible") is False:
        return "deployment_eligible_false"
    issues = [str(x) for x in literature.ensure_list(row.get("blocking_issues")) if str(x).strip()]
    if issues:
        return "; ".join(issues)
    name = literature.normalize_key(row.get("model_name") or row.get("canonical_name"))
    if name == "amp":
        return "ambiguous_generic_model_identity"
    return "failed_strict_main_amp_deployment_gate"


def _screening_bucket(row: dict[str, Any]) -> tuple[str, str]:
    """Classify a retrieved candidate without conflating relevance and deployability."""
    if literature._strict_main_deployment_candidate(row):
        return "valid_main_benchmark_candidate", "passed_strict_main_amp_deployment_gate"

    name = literature.normalize_key(row.get("model_name") or row.get("canonical_name"))
    task_text = literature.normalize_key(" ".join([
        str(row.get("task_type") or ""), str(row.get("method_family") or ""),
        str(row.get("paper_title") or ""), str(row.get("source_title") or ""),
        str(row.get("candidate_reason") or ""),
    ]))
    issue_text = literature.normalize_key(" ".join(
        str(x) for x in literature.ensure_list(row.get("blocking_issues"))
    ))
    scope_text = literature.normalize_key(row.get("scope_status"))
    combined = " ".join([task_text, issue_text, scope_text])

    if name in {"amp", "camp", "dbaasp", "campr3", "apd3", "dramp", "dbamp", "ampsphere"}:
        return "misretrieval_or_out_of_scope", "ambiguous_model_identity_or_database_platform"
    explicit_scope_terms = [
        "out of scope", "task mismatch", "not amp", "non amp",
        "minor histocompatibility", "hla binding", "histocompatibility antigen",
    ]
    if any(term in combined for term in explicit_scope_terms):
        return "misretrieval_or_out_of_scope", "explicit_scope_or_task_exclusion"
    non_main_task_terms = [
        "mic regression", "minimum inhibitory concentration", "regression task",
        "antifungal", "anti fungal", "antiviral", "anti viral",
        "anticancer", "anti cancer", "toxicity", "hemolysis",
        "generation", "generator", "generative", "peptide design",
    ]
    if any(term in task_text for term in non_main_task_terms):
        return "misretrieval_or_out_of_scope", "non_binary_or_secondary_peptide_task"

    is_main_task = literature._is_main_amp_binary_candidate(row)
    if is_main_task:
        if not literature.model_has_real_evidence(row):
            return "relevant_but_not_deployable", "insufficient_verifiable_model_evidence"
        if not literature.has_code_repository_url(row):
            return "relevant_but_not_deployable", "missing_code_repository"
        return "relevant_but_not_deployable", "failed_deployment_readiness_gate"

    if any(term in task_text for term in ["antimicrobial peptide", "antibacterial peptide", "amp prediction", "amp classification"]):
        return "manual_review_required", "amp_relevance_present_but_binary_task_identity_unclear"
    if task_text:
        return "misretrieval_or_out_of_scope", "no_main_amp_binary_task_evidence"
    return "manual_review_required", "insufficient_task_metadata"


def _build_screening_census(
    index: dict[str, dict[str, Any]], final_names: set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    counts = {
        "valid_main_benchmark_candidate": 0,
        "misretrieval_or_out_of_scope": 0,
        "relevant_but_not_deployable": 0,
        "manual_review_required": 0,
    }
    for row in index.values():
        bucket, reason = _screening_bucket(row)
        counts[bucket] += 1
        name = str(row.get("model_name") or row.get("canonical_name") or "")
        decisions.append({
            "model_name": name,
            "meeting_decision": "accept" if bucket == "valid_main_benchmark_candidate" else "reject_or_hold",
            "screening_bucket": bucket,
            "decision_reason": reason,
            "selected_in_final_deployment": _key(name) in final_names,
            "task_type": row.get("task_type"),
            "source_doi": row.get("source_doi"),
            "source_pmid": row.get("source_pmid"),
            "code_repository_url": row.get("code_repository_url"),
            "benchmark_candidate": row.get("benchmark_candidate"),
            "deployment_eligible": row.get("deployment_eligible"),
            "blocking_issues": literature.ensure_list(row.get("blocking_issues")),
        })
    decisions.sort(key=lambda item: (item["screening_bucket"], literature.normalize_key(item["model_name"])))
    total = len(decisions)
    valid = counts["valid_main_benchmark_candidate"]
    rejected = total - valid
    census = {
        "total_unique_models_retrieved": total,
        "meeting_valid_models": valid,
        "meeting_valid_ratio": _ratio(valid, total),
        "meeting_valid_percent": _percent(valid, total),
        "meeting_rejected_or_held_models": rejected,
        "meeting_rejected_or_held_ratio": _ratio(rejected, total),
        "meeting_rejected_or_held_percent": _percent(rejected, total),
        "meeting_misretrieval_or_out_of_scope_models": counts["misretrieval_or_out_of_scope"],
        "meeting_misretrieval_or_out_of_scope_ratio": _ratio(counts["misretrieval_or_out_of_scope"], total),
        "meeting_misretrieval_or_out_of_scope_percent": _percent(counts["misretrieval_or_out_of_scope"], total),
        "relevant_but_not_deployable_models": counts["relevant_but_not_deployable"],
        "relevant_but_not_deployable_ratio": _ratio(counts["relevant_but_not_deployable"], total),
        "relevant_but_not_deployable_percent": _percent(counts["relevant_but_not_deployable"], total),
        "manual_review_required_models": counts["manual_review_required"],
        "manual_review_required_ratio": _ratio(counts["manual_review_required"], total),
        "manual_review_required_percent": _percent(counts["manual_review_required"], total),
        "final_deployment_models": len(final_names),
    }
    return census, decisions


def evaluate_literature_agent(
    memory_path: str | Path = DEFAULT_MEMORY,
    gold_path: str | Path = DEFAULT_GOLD,
) -> dict[str, Any]:
    memory_path = Path(memory_path)
    gold_path = Path(gold_path)
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    gold_doc = json.loads(gold_path.read_text(encoding="utf-8"))
    labels = [row for row in gold_doc.get("labels", []) if isinstance(row, dict)]
    index = _index_memory_models(memory)
    final_names = {
        _key(row.get("model_name") or row.get("canonical_name"))
        for row in literature.ensure_list(memory.get("final_deployment_models"))
        if isinstance(row, dict)
    }
    screening_census, screening_decisions = _build_screening_census(index, final_names)

    details: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    valid_total = invalid_total = valid_retrieved = invalid_retrieved = 0
    invalid_rejected = invalid_selected = valid_selected = 0
    rejected_with_reason = 0
    metadata_fields = metadata_correct = 0

    for gold in labels:
        name = str(gold.get("model_name") or "")
        gold_valid = gold.get("gold_label") == "eligible_main_amp_binary"
        valid_total += int(gold_valid)
        invalid_total += int(not gold_valid)
        row = index.get(_key(name))
        retrieved = row is not None
        accepted = bool(retrieved and literature._strict_main_deployment_candidate(row))
        selected = _key(name) in final_names
        reason = _decision_reason(row or {"model_name": name}, accepted) if retrieved else "not_retrieved"

        if gold_valid:
            valid_retrieved += int(retrieved)
            if retrieved and accepted:
                tp += 1
            else:
                fn += 1
            valid_selected += int(selected)
        else:
            invalid_retrieved += int(retrieved)
            if retrieved and accepted:
                fp += 1
            else:
                tn += 1
            invalid_rejected += int(retrieved and not accepted)
            invalid_selected += int(selected)
            rejected_with_reason += int(retrieved and not accepted and reason not in {"", "failed_strict_main_amp_deployment_gate"})

        expected_metadata = gold.get("expected_primary_metadata") or {}
        field_results: dict[str, bool] = {}
        for field, expected in expected_metadata.items():
            metadata_fields += 1
            correct = bool(retrieved and _same_value((row or {}).get(field), expected))
            metadata_correct += int(correct)
            field_results[field] = correct

        details.append({
            "model_name": name,
            "gold_label": gold.get("gold_label"),
            "retrieved": retrieved,
            "meeting_gate_decision": "accept" if accepted else "reject",
            "decision_correct": accepted == gold_valid,
            "decision_reason": reason,
            "selected_in_final_deployment": selected,
            "metadata_field_checks": field_results,
            "adjudication_reason": gold.get("adjudication_reason"),
        })

    retrieved_total = valid_retrieved + invalid_retrieved
    audited_selected = valid_selected + invalid_selected
    final_total = len(final_names)
    metrics = {
        "gold_models": len(labels),
        "gold_valid_models": valid_total,
        "gold_invalid_models": invalid_total,
        "audited_models_retrieved": retrieved_total,
        "valid_model_retrieval_recall": _ratio(valid_retrieved, valid_total),
        "audited_retrieval_precision": _ratio(valid_retrieved, retrieved_total),
        "valid_model_retention_rate": _ratio(tp, tp + fn),
        "wrong_model_detection_recall": _ratio(tn, tn + fp),
        "wrong_model_leakage_rate": _ratio(fp, tn + fp),
        "meeting_screen_precision": _ratio(tp, tp + fp),
        "meeting_screen_accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "meeting_screen_mcc": _mcc(tp, tn, fp, fn),
        "discussion_filter_yield": _ratio(invalid_rejected, retrieved_total),
        "exclusion_reason_traceability": _ratio(rejected_with_reason, invalid_rejected),
        "primary_metadata_field_accuracy": _ratio(metadata_correct, metadata_fields),
        "valid_gold_models_selected_in_final": valid_selected,
        "final_audited_precision": _ratio(valid_selected, audited_selected),
        "final_deployment_contamination_rate": _ratio(invalid_selected, final_total),
        "final_deployment_invalid_gold_count": invalid_selected,
        "final_deployment_total": final_total,
    }
    return {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_scope": "literature retrieval + meeting screening + final deployment protection",
        "attribution_status": "current_state_proxy_without_pre_discussion_snapshot",
        "attribution_note": (
            "These metrics evaluate the current end-to-end literature pipeline. Strict causal measurement of "
            "meeting value-added requires immutable pre-discussion and post-discussion snapshots for each run."
        ),
        "memory_path": str(memory_path),
        "gold_path": str(gold_path),
        "meeting_screening_census": screening_census,
        "screening_decisions": screening_decisions,
        "confusion_matrix": {"tp_valid_accepted": tp, "tn_invalid_rejected": tn, "fp_invalid_accepted": fp, "fn_valid_rejected_or_missing": fn},
        "metrics": metrics,
        "models": details,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    m = report["metrics"]
    census = report["meeting_screening_census"]
    lines = [
        "# Literature Meeting Agent Evaluation", "",
        f"- Evaluation version: {report['evaluation_version']}",
        f"- Attribution status: `{report['attribution_status']}`",
        f"- Gold models: {m['gold_models']} (valid={m['gold_valid_models']}, invalid={m['gold_invalid_models']})",
        "",
        "## Full meeting screening census", "",
        "The following counts are meeting/gate decisions over all unique retrieved model identities; only the independent gold subset is an externally audited accuracy estimate.", "",
        "| Category | Count | Ratio of retrieved |", "|---|---:|---:|",
        f"| Total unique models retrieved | {census['total_unique_models_retrieved']} | 1.0 |",
        f"| Valid main benchmark candidates | {census['meeting_valid_models']} | {census['meeting_valid_percent']}% |",
        f"| Rejected or held in total | {census['meeting_rejected_or_held_models']} | {census['meeting_rejected_or_held_percent']}% |",
        f"| Misretrieval or out of scope | {census['meeting_misretrieval_or_out_of_scope_models']} | {census['meeting_misretrieval_or_out_of_scope_percent']}% |",
        f"| AMP-relevant but not deployable | {census['relevant_but_not_deployable_models']} | {census['relevant_but_not_deployable_percent']}% |",
        f"| Manual review required | {census['manual_review_required_models']} | {census['manual_review_required_percent']}% |",
        "",
        "## Core metrics", "",
        "| Metric | Value |", "|---|---:|",
    ]
    for key in [
        "valid_model_retrieval_recall", "valid_model_retention_rate",
        "wrong_model_detection_recall", "wrong_model_leakage_rate",
        "meeting_screen_precision", "meeting_screen_accuracy", "meeting_screen_mcc",
        "discussion_filter_yield", "exclusion_reason_traceability",
        "primary_metadata_field_accuracy", "final_audited_precision",
        "final_deployment_contamination_rate",
    ]:
        lines.append(f"| {key} | {m.get(key)} |")
    lines.extend(["", "## Audited decisions", "", "| Model | Gold | Retrieved | Decision | Correct | Final | Reason |", "|---|---|---:|---|---:|---:|---|"])
    for row in report["models"]:
        lines.append(
            f"| {row['model_name']} | {row['gold_label']} | {row['retrieved']} | "
            f"{row['meeting_gate_decision']} | {row['decision_correct']} | "
            f"{row['selected_in_final_deployment']} | {row['decision_reason']} |"
        )
    lines.extend(["", "## Full per-model meeting decisions", "", "| Model | Decision | Category | Final | Reason |", "|---|---|---|---:|---|"])
    for row in report["screening_decisions"]:
        lines.append(
            f"| {row['model_name']} | {row['meeting_decision']} | {row['screening_bucket']} | "
            f"{row['selected_in_final_deployment']} | {row['decision_reason']} |"
        )
    lines.extend(["", "> " + report["attribution_note"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate literature-search and meeting-screening agent quality")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()
    report = evaluate_literature_agent(args.memory, args.gold)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "model_name", "meeting_decision", "screening_bucket", "decision_reason",
        "selected_in_final_deployment", "task_type", "source_doi", "source_pmid",
        "code_repository_url", "benchmark_candidate", "deployment_eligible", "blocking_issues",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in report["screening_decisions"]:
            csv_row = dict(row)
            csv_row["blocking_issues"] = "; ".join(str(x) for x in csv_row.get("blocking_issues", []))
            writer.writerow({field: csv_row.get(field) for field in csv_fields})
    print(json.dumps(report["meeting_screening_census"], ensure_ascii=False, indent=2))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_json}, {output_md}, and {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
