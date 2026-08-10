"""Generate blinded Literature Evidence Agent proposals for the 51 weight meetings.

This script consumes only the deliberately blinded evidence bundle.  It never reads
model identities, scores, or rankings.  The prose generated here is an auditable
local Agent proposal: ``llm_prior`` statements are explicitly marked and are not
represented as newly retrieved literature.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "data/results_manual/codex_agent_weight_meeting/agent_evidence_bundle.json"
OUTPUT_PATH = ROOT / "data/results_manual/codex_agent_weight_meeting/literature_agent_proposals.json"

MIN_WEIGHT = 0.005
MAX_WEIGHT = 0.35
MAX_L1 = 0.30


# Literature-memory anchor.  Non-primary metrics retain non-zero mass because the
# meeting contract requires all twelve dimensions and because discrimination,
# threshold behaviour, and calibration are complementary rather than substitutes.
INITIAL_WEIGHTS = {
    "ACC": 0.015,
    "AUPRC": 0.250,
    "AUROC": 0.075,
    "BalancedAccuracy": 0.060,
    "BrierScore": 0.040,
    "ECE": 0.030,
    "F1-Score": 0.055,
    "MCC": 0.190,
    "NPV": 0.025,
    "Precision": 0.100,
    "Recall": 0.130,
    "Specificity": 0.030,
}

# Agent-local interpretation of the six anonymous evidence fields.  This is an
# explicitly labelled llm_prior, not a claimed published coefficient set.
EVIDENCE_FIELD_WEIGHTS = {
    "coverage": 0.08,
    "separation": 0.20,
    "consistency": 0.18,
    "consensus": 0.18,
    "uniqueness": 0.11,
    "committee_support": 0.25,
}


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Project a positive vector onto the bounded simplex deterministically."""
    keys = list(weights)
    values = {key: max(MIN_WEIGHT, min(MAX_WEIGHT, float(weights[key]))) for key in keys}
    # Iteratively distribute residual only over dimensions that can move.
    for _ in range(100):
        residual = 1.0 - sum(values.values())
        if abs(residual) < 1e-12:
            break
        if residual > 0:
            movable = [key for key in keys if values[key] < MAX_WEIGHT - 1e-15]
            capacity = sum(MAX_WEIGHT - values[key] for key in movable)
            if not movable or capacity <= 0:
                raise ValueError("bounded simplex has no positive capacity")
            for key in movable:
                values[key] += residual * (MAX_WEIGHT - values[key]) / capacity
        else:
            movable = [key for key in keys if values[key] > MIN_WEIGHT + 1e-15]
            capacity = sum(values[key] - MIN_WEIGHT for key in movable)
            if not movable or capacity <= 0:
                raise ValueError("bounded simplex has no negative capacity")
            for key in movable:
                values[key] += residual * (values[key] - MIN_WEIGHT) / capacity
        values = {key: max(MIN_WEIGHT, min(MAX_WEIGHT, value)) for key, value in values.items()}
    # Stable rounding with exact residual correction.
    values = {key: round(value, 9) for key, value in values.items()}
    residual = round(1.0 - sum(values.values()), 9)
    if residual:
        candidates = sorted(keys, key=lambda key: values[key], reverse=(residual < 0))
        for key in candidates:
            candidate = values[key] + residual
            if MIN_WEIGHT <= candidate <= MAX_WEIGHT:
                values[key] = round(candidate, 9)
                break
    return values


def _l1(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(abs(left[key] - right[key]) for key in left)


def _field_score(evidence: dict[str, float]) -> float:
    return sum(EVIDENCE_FIELD_WEIGHTS[field] * float(evidence[field]) for field in EVIDENCE_FIELD_WEIGHTS)


def _dataset_adjustments(sampled: list[str]) -> dict[str, float]:
    """Small transparent task-context adjustment, never model-outcome feedback."""
    counts = {name: sampled.count(name) for name in ("Dataset_A", "Dataset_B", "Dataset_C")}
    n = max(1, len(sampled))
    severe_imbalance = counts["Dataset_A"] / n
    balanced = (counts["Dataset_B"] + counts["Dataset_C"]) / n
    return {
        "ACC": -0.10 * severe_imbalance + 0.02 * balanced,
        "AUPRC": 0.11 * severe_imbalance,
        "AUROC": 0.025 * balanced - 0.02 * severe_imbalance,
        "BalancedAccuracy": 0.045 * severe_imbalance + 0.02 * balanced,
        "BrierScore": 0.035 * severe_imbalance + 0.015 * balanced,
        "ECE": 0.035 * severe_imbalance + 0.015 * balanced,
        "F1-Score": 0.025 * balanced,
        "MCC": 0.075 * severe_imbalance + 0.025 * balanced,
        "NPV": -0.015 * severe_imbalance + 0.015 * balanced,
        "Precision": 0.055 * severe_imbalance + 0.015 * balanced,
        "Recall": 0.060 * severe_imbalance + 0.020 * balanced,
        "Specificity": 0.030 * severe_imbalance + 0.015 * balanced,
    }


def _evidence_target(
    metrics: list[str], metric_evidence: dict[str, dict[str, float]], sampled: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    raw_scores = {metric: _field_score(metric_evidence[metric]) for metric in metrics}
    adjustments = _dataset_adjustments(sampled)
    adjusted = {
        metric: max(1e-9, raw_scores[metric] * (1.0 + adjustments[metric])) for metric in metrics
    }
    total = sum(adjusted.values())
    normalized_evidence = {metric: adjusted[metric] / total for metric in metrics}

    # 78% literature/task anchor + 22% round-specific anonymous evidence.  The
    # modest evidence share prevents bootstrap noise from rewriting endpoint intent.
    target = {
        metric: 0.78 * INITIAL_WEIGHTS[metric] + 0.22 * normalized_evidence[metric]
        for metric in metrics
    }
    return _normalize(target), raw_scores


def _proposal(previous: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    # This Agent updates only 35% of the way toward each round target.  The explicit
    # cap is retained as a guard even though this conservative step is usually far below it.
    alpha = 0.35
    proposed = _normalize(
        {metric: (1.0 - alpha) * previous[metric] + alpha * target[metric] for metric in previous}
    )
    change = _l1(previous, proposed)
    if change > MAX_L1:
        shrink = MAX_L1 / change
        proposed = _normalize(
            {
                metric: previous[metric] + shrink * (proposed[metric] - previous[metric])
                for metric in previous
            }
        )
    return proposed


def _top_items(values: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:n]


def _bottom_items(values: dict[str, float], n: int = 2) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: (item[1], item[0]))[:n]


def _fmt(items: list[tuple[str, float]]) -> str:
    return ", ".join(f"{metric}={value:.3f}" for metric, value in items)


def _confidence(raw_scores: dict[str, float], metric_evidence: dict[str, dict[str, float]]) -> float:
    values = list(raw_scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    mean_coverage = sum(float(v["coverage"]) for v in metric_evidence.values()) / len(metric_evidence)
    # Confidence is capped because independence/homology gates remain unresolved.
    return round(min(0.82, max(0.55, 0.62 + 0.18 * mean_coverage - 0.30 * math.sqrt(variance))), 3)


def _validate_group(weights: dict[str, float], metrics: list[str], label: str) -> None:
    if set(weights) != set(metrics):
        raise ValueError(f"{label}: metric set mismatch")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError(f"{label}: sum is {sum(weights.values())}")
    for metric, weight in weights.items():
        if not MIN_WEIGHT - 1e-12 <= weight <= MAX_WEIGHT + 1e-12:
            raise ValueError(f"{label}: {metric}={weight} violates bounds")


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    metrics = list(bundle["eligible_metrics"])
    if set(metrics) != set(INITIAL_WEIGHTS):
        raise ValueError("INITIAL_WEIGHTS does not match the blinded metric contract")
    if len(bundle["rounds"]) != 50:
        raise ValueError(f"expected 50 rounds, found {len(bundle['rounds'])}")

    initial_weights = _normalize({metric: INITIAL_WEIGHTS[metric] for metric in metrics})
    _validate_group(initial_weights, metrics, "initial")

    result: dict[str, Any] = {
        "role": "literature_agent",
        "role_mandate": (
            "Protect endpoint intent using the supplied literature-memory consensus; interpret only "
            "anonymous metric evidence and dataset prevalence, never model identity, score, or rank."
        ),
        "policy": {
            "information_boundary": (
                "Reads only agent_evidence_bundle.json. No model identities, predictions, scores, "
                "leaderboards, or downstream winner information are accessed."
            ),
            "source_policy": (
                "literature_consensus is treated as supplied project memory. Statements labelled "
                "llm_prior are this Agent's methodological judgement, not new external citations."
            ),
            "weight_policy": (
                "All 12 metrics remain present; weights are constrained to [0.005, 0.35], sum to 1, "
                "and each round is smoothed from this role's preceding proposal with L1 <= 0.30."
            ),
            "update_rule": (
                "Round target = 0.78 literature/task anchor + 0.22 normalized anonymous evidence; "
                "proposal = 0.65 previous role proposal + 0.35 current target."
            ),
        },
        "initial": {
            "analysis": (
                "The supplied project literature memory identifies AUPRC as the primary endpoint for "
                "strong imbalance, MCC as an all-confusion-cell robust summary, and Recall/Precision as "
                "the missed-candidate versus wet-lab-burden pair. I therefore retain these four as the "
                "largest components while reserving meaningful mass for AUROC, balanced accuracy, F1, "
                "calibration, and negative-class behaviour. llm_prior: calibration should remain visible "
                "because this benchmark compares candidate probabilities, but it should not displace the "
                "literature-defined primary discrimination endpoints before calibration protocols are audited."
            ),
            "evidence_sources": [
                {
                    "source": "supplied literature_consensus.historical_primary_proposal",
                    "use": "sets AUPRC, MCC, Recall, and Precision as dominant endpoint family",
                },
                {
                    "source": "supplied literature_consensus.historical_revised_proposal",
                    "use": "retains AUROC as a secondary threshold-free view",
                },
                {
                    "source": "supplied metric_definitions and dataset_profiles",
                    "use": "preserves threshold, calibration, and negative-class dimensions across severe and moderate prevalence",
                },
                {
                    "source": "llm_prior",
                    "use": "allocates mandatory residual mass across complementary metrics; not an external citation",
                },
            ],
            "weights": initial_weights,
            "confidence": 0.76,
            "uncertainties": [
                "The supplied consensus is a project memory summary rather than a claim-level citation audit.",
                "Homology leakage and independence gates are pending for all three datasets.",
                "Threshold-selection and calibration protocols have not yet been independently verified.",
            ],
        },
        "rounds": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_bundle": str(BUNDLE_PATH.relative_to(ROOT)).replace("\\", "/"),
    }

    previous = initial_weights
    for expected_round, round_info in enumerate(bundle["rounds"], start=1):
        round_number = int(round_info["round"])
        if round_number != expected_round:
            raise ValueError(f"round sequence mismatch at {expected_round}: {round_number}")
        sampled = list(round_info["sampled_datasets"])
        metric_evidence = round_info["metric_evidence"]
        target, raw_scores = _evidence_target(metrics, metric_evidence, sampled)
        proposed = _proposal(previous, target)
        change = _l1(previous, proposed)
        if change > MAX_L1 + 1e-8:
            raise ValueError(f"round {round_number}: L1 {change} exceeds {MAX_L1}")

        top_evidence = _top_items(raw_scores)
        bottom_evidence = _bottom_items(raw_scores)
        top_weights = _top_items(proposed, 4)
        severe_count = sampled.count("Dataset_A")
        prevalence_context = (
            f"Dataset_A appears {severe_count}/3 times; its supplied positive fraction is about 1.75%, "
            "so imbalance-sensitive endpoints retain priority."
            if severe_count
            else
            "This resample contains only Dataset_B/C; I retain the global imbalance anchor rather than "
            "letting a locally balanced resample erase the benchmark's severe-imbalance use case."
        )
        deltas = {metric: proposed[metric] - previous[metric] for metric in metrics}
        rises = sorted(deltas.items(), key=lambda item: (-item[1], item[0]))[:2]
        falls = sorted(deltas.items(), key=lambda item: (item[1], item[0]))[:2]
        result["rounds"].append(
            {
                "round": round_number,
                "sampled_datasets": sampled,
                "analysis": (
                    f"Round {round_number} anonymous evidence is strongest for {_fmt(top_evidence)} and "
                    f"weakest for {_fmt(bottom_evidence)}. {prevalence_context} The smoothed response "
                    f"most increases {_fmt(rises)} and most decreases {_fmt(falls)} relative to this "
                    "Literature Agent's own prior-round proposal. The resulting leading weights remain "
                    f"{_fmt(top_weights)}, consistent with the supplied endpoint hierarchy."
                ),
                "evidence_response": {
                    "anonymous_evidence_top": [
                        {"metric": metric, "agent_evidence_score": round(score, 6)}
                        for metric, score in top_evidence
                    ],
                    "anonymous_evidence_bottom": [
                        {"metric": metric, "agent_evidence_score": round(score, 6)}
                        for metric, score in bottom_evidence
                    ],
                    "sample_context": {
                        "Dataset_A_count": sampled.count("Dataset_A"),
                        "Dataset_B_count": sampled.count("Dataset_B"),
                        "Dataset_C_count": sampled.count("Dataset_C"),
                    },
                    "method": (
                        "llm_prior: weighted synthesis of coverage, separation, consistency, consensus, "
                        "uniqueness, and committee_support, then blended with the supplied literature anchor; "
                        "these coefficients are Agent judgement, not published coefficients."
                    ),
                    "target_before_role_smoothing": target,
                },
                "proposed_weights": proposed,
                "previous_role_weights": previous,
                "l1_from_previous": round(change, 9),
                "confidence": _confidence(raw_scores, metric_evidence),
                "uncertainties": [
                    "Bootstrap composition changes evidence emphasis but does not establish dataset independence.",
                    "Anonymous committee_support is interpreted as supplied internal evidence, not a literature citation.",
                    "llm_prior synthesis coefficients encode methodological judgement and require Chief/Reviewer scrutiny.",
                ],
            }
        )
        previous = proposed

    # Complete post-generation audit of all 51 groups and transition constraints.
    _validate_group(result["initial"]["weights"], metrics, "initial")
    prior = result["initial"]["weights"]
    max_seen_l1 = 0.0
    for round_entry in result["rounds"]:
        weights = round_entry["proposed_weights"]
        _validate_group(weights, metrics, f"round_{round_entry['round']:03d}")
        actual_l1 = _l1(prior, weights)
        if not math.isclose(actual_l1, round_entry["l1_from_previous"], abs_tol=2e-8):
            raise ValueError(f"round {round_entry['round']}: recorded L1 mismatch")
        if actual_l1 > MAX_L1 + 1e-8:
            raise ValueError(f"round {round_entry['round']}: L1 limit exceeded")
        max_seen_l1 = max(max_seen_l1, actual_l1)
        prior = weights
    result["validation"] = {
        "weight_groups_checked": 51,
        "metric_count_each": len(metrics),
        "all_sums_equal_one": True,
        "all_weights_within_bounds": True,
        "all_round_l1_within_0_30": True,
        "maximum_observed_l1": round(max_seen_l1, 9),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["validation"], ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
