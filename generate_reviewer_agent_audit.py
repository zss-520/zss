"""Generate an independent Reviewer Agent audit for the 51 weight meetings.

The reviewer is deliberately information-limited.  It reads only the shared
anonymous evidence bundle and the three expert proposal files.  It never reads
model predictions, model scores, model names, winner counts, or rankings, and
it never emits accepted/final weights.  Its numeric output is a directional
priority adjustment in [-1, 1], which is advice for a later Chief Agent.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MEETING_DIR = BASE_DIR / "data" / "results_manual" / "codex_agent_weight_meeting"
EVIDENCE_PATH = MEETING_DIR / "agent_evidence_bundle.json"
PROPOSAL_PATHS = {
    "literature_agent": MEETING_DIR / "literature_agent_proposals.json",
    "statistics_agent": MEETING_DIR / "statistics_agent_proposals.json",
    "screening_agent": MEETING_DIR / "screening_agent_proposals.json",
}
OUTPUT_PATH = MEETING_DIR / "reviewer_agent_audit.json"

METRICS = [
    "ACC",
    "AUPRC",
    "AUROC",
    "BalancedAccuracy",
    "BrierScore",
    "ECE",
    "F1-Score",
    "MCC",
    "NPV",
    "Precision",
    "Recall",
    "Specificity",
]

# These are Reviewer construct judgements, not weights and not literature facts.
# They are deliberately modest; round evidence and disagreement modify them.
CONSTRUCT_DIRECTION = {
    "ACC": -0.22,
    "AUPRC": 0.24,
    "AUROC": 0.02,
    "BalancedAccuracy": 0.04,
    "BrierScore": 0.12,
    "ECE": 0.09,
    "F1-Score": -0.04,
    "MCC": 0.19,
    "NPV": -0.02,
    "Precision": 0.06,
    "Recall": 0.10,
    "Specificity": -0.03,
}

CALIBRATION = {"BrierScore", "ECE"}
THRESHOLD_DEPENDENT = {
    "ACC",
    "BalancedAccuracy",
    "F1-Score",
    "MCC",
    "NPV",
    "Precision",
    "Recall",
    "Specificity",
}
REDUNDANCY_FAMILIES = {
    "threshold_free_discrimination": ["AUPRC", "AUROC"],
    "confusion_matrix_summary": ["ACC", "BalancedAccuracy", "F1-Score", "MCC"],
    "operating_point_rates": ["NPV", "Precision", "Recall", "Specificity"],
    "probability_calibration": ["BrierScore", "ECE"],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def weight_map(record: dict[str, Any], initial: bool) -> dict[str, float]:
    key = "weights" if initial else "proposed_weights"
    values = record.get(key)
    if not isinstance(values, dict) or set(values) != set(METRICS):
        raise ValueError(f"Proposal field {key!r} does not contain the 12 required metrics")
    result = {metric: float(values[metric]) for metric in METRICS}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite value found in {key}")
    return result


def proposal_views(
    proposals: dict[str, dict[str, Any]], round_index: int | None
) -> dict[str, dict[str, float]]:
    initial = round_index is None
    views: dict[str, dict[str, float]] = {}
    for role, document in proposals.items():
        record = document["initial"] if initial else document["rounds"][round_index]
        views[role] = weight_map(record, initial=initial)
    return views


def disagreement(views: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = {role: weights[metric] for role, weights in views.items()}
        ordered = sorted(values.items(), key=lambda item: item[1])
        result[metric] = {
            "spread": round(ordered[-1][1] - ordered[0][1], 9),
            "mean_proposal": round(sum(values.values()) / len(values), 9),
            "highest_role": ordered[-1][0],
            "lowest_role": ordered[0][0],
            "role_proposals": {role: round(value, 9) for role, value in values.items()},
        }
    return result


def evidence_composite(metric_evidence: dict[str, dict[str, float]]) -> dict[str, float]:
    """Reviewer-only summary of anonymous evidence, not a proposed weight."""
    coefficients = {
        "coverage": 0.10,
        "separation": 0.17,
        "consistency": 0.18,
        "consensus": 0.15,
        "uniqueness": 0.25,
        "committee_support": 0.15,
    }
    return {
        metric: sum(float(fields[name]) * coefficient for name, coefficient in coefficients.items())
        for metric, fields in metric_evidence.items()
    }


def dataset_prevalence(bundle: dict[str, Any]) -> dict[str, float]:
    profiles = bundle.get("dataset_profiles", {})
    output: dict[str, float] = {}
    if isinstance(profiles, list):
        items = ((str(profile.get("dataset")), profile) for profile in profiles)
    elif isinstance(profiles, dict):
        items = profiles.items()
    else:
        items = []
    for name, profile in items:
        if not isinstance(profile, dict):
            continue
        value = profile.get("positive_fraction", profile.get("prevalence"))
        if value is not None:
            output[name] = float(value)
    return output


def direction_label(value: float) -> str:
    if value >= 0.12:
        return "increase"
    if value <= -0.12:
        return "decrease"
    if value > 0.035:
        return "slight_increase"
    if value < -0.035:
        return "slight_decrease"
    return "hold_or_reconcile"


def make_adjustments(
    views: dict[str, dict[str, float]],
    details: dict[str, dict[str, Any]],
    metric_evidence: dict[str, dict[str, float]] | None,
    sampled_datasets: list[str],
    prevalences: dict[str, float],
) -> dict[str, float]:
    means = {metric: details[metric]["mean_proposal"] for metric in METRICS}
    calibration_mass = means["BrierScore"] + means["ECE"]

    composites: dict[str, float] = {}
    composite_midpoint = 0.0
    if metric_evidence:
        composites = evidence_composite(metric_evidence)
        composite_midpoint = median(composites.values())

    sampled_prevalences = [prevalences[name] for name in sampled_datasets if name in prevalences]
    severe_imbalance = bool(sampled_prevalences) and median(sampled_prevalences) < 0.20

    adjustments: dict[str, float] = {}
    for metric in METRICS:
        value = CONSTRUCT_DIRECTION[metric]
        mean_weight = means[metric]
        spread = details[metric]["spread"]

        # Evidence can move the direction, but cannot erase construct review.
        if metric_evidence:
            value += 1.10 * (composites[metric] - composite_midpoint)
            uniqueness = float(metric_evidence[metric]["uniqueness"])
            consensus = float(metric_evidence[metric]["consensus"])
            if uniqueness < 0.30:
                value -= 0.08
            elif uniqueness > 0.55:
                value += 0.06
            if consensus < 0.72:
                value -= 0.035

        # Single-metric dominance is challenged regardless of the role proposing it.
        if mean_weight > 0.22:
            value -= min(0.24, 1.8 * (mean_weight - 0.22))
        elif mean_weight < 0.025 and metric not in {"ACC", "NPV", "Specificity"}:
            value += 0.055

        # Do not let high expert disagreement masquerade as agreement to increase.
        if spread > 0.055:
            value -= 0.04 if value > 0 else 0.0

        # Calibration has a protected floor in a probability-output benchmark.
        if metric in CALIBRATION and calibration_mass < 0.105:
            value += 0.12

        # Threshold-derived results are provisional until threshold provenance is frozen.
        if metric in THRESHOLD_DEPENDENT:
            value -= 0.025

        if severe_imbalance:
            if metric in {"AUPRC", "MCC", "Recall", "Precision"}:
                value += 0.045
            if metric in {"ACC", "AUROC"}:
                value -= 0.045

        adjustments[metric] = round(clamp(value), 6)
    return adjustments


def disagreement_summary(details: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(METRICS, key=lambda metric: details[metric]["spread"], reverse=True)
    top = ranked[:4]
    return {
        "largest_disagreements": [
            {
                "metric": metric,
                **details[metric],
            }
            for metric in top
        ],
        "mean_absolute_pairwise_spread": round(
            sum(details[metric]["spread"] for metric in METRICS) / len(METRICS), 9
        ),
        "reviewer_interpretation": (
            f"The largest disagreement is {top[0]} (range {details[top[0]]['spread']:.4f}), "
            f"followed by {top[1]} and {top[2]}. The Chief should resolve these construct-level "
            "differences explicitly rather than averaging them silently."
        ),
    }


def family_totals(details: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        family: round(sum(details[metric]["mean_proposal"] for metric in members), 6)
        for family, members in REDUNDANCY_FAMILIES.items()
    }


def make_review(
    bundle: dict[str, Any],
    proposals: dict[str, dict[str, Any]],
    round_index: int | None,
    prevalences: dict[str, float],
) -> dict[str, Any]:
    initial = round_index is None
    views = proposal_views(proposals, round_index)
    details = disagreement(views)
    disagreement_info = disagreement_summary(details)
    families = family_totals(details)

    if initial:
        evidence = None
        sampled: list[str] = []
        evidence_leaders: list[str] = []
        evidence_laggards: list[str] = []
        round_label = "initial meeting"
    else:
        evidence_record = bundle["rounds"][round_index]
        evidence = evidence_record["metric_evidence"]
        sampled = list(evidence_record.get("sampled_datasets", []))
        composites = evidence_composite(evidence)
        ordered = sorted(METRICS, key=lambda metric: composites[metric], reverse=True)
        evidence_leaders = ordered[:3]
        evidence_laggards = ordered[-3:]
        round_label = f"round {round_index + 1}"

    adjustments = make_adjustments(views, details, evidence, sampled, prevalences)
    preferred = {
        label: [metric for metric in METRICS if direction_label(adjustments[metric]) == label]
        for label in (
            "increase",
            "slight_increase",
            "hold_or_reconcile",
            "slight_decrease",
            "decrease",
        )
    }
    largest = disagreement_info["largest_disagreements"]

    if initial:
        evidence_clause = (
            "No round-level benchmark evidence exists yet; directions therefore reflect supplied "
            "literature memory, metric definitions, proposal structure, and explicitly labelled llm_prior."
        )
    else:
        evidence_clause = (
            f"Anonymous evidence favours {', '.join(evidence_leaders)} and is weakest for "
            f"{', '.join(evidence_laggards)} under the Reviewer composite. This is descriptive "
            "post-hoc evidence, not permission to optimise on a formal test set."
        )

    analysis = (
        f"Independent review of {round_label}: the three experts allocate, on average, "
        f"{families['threshold_free_discrimination']:.3f} to threshold-free discrimination, "
        f"{families['confusion_matrix_summary']:.3f} to overlapping confusion-matrix summaries, "
        f"{families['operating_point_rates']:.3f} to operating-point rates, and "
        f"{families['probability_calibration']:.3f} to calibration. "
        f"The widest proposal ranges are {largest[0]['metric']} ({largest[0]['spread']:.4f}), "
        f"{largest[1]['metric']} ({largest[1]['spread']:.4f}), and "
        f"{largest[2]['metric']} ({largest[2]['spread']:.4f}). {evidence_clause}"
    )

    criticisms: list[str] = [
        (
            "Literature/llm_prior boundary: the supplied literature consensus supports AUPRC, MCC, "
            "Recall and Precision as central constructs, but exact residual allocations and any "
            "miss-versus-false-positive cost ratio remain llm_prior rather than measured literature facts."
        ),
        (
            "Hidden preference check: the reviewed files expose metric-level evidence and role proposals "
            "but no model identity, model score or rank. No proposal may be justified by an implied winner."
        ),
        (
            "Post-hoc tuning risk: resampled benchmark metric evidence is useful for sensitivity analysis "
            "but must not be used to claim a leakage-free test ranking; weights require preregistration or "
            "an independent development set before confirmatory evaluation."
        ),
        (
            f"Redundancy risk: the proposed mean mass on confusion-matrix summaries is "
            f"{families['confusion_matrix_summary']:.3f}; ACC, BalancedAccuracy, F1 and MCC partially "
            "reuse the same contingency-table information and should not be treated as four independent votes."
        ),
        (
            f"Calibration visibility: BrierScore plus ECE receive mean mass "
            f"{families['probability_calibration']:.3f}. BrierScore is a proper scoring rule while ECE is "
            "binning-dependent; both must remain visible, with protocol and directionality documented."
        ),
        (
            "Prevalence/threshold dependence: ACC, predictive values, F1, sensitivity and specificity "
            "depend on prevalence and/or a decision threshold. The threshold-selection dataset and freezing "
            "point remain unresolved, so round-specific movements in these metrics are provisional."
        ),
    ]
    if largest[0]["spread"] > 0.055:
        criticisms.append(
            f"Material expert conflict persists for {largest[0]['metric']}: "
            f"{largest[0]['highest_role']} proposes the most and {largest[0]['lowest_role']} the least. "
            "The Chief must record a construct-based rationale for resolving this gap."
        )
    if not initial and evidence is not None:
        low_unique = sorted(METRICS, key=lambda metric: evidence[metric]["uniqueness"])[:3]
        criticisms.append(
            f"This round's lowest uniqueness occurs for {', '.join(low_unique)}; increasing all of them "
            "together would amplify redundant evidence rather than add independent information."
        )

    required_changes = [
        "Keep literature-supported claims separate from statements labelled llm_prior; do not present an exact allocation as citation-derived.",
        "Do not use model identity, score, rank, winner frequency, or downstream ranking stability when adjudicating metric priority.",
        "Treat these 50 adaptive rounds as exploratory sensitivity analysis and freeze a single rule before any confirmatory test evaluation.",
        "Audit metric redundancy explicitly, particularly confusion-matrix summaries and the Recall/Specificity-derived composites.",
        "Retain both discrimination and calibration dimensions; state that lower BrierScore/ECE is better before normalization.",
        "Document prevalence, threshold-selection source, threshold freezing, homology partitioning, and dataset independence before a formal benchmark claim.",
    ]
    if not initial:
        required_changes.append(
            f"For {round_label}, explain why evidence leaders {', '.join(evidence_leaders)} should or should not "
            f"override construct concerns, and why evidence laggards {', '.join(evidence_laggards)} remain represented."
        )

    leakage_check = {
        "model_identity_accessed": False,
        "model_scores_accessed": False,
        "model_rankings_accessed": False,
        "review_inputs": [
            "anonymous shared metric evidence",
            "supplied literature-memory consensus",
            "metric definitions and anonymous prevalence profiles",
            "three blinded expert proposals",
        ],
        "status": "conditional_pass_for_exploratory_review_only",
        "test_set_adaptation_risk": (
            "Round evidence appears derived from the evaluated benchmark datasets. Adaptive weights therefore "
            "cannot support a leakage-free confirmatory ranking without an independent development/validation "
            "stage and prospectively frozen weights."
        ),
    }

    unresolved_risks = [
        "The evaluated datasets' independence and homology-leakage gates are unresolved.",
        "The threshold-selection provenance and whether thresholds were frozen before evaluation are unresolved.",
        "No project-measured utility ratio quantifies missed AMP candidates versus false candidates sent to wet-lab validation.",
        "Correlated metrics may double-count the same discrimination or confusion-matrix signal.",
        "Calibration estimates may be unstable across prevalence shifts, and ECE depends on binning choices.",
        "Repeated use of benchmark-derived evidence creates post-hoc selection risk even while model identities remain blinded.",
    ]
    if not initial:
        unresolved_risks.append(
            f"The sampled composition for {round_label} ({', '.join(sampled)}) can change prevalence emphasis and "
            "must not be interpreted as an independent replication."
        )

    review: dict[str, Any] = {
        "analysis": analysis,
        "criticisms": criticisms,
        "required_changes": required_changes,
        "preferred_directions": {
            **preferred,
            "interpretation": (
                "Directions summarize Reviewer pressure only. They are not weights, do not sum to one, "
                "and must be reconciled by the Chief with explicit reasons."
            ),
        },
        "metric_priority_adjustments": adjustments,
        "leakage_check": leakage_check,
        "unresolved_risks": unresolved_risks,
        "expert_disagreement_summary": disagreement_info,
    }
    if not initial:
        review = {"round": round_index + 1, **review}
    return review


def validate_output(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("role") != "reviewer_agent":
        raise ValueError("role must be reviewer_agent")
    if len(document.get("rounds", [])) != 50:
        raise ValueError("Exactly 50 round reviews are required")
    meetings = [document["initial"], *document["rounds"]]
    for expected, meeting in enumerate(meetings):
        adjustments = meeting.get("metric_priority_adjustments")
        if not isinstance(adjustments, dict) or set(adjustments) != set(METRICS):
            raise ValueError(f"Meeting {expected} lacks all 12 metric adjustments")
        if any(float(value) < -1 or float(value) > 1 for value in adjustments.values()):
            raise ValueError(f"Meeting {expected} has an adjustment outside [-1, 1]")
        if expected > 0 and meeting.get("round") != expected:
            raise ValueError(f"Round sequence error at {expected}")

    forbidden = {"final_weights", "accepted_weights"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden.intersection(value)
            if overlap:
                raise ValueError(f"Forbidden output field(s): {sorted(overlap)}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document)
    return {
        "review_count": len(meetings),
        "round_count": len(document["rounds"]),
        "metrics_per_review": len(METRICS),
        "adjustment_range": [-1, 1],
        "forbidden_weight_fields_absent": True,
        "model_scores_or_rankings_read": False,
    }


def main() -> None:
    bundle = load_json(EVIDENCE_PATH)
    proposals = {role: load_json(path) for role, path in PROPOSAL_PATHS.items()}

    if bundle.get("eligible_metrics") != METRICS:
        raise ValueError("Eligible metric order/content differs from the Reviewer contract")
    if len(bundle.get("rounds", [])) != 50:
        raise ValueError("Evidence bundle must contain exactly 50 rounds")
    for role, document in proposals.items():
        if len(document.get("rounds", [])) != 50:
            raise ValueError(f"{role} must contain exactly 50 proposals")
        if [row.get("round") for row in document["rounds"]] != list(range(1, 51)):
            raise ValueError(f"{role} round numbers are not 1..50")

    prevalences = dataset_prevalence(bundle)
    document: dict[str, Any] = {
        "role": "reviewer_agent",
        "role_mandate": (
            "Independently audit the literature/llm_prior boundary, hidden model preference, post-hoc "
            "test-set tuning, single-metric dominance, metric redundancy, calibration visibility, "
            "prevalence/threshold dependence, and expert disagreement. Provide only directional "
            "metric_priority_adjustments in [-1, 1]; never decide or emit final weights."
        ),
        "policy": {
            "information_boundary": (
                "Reads only agent_evidence_bundle.json and the three expert proposal JSON files. "
                "No model prediction, identity, score, ranking, winner count, or bubble-plot result is read."
            ),
            "literature_boundary": (
                "Supplied literature_consensus is project memory. Reviewer methodological judgement is "
                "llm_prior and cannot be represented as a newly verified citation."
            ),
            "decision_boundary": (
                "metric_priority_adjustments are signed review directions, not normalized weights; "
                "the Chief alone is responsible for adjudication."
            ),
            "anti_leakage": (
                "Benchmark-derived round evidence may inform exploratory robustness discussion only. "
                "Confirmatory weights must be selected without the formal test outcomes and frozen prospectively."
            ),
            "metric_safeguards": (
                "Challenge dominance and redundancy, preserve calibration, flag lower-is-better metric handling, "
                "and condition threshold/prevalence-dependent metrics on a documented frozen protocol."
            ),
        },
        "inputs": {
            "evidence_bundle": EVIDENCE_PATH.name,
            "expert_proposals": [path.name for path in PROPOSAL_PATHS.values()],
            "explicitly_excluded": [
                "model identities",
                "model predictions",
                "model scores",
                "rankings",
                "winner frequencies",
                "final result plots",
            ],
        },
        "initial": make_review(bundle, proposals, None, prevalences),
        "rounds": [make_review(bundle, proposals, index, prevalences) for index in range(50)],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    document["validation"] = validate_output(document)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Re-read from disk and validate the actual artifact, not only the in-memory object.
    written = load_json(OUTPUT_PATH)
    validation = validate_output(written)
    print(f"Wrote: {OUTPUT_PATH}")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
