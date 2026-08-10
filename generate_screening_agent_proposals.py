"""Generate the blinded AMP Screening Agent's initial and 50-round proposals.

This module deliberately consumes only the anonymised evidence bundle.  It does
not read benchmark predictions, model identities, model scores, or rankings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = (
    ROOT
    / "data"
    / "results_manual"
    / "codex_agent_weight_meeting"
    / "agent_evidence_bundle.json"
)
OUTPUT_PATH = BUNDLE_PATH.with_name("screening_agent_proposals.json")

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

# Independent role prior. All values are explicit rather than inferred from any
# model-level result. The sum is exactly one before serialisation.
INITIAL_WEIGHTS = {
    "ACC": 0.015,
    "AUPRC": 0.210,
    "AUROC": 0.050,
    "BalancedAccuracy": 0.055,
    "BrierScore": 0.030,
    "ECE": 0.030,
    "F1-Score": 0.055,
    "MCC": 0.170,
    "NPV": 0.045,
    "Precision": 0.115,
    "Recall": 0.170,
    "Specificity": 0.055,
}

ROLE_MANDATE = (
    "Represent downstream AMP screening utility under blinded model identity: "
    "balance the cost of missed active peptides against the cost of false "
    "candidates entering wet-lab validation, while preserving imbalance-aware, "
    "confusion-matrix, specificity/NPV, and calibration evidence."
)

POLICY = {
    "blinding": (
        "Use only anonymous dataset composition, literature consensus, metric "
        "definitions, and round-level metric evidence. Never use model identity, "
        "model scores, winner frequency, or rank."
    ),
    "literature_anchor": (
        "AUPRC and MCC are the primary imbalance-aware anchors; Recall and "
        "Precision encode asymmetric screening costs."
    ),
    "llm_prior": (
        "[llm_prior] Missing a true AMP wastes a discovery opportunity, whereas "
        "a false positive consumes synthesis and wet-lab capacity. The relative "
        "cost is project-dependent, so Recall receives slightly more prior weight "
        "than Precision without claiming a measured cost ratio."
    ),
    "secondary_dimensions": (
        "Retain non-zero Specificity and NPV for exclusion quality, BrierScore and "
        "ECE for probability reliability, and AUROC/BalancedAccuracy/F1/ACC as "
        "complementary diagnostics."
    ),
    "round_update": (
        "Form an evidence-conditioned target from committee support, consistency, "
        "separation, consensus and uniqueness; apply anonymous prevalence-aware "
        "screening adjustments; then move 45% toward that target."
    ),
    "constraints": {
        "minimum_each": 0.005,
        "maximum_each": 0.35,
        "sum": 1.0,
        "maximum_role_round_l1": 0.30,
    },
}


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Normalise, enforce bounds and make six-decimal weights sum to one."""
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Weights must have a finite positive total")
    result = {metric: weights[metric] / total for metric in METRICS}
    if any(value < 0.005 or value > 0.35 for value in result.values()):
        raise ValueError("An unrounded proposed weight violates the bounds")
    rounded = {metric: round(result[metric], 6) for metric in METRICS}
    residual = round(1.0 - sum(rounded.values()), 6)
    anchor = max(METRICS, key=rounded.get)
    rounded[anchor] = round(rounded[anchor] + residual, 6)
    return rounded


def _l1(left: dict[str, float], right: dict[str, float]) -> float:
    return round(sum(abs(left[m] - right[m]) for m in METRICS), 6)


def _effective_signal(evidence: dict[str, dict[str, float]], metric: str) -> float:
    item = evidence[metric]
    return (
        0.38 * item["committee_support"]
        + 0.20 * item["consistency"]
        + 0.18 * item["separation"]
        + 0.14 * item["consensus"]
        + 0.10 * item["uniqueness"]
    )


def _round_target(
    evidence: dict[str, dict[str, float]],
    sampled_datasets: list[str],
    profiles: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    signals = {metric: _effective_signal(evidence, metric) for metric in METRICS}
    mean_signal = sum(signals.values()) / len(signals)

    # Bound the evidence multiplier so a single bootstrap round cannot overturn
    # the literature-anchored screening mandate.
    raw: dict[str, float] = {}
    for metric in METRICS:
        ratio = signals[metric] / mean_signal
        ratio = min(1.35, max(0.65, ratio))
        raw[metric] = INITIAL_WEIGHTS[metric] * (0.65 + 0.35 * ratio)

    positive_fractions = [profiles[name]["positive_fraction"] for name in sampled_datasets]
    mean_prevalence = sum(positive_fractions) / len(positive_fractions)
    severe_imbalance_fraction = sampled_datasets.count("Dataset_A") / len(sampled_datasets)
    near_balanced_fraction = 1.0 - severe_imbalance_fraction

    # These are screening-utility judgements, not empirical cost estimates.
    # They are deliberately mild and labelled llm_prior in the output narrative.
    raw["AUPRC"] *= 1.0 + 0.12 * severe_imbalance_fraction
    raw["Recall"] *= 1.0 + 0.08 * severe_imbalance_fraction
    raw["Precision"] *= 1.0 + 0.06 * severe_imbalance_fraction
    raw["NPV"] *= 1.0 + 0.02 * severe_imbalance_fraction
    raw["MCC"] *= 1.0 + 0.05 * near_balanced_fraction
    raw["BalancedAccuracy"] *= 1.0 + 0.04 * near_balanced_fraction
    raw["BrierScore"] *= 1.0 + 0.03 * near_balanced_fraction
    raw["ECE"] *= 1.0 + 0.03 * near_balanced_fraction

    target = _normalise(raw)
    ranked = sorted(METRICS, key=signals.get, reverse=True)
    response = {
        "anonymous_dataset_mix": {
            "sampled_datasets": sampled_datasets,
            "mean_positive_fraction": round(mean_prevalence, 6),
            "severe_imbalance_fraction": round(severe_imbalance_fraction, 6),
        },
        "top_effective_signals": [
            {
                "metric": metric,
                "effective_signal": round(signals[metric], 6),
                "committee_support": evidence[metric]["committee_support"],
                "consistency": evidence[metric]["consistency"],
                "separation": evidence[metric]["separation"],
            }
            for metric in ranked[:4]
        ],
        "screening_cost_metrics": {
            metric: {
                "effective_signal": round(signals[metric], 6),
                "committee_support": evidence[metric]["committee_support"],
            }
            for metric in ("Recall", "Precision", "Specificity", "NPV")
        },
        "calibration_metrics": {
            metric: {
                "effective_signal": round(signals[metric], 6),
                "committee_support": evidence[metric]["committee_support"],
            }
            for metric in ("BrierScore", "ECE")
        },
        "evidence_conditioned_target": target,
    }
    return target, response


def _round_analysis(round_no: int, response: dict[str, Any]) -> str:
    mix = response["anonymous_dataset_mix"]
    top = response["top_effective_signals"]
    top_text = "、".join(f"{item['metric']}({item['effective_signal']:.3f})" for item in top)
    cost = response["screening_cost_metrics"]
    recall = cost["Recall"]["effective_signal"]
    precision = cost["Precision"]["effective_signal"]
    calibration = response["calibration_metrics"]
    cal_text = "、".join(
        f"{name}({item['effective_signal']:.3f})" for name, item in calibration.items()
    )
    if mix["severe_imbalance_fraction"] > 0:
        prevalence_note = (
            "本轮包含极低阳性率匿名数据集，因此提高AUPRC对类别不平衡的响应，并对"
            "Recall与Precision作温和代价修正。"
        )
    else:
        prevalence_note = (
            "本轮由较平衡匿名数据集构成，因此不额外放大极端不平衡修正，适度保留"
            "MCC、BalancedAccuracy与校准维度。"
        )
    return (
        f"第{round_no}轮只读取匿名指标证据。综合信号最高的是{top_text}。"
        f"Recall与Precision的有效信号分别为{recall:.3f}和{precision:.3f}；"
        f"校准证据为{cal_text}。{prevalence_note}"
        "[llm_prior] 漏检AMP会损失发现机会，假阳性则消耗合成与湿实验资源；当前项目"
        "没有给出可验证的货币化代价比，因此只维持Recall略高于Precision的角色偏好，"
        "不把该判断伪装成数据结论。最终提案以45%步长靠近本轮证据目标，以避免单次"
        "bootstrap组成主导权重。"
    )


def _validate_output(payload: dict[str, Any]) -> dict[str, Any]:
    groups = [("initial", payload["initial"]["weights"])] + [
        (f"round_{item['round']:03d}", item["proposed_weights"])
        for item in payload["rounds"]
    ]
    if len(groups) != 51 or len(payload["rounds"]) != 50:
        raise AssertionError("Expected one initial proposal and 50 rounds")

    previous = None
    max_l1 = 0.0
    for label, weights in groups:
        if list(weights) != METRICS:
            raise AssertionError(f"{label}: missing or reordered metrics")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise AssertionError(f"{label}: weights do not sum to one")
        if any(value < 0.005 or value > 0.35 for value in weights.values()):
            raise AssertionError(f"{label}: a weight violates [0.005, 0.35]")
        if previous is not None:
            step_l1 = _l1(previous, weights)
            max_l1 = max(max_l1, step_l1)
            if step_l1 > 0.30:
                raise AssertionError(f"{label}: L1 step {step_l1} exceeds 0.30")
        previous = weights
    return {
        "weight_groups_validated": len(groups),
        "rounds_validated": len(payload["rounds"]),
        "metric_count_per_group": len(METRICS),
        "all_sums_equal_one": True,
        "all_weights_within_bounds": True,
        "max_observed_round_l1": round(max_l1, 6),
        "model_level_data_used": False,
    }


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle["eligible_metrics"] != METRICS:
        raise ValueError("Evidence bundle metric contract changed")
    if len(bundle["rounds"]) != 50:
        raise ValueError("Evidence bundle must contain exactly 50 rounds")

    profiles = {item["dataset"]: item for item in bundle["dataset_profiles"]}
    initial = _normalise(INITIAL_WEIGHTS)
    payload: dict[str, Any] = {
        "role": "AMP Screening Agent",
        "role_mandate": ROLE_MANDATE,
        "policy": POLICY,
        "initial": {
            "analysis": (
                "文献共识把AUPRC与MCC置于不平衡AMP评估核心，并要求同时报告Recall和"
                "Precision。作为筛选角色，我把AUPRC/MCC设为证据主轴，使Recall承担漏检"
                "风险、Precision承担湿实验假阳性负担；Specificity/NPV保留排除质量，"
                "BrierScore/ECE保留概率可靠性。[llm_prior] 在没有项目实测代价矩阵时，"
                "我判断漏检发现机会的科学代价略高，因而Recall高于Precision；这不是"
                "文献测得的固定比例。ACC仅作低权重总体诊断。"
            ),
            "evidence_sources": [
                "literature_deep_research_memory.metric_consensus: AUPRC primary under imbalance",
                "literature_deep_research_memory.metric_consensus: MCC uses all confusion cells",
                "literature_deep_research_memory.metric_consensus: Recall and Precision encode screening errors",
                "agent_evidence_bundle.dataset_profiles: anonymous prevalence only",
                "llm_prior: qualitative downstream miss-versus-wet-lab cost judgement",
            ],
            "weights": initial,
            "confidence": 0.78,
            "uncertainties": [
                "No measured project-specific cost ratio for false negatives versus false positives.",
                "Dataset independence and homology-leakage gates remain pending.",
                "Threshold selection and freezing evidence is not supplied to this role.",
                "Calibration evidence is summary-level and ECE is binning-dependent.",
            ],
        },
        "rounds": [],
    }

    previous = initial
    for round_item in bundle["rounds"]:
        round_no = round_item["round"]
        sampled = round_item["sampled_datasets"]
        evidence = round_item["metric_evidence"]
        target, response = _round_target(evidence, sampled, profiles)
        blended = {
            metric: 0.55 * previous[metric] + 0.45 * target[metric]
            for metric in METRICS
        }
        proposal = _normalise(blended)
        step_l1 = _l1(previous, proposal)
        payload["rounds"].append(
            {
                "round": round_no,
                "sampled_datasets": sampled,
                "analysis": _round_analysis(round_no, response),
                "evidence_response": response,
                "proposed_weights": proposal,
                "previous_role_weights": previous,
                "l1_from_previous": step_l1,
                "confidence": round(0.70 + 0.08 * min(1.0, 1.0 - step_l1 / 0.30), 3),
                "uncertainties": [
                    "Bootstrap evidence reflects only three anonymous datasets.",
                    "Independence and homology gates are unresolved.",
                    "[llm_prior] Downstream experimental costs are qualitative, not measured.",
                ],
            }
        )
        previous = proposal

    payload["validation"] = _validate_output(payload)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
