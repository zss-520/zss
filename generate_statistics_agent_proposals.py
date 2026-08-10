"""Generate blinded Benchmark Statistics Agent proposals for 51 weight meetings.

The generator consumes only metric-level evidence and anonymised dataset labels from
``agent_evidence_bundle.json``.  It deliberately never reads model identities,
model-level scores, or rankings.  Literature/LLM knowledge defines the initial
statistical policy; each subsequent proposal reacts to the round-specific bootstrap
evidence while respecting the meeting constraints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/results_manual/codex_agent_weight_meeting/agent_evidence_bundle.json"
OUTPUT = ROOT / "data/results_manual/codex_agent_weight_meeting/statistics_agent_proposals.json"

MIN_WEIGHT = 0.005
MAX_WEIGHT = 0.35
MAX_L1 = 0.30

METRICS = [
    "ACC", "AUPRC", "AUROC", "BalancedAccuracy", "BrierScore", "ECE",
    "F1-Score", "MCC", "NPV", "Precision", "Recall", "Specificity",
]

# Statistical prior, informed by the supplied literature consensus and general
# knowledge of imbalanced binary classification.  It intentionally retains both
# discrimination and calibration dimensions and does not depend on benchmark ranks.
INITIAL_WEIGHTS = {
    "ACC": 0.025,
    "AUPRC": 0.210,
    "AUROC": 0.070,
    "BalancedAccuracy": 0.065,
    "BrierScore": 0.070,
    "ECE": 0.055,
    "F1-Score": 0.060,
    "MCC": 0.190,
    "NPV": 0.025,
    "Precision": 0.095,
    "Recall": 0.100,
    "Specificity": 0.035,
}

# Importance reflects construct relevance before seeing benchmark outcomes.  It is
# intentionally separate from evidence quality, which changes every round.
DOMAIN_IMPORTANCE = {
    "ACC": 0.45,
    "AUPRC": 1.35,
    "AUROC": 0.80,
    "BalancedAccuracy": 0.82,
    "BrierScore": 0.90,
    "ECE": 0.78,
    "F1-Score": 0.72,
    "MCC": 1.28,
    "NPV": 0.48,
    "Precision": 0.95,
    "Recall": 1.00,
    "Specificity": 0.62,
}

# Metrics calculated at a frozen decision threshold are useful but correlated.
THRESHOLD_DISCOUNT = {
    "ACC": 0.72,
    "BalancedAccuracy": 0.88,
    "F1-Score": 0.84,
    "MCC": 0.94,
    "NPV": 0.76,
    "Precision": 0.92,
    "Recall": 0.94,
    "Specificity": 0.82,
}

CALIBRATION_METRICS = {"BrierScore", "ECE"}
REDUNDANT_THRESHOLD_FAMILY = {
    "ACC", "BalancedAccuracy", "F1-Score", "MCC", "NPV", "Precision",
    "Recall", "Specificity",
}


def normalise_bounded(weights: dict[str, float]) -> dict[str, float]:
    """Project weights onto the bounded simplex with stable metric ordering."""
    values = {m: max(MIN_WEIGHT, min(MAX_WEIGHT, float(weights[m]))) for m in METRICS}
    for _ in range(100):
        delta = 1.0 - sum(values.values())
        if abs(delta) < 1e-12:
            break
        if delta > 0:
            free = [m for m in METRICS if values[m] < MAX_WEIGHT - 1e-12]
            room = sum(MAX_WEIGHT - values[m] for m in free)
            for m in free:
                values[m] += delta * (MAX_WEIGHT - values[m]) / room
        else:
            free = [m for m in METRICS if values[m] > MIN_WEIGHT + 1e-12]
            room = sum(values[m] - MIN_WEIGHT for m in free)
            for m in free:
                values[m] += delta * (values[m] - MIN_WEIGHT) / room
    rounded = {m: round(values[m], 8) for m in METRICS}
    # Put any floating-point remainder on a metric with ample room.
    residual = round(1.0 - sum(rounded.values()), 8)
    target = "MCC" if MIN_WEIGHT <= rounded["MCC"] + residual <= MAX_WEIGHT else "AUPRC"
    rounded[target] = round(rounded[target] + residual, 8)
    return rounded


def l1(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a[m] - b[m]) for m in METRICS)


def evidence_quality(metric: str, ev: dict[str, float]) -> float:
    """Score evidence without any access to model outcomes or identities."""
    quality = (
        0.17 * ev["coverage"]
        + 0.20 * ev["separation"]
        + 0.17 * ev["consistency"]
        + 0.16 * ev["consensus"]
        + 0.16 * ev["uniqueness"]
        + 0.14 * ev["committee_support"]
    )
    quality *= DOMAIN_IMPORTANCE[metric]
    quality *= THRESHOLD_DISCOUNT.get(metric, 1.0)

    # A high-consensus/low-uniqueness threshold statistic adds little information
    # beyond its neighbours.  MCC is only mildly penalised because it uses all four
    # confusion-matrix cells; raw ACC is penalised more under class imbalance.
    redundancy = max(0.0, ev["consensus"] - ev["uniqueness"])
    if metric in REDUNDANT_THRESHOLD_FAMILY:
        strength = 0.10 if metric == "MCC" else (0.24 if metric == "ACC" else 0.17)
        quality *= 1.0 - strength * redundancy

    # ECE is binning-dependent, so high uniqueness may partly reflect estimator
    # instability.  Brier score is a smoother proper scoring rule.
    if metric == "ECE":
        quality *= 0.91
    elif metric == "BrierScore":
        quality *= 1.04
    return max(1e-12, quality)


def propose_round(previous: dict[str, float], metric_evidence: dict[str, Any]) -> dict[str, float]:
    qualities = {m: evidence_quality(m, metric_evidence[m]) for m in METRICS}
    total = sum(qualities.values())
    evidence_target = {m: qualities[m] / total for m in METRICS}

    # Retain the literature/LLM prior but allow real bootstrap evidence to move the
    # proposal.  Smoothing also guards against overreacting to a single resample.
    prior_target = normalise_bounded({
        m: 0.62 * INITIAL_WEIGHTS[m] + 0.38 * evidence_target[m] for m in METRICS
    })
    candidate = normalise_bounded({
        m: 0.58 * previous[m] + 0.42 * prior_target[m] for m in METRICS
    })

    distance = l1(candidate, previous)
    if distance > MAX_L1:
        scale = MAX_L1 / distance
        candidate = normalise_bounded({
            m: previous[m] + scale * (candidate[m] - previous[m]) for m in METRICS
        })
    return candidate


def signed_changes(previous: dict[str, float], current: dict[str, float]) -> tuple[list[str], list[str]]:
    changes = sorted(((current[m] - previous[m], m) for m in METRICS), reverse=True)
    up = [m for d, m in changes if d > 1e-6][:3]
    down = [m for d, m in reversed(changes) if d < -1e-6][:3]
    return up, down


def metric_phrase(metric: str, ev: dict[str, float]) -> str:
    return (
        f"{metric}(coverage={ev['coverage']:.3f}, separation={ev['separation']:.3f}, "
        f"consistency={ev['consistency']:.3f}, consensus={ev['consensus']:.3f}, "
        f"uniqueness={ev['uniqueness']:.3f}, committee_support={ev['committee_support']:.3f})"
    )


def make_round_record(
    round_data: dict[str, Any], previous: dict[str, float], current: dict[str, float]
) -> dict[str, Any]:
    evidence = round_data["metric_evidence"]
    up, down = signed_changes(previous, current)
    qualities = {m: evidence_quality(m, evidence[m]) for m in METRICS}
    strongest = sorted(METRICS, key=lambda m: qualities[m], reverse=True)[:3]
    redundant = sorted(
        REDUNDANT_THRESHOLD_FAMILY,
        key=lambda m: evidence[m]["consensus"] - evidence[m]["uniqueness"],
        reverse=True,
    )[:3]
    weakest_coverage = min(evidence[m]["coverage"] for m in METRICS)
    mean_consistency = sum(evidence[m]["consistency"] for m in METRICS) / len(METRICS)
    mean_support = sum(evidence[m]["committee_support"] for m in METRICS) / len(METRICS)
    direction = (
        f"本轮上调 {', '.join(up) if up else '无'}，下调 {', '.join(down) if down else '无'}。"
        f"证据质量领先的是 {'; '.join(metric_phrase(m, evidence[m]) for m in strongest)}。"
    )
    analysis = (
        f"第{round_data['round']}轮匿名重采样为 {', '.join(round_data['sampled_datasets'])}。"
        f"最低覆盖率为{weakest_coverage:.3f}，平均一致性为{mean_consistency:.3f}，"
        f"平均委员会支持为{mean_support:.3f}。{direction}"
        f"对 {', '.join(redundant)} 的高共识但低独特性按冗余处理；"
        "阈值型指标仅在阈值由独立验证集确定并冻结时可解释。"
        "校准维度同时保留BrierScore与ECE，但因ECE受分箱影响，其独特性不直接等同于可靠性。"
    )
    confidence = max(0.55, min(0.92, 0.50 + 0.22 * mean_consistency + 0.18 * mean_support))
    return {
        "round": round_data["round"],
        "sampled_datasets": round_data["sampled_datasets"],
        "analysis": analysis,
        "evidence_response": {
            "llm_prior": (
                "文献与统计知识将AUPRC和MCC作为类别不均衡下的主干证据，Recall/Precision反映漏检与"
                "实验假阳性代价；AUROC、阈值型混淆矩阵指标及校准指标必须保留但避免重复计权。"
            ),
            "benchmark_evidence": (
                f"仅使用本轮匿名metric_evidence；综合coverage、separation、consistency、consensus、"
                f"uniqueness与committee_support后，上调 {', '.join(up) if up else '无'}，"
                f"下调 {', '.join(down) if down else '无'}。未使用模型身份、得分或排名。"
            ),
            "redundancy_control": (
                f"本轮冗余折扣优先作用于 {', '.join(redundant)}；ACC在不均衡数据上使用更强折扣，"
                "MCC因使用完整混淆矩阵而仅轻度折扣。"
            ),
            "calibration_control": (
                f"BrierScore evidence={metric_phrase('BrierScore', evidence['BrierScore'])}; "
                f"ECE evidence={metric_phrase('ECE', evidence['ECE'])}。"
            ),
        },
        "proposed_weights": current,
        "previous_role_weights": previous,
        "l1_from_previous": round(l1(previous, current), 8),
        "confidence": round(confidence, 4),
        "uncertainties": [
            "三个匿名数据集的独立性、同源性去泄漏与来源门控仍为pending。",
            "Precision、NPV及ACC受类别流行率影响，跨数据集迁移解释受限。",
            "所有阈值型指标要求在独立验证证据上选阈值并在测试前冻结。",
            "ECE依赖分箱方案；BrierScore同时混合校准与概率分辨能力。",
        ],
    }


def validate_weights(label: str, weights: dict[str, float], previous: dict[str, float] | None = None) -> None:
    if list(weights) != METRICS:
        raise ValueError(f"{label}: metric order/set mismatch")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-7):
        raise ValueError(f"{label}: weights sum to {sum(weights.values())}")
    for metric, value in weights.items():
        if not MIN_WEIGHT <= value <= MAX_WEIGHT:
            raise ValueError(f"{label}: {metric}={value} outside bounds")
    if previous is not None and l1(weights, previous) > MAX_L1 + 1e-7:
        raise ValueError(f"{label}: L1={l1(weights, previous)} exceeds {MAX_L1}")


def main() -> None:
    bundle = json.loads(INPUT.read_text(encoding="utf-8"))
    if bundle["eligible_metrics"] != METRICS:
        raise ValueError("Unexpected metric list; refusing to silently remap metrics")
    if len(bundle["rounds"]) != 50:
        raise ValueError(f"Expected 50 rounds, got {len(bundle['rounds'])}")

    initial = normalise_bounded(INITIAL_WEIGHTS)
    validate_weights("initial", initial)
    output: dict[str, Any] = {
        "role": "Benchmark Statistics Agent",
        "role_mandate": (
            "在模型身份、模型分数与排名完全盲化的条件下，为AMP二分类基准提出可审计的指标权重；"
            "统筹类别不均衡、指标冗余、阈值依赖、跨数据集稳定性与概率校准。"
        ),
        "policy": {
            "model_blinding": "No model identity, model score, or ranking is read or used.",
            "knowledge_separation": "llm_prior defines construct importance; benchmark_evidence controls round adaptation.",
            "evidence_formula": (
                "0.17 coverage + 0.20 separation + 0.17 consistency + 0.16 consensus + "
                "0.16 uniqueness + 0.14 committee_support, followed by construct, threshold, "
                "redundancy and calibration adjustments"
            ),
            "temporal_smoothing": "42% current prior/evidence target + 58% previous role proposal.",
            "constraints": {"min_weight": MIN_WEIGHT, "max_weight": MAX_WEIGHT, "sum": 1.0, "max_l1": MAX_L1},
        },
        "initial": {
            "analysis": (
                "初始提案以AUPRC与MCC为不均衡二分类的主干；Recall与Precision分别约束漏掉AMP候选和"
                "将假阳性送入湿实验的代价。AUROC提供阈值无关的补充判别信息。BalancedAccuracy、F1、"
                "Specificity、NPV与ACC均保留但因阈值依赖和混淆矩阵信息重叠而降权，ACC另受流行率影响。"
                "BrierScore与ECE保留独立校准维度，其中Brier是适当评分规则，ECE因分箱依赖而权重较低。"
            ),
            "evidence_sources": {
                "llm_prior": [
                    "不均衡分类中AUPRC比ACC或单独AUROC更贴近阳性检索质量。",
                    "MCC利用TP/TN/FP/FN并对类别不均衡更稳健。",
                    "Recall与Precision编码AMP发现漏检和实验假阳性两类不同代价。",
                    "概率输出需要用BrierScore与ECE检查校准，不能只报告判别能力。",
                ],
                "literature_evidence": bundle["literature_consensus"]["claims"],
                "benchmark_evidence": "初始会议尚无单轮重采样证据，故不利用任何模型结果。",
            },
            "weights": initial,
            "confidence": 0.76,
            "uncertainties": [
                "不同应用场景中漏检与假阳性成本尚未由前瞻性实验量化。",
                "数据集独立性、同源性去泄漏与来源门控尚未解决。",
                "阈值选择协议及ECE分箱方案需要预注册。",
            ],
        },
        "rounds": [],
    }

    previous = initial
    for round_data in bundle["rounds"]:
        current = propose_round(previous, round_data["metric_evidence"])
        validate_weights(f"round_{round_data['round']}", current, previous)
        output["rounds"].append(make_round_record(round_data, previous, current))
        previous = current

    if len(output["rounds"]) != 50:
        raise AssertionError("Output does not contain 50 rounds")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={OUTPUT}")
    print("validated_weight_sets=51")
    print(f"max_round_l1={max(r['l1_from_previous'] for r in output['rounds']):.8f}")


if __name__ == "__main__":
    main()
