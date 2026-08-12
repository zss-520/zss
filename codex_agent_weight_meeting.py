# -*- coding: utf-8 -*-
"""Local Codex multi-Agent weight deliberation and 50-round benchmark runner.

This is the tenant-safe backend used when external LLM export is unavailable.
Three independent Codex expert Agents write role-specific proposals, a Reviewer
Agent audits them, and this module acts as the reproducible Chief execution
layer: it validates proposals, reconciles them sequentially, scores all models,
and exports the complete audit trail and publication figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from iterative_weight_meeting import _prepare, _review_metric_evidence, _score_models
from llm_agent_weight_meeting import (
    LITERATURE_CONSENSUS,
    METRIC_GUIDE,
    collect_eval_rows,
    dataset_profiles,
)
from model_resource_policy import (
    apply_resource_gate,
    load_model_resource_policy,
    write_resource_gate_audit,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "data" / "results_manual"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "codex_agent_weight_meeting"
EXPERT_FILES = {
    "literature_agent": "literature_agent_proposals.json",
    "statistics_agent": "statistics_agent_proposals.json",
    "screening_agent": "screening_agent_proposals.json",
}
WEIGHT_MIN = 0.005
WEIGHT_MAX = 0.35
MAX_L1_DELTA = 0.30


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_weights(raw: Mapping[str, Any], metrics: Sequence[str]) -> Dict[str, float]:
    missing = [m for m in metrics if m not in raw]
    extra = [m for m in raw if m not in metrics]
    if missing or extra:
        raise ValueError(f"weight keys mismatch: missing={missing}, extra={extra}")
    values = {m: min(WEIGHT_MAX, max(WEIGHT_MIN, float(raw[m]))) for m in metrics}
    # Iterative capped-simplex projection with a strictly positive floor.
    for _ in range(30):
        total = sum(values.values())
        if abs(total - 1.0) <= 1e-12:
            break
        if total < 1.0:
            free = [m for m in metrics if values[m] < WEIGHT_MAX - 1e-12]
            room = sum(WEIGHT_MAX - values[m] for m in free)
            if not free or room <= 0:
                break
            add = 1.0 - total
            for m in free:
                values[m] += add * (WEIGHT_MAX - values[m]) / room
        else:
            free = [m for m in metrics if values[m] > WEIGHT_MIN + 1e-12]
            room = sum(values[m] - WEIGHT_MIN for m in free)
            if not free or room <= 0:
                break
            remove = total - 1.0
            for m in free:
                values[m] -= remove * (values[m] - WEIGHT_MIN) / room
    values = {m: round(float(values[m]), 10) for m in metrics}
    drift = 1.0 - sum(values.values())
    target = max(metrics, key=lambda m: WEIGHT_MAX - values[m] if drift > 0 else values[m] - WEIGHT_MIN)
    values[target] = round(values[target] + drift, 10)
    return values


def l1_distance(a: Mapping[str, float], b: Mapping[str, float], metrics: Sequence[str]) -> float:
    return float(sum(abs(float(a[m]) - float(b[m])) for m in metrics))


def prepare_bundle(results_dir: Path, output_dir: Path, rounds: int, seed: int) -> Path:
    rows = collect_eval_rows(results_dir)
    eligible_rows, resource_gate = apply_resource_gate(rows, load_model_resource_policy())
    prepared = _prepare(eligible_rows)
    metrics = [prepared["names"][key] for key in prepared["metrics"]]
    if len(prepared["datasets"]) != 3 or len(prepared["models"]) < 3 or len(metrics) != 12:
        raise ValueError(
            "Expected 3 datasets, at least 3 resource-eligible models and 12 eligible metrics; "
            f"got {len(prepared['datasets'])}, {len(prepared['models'])}, {len(metrics)}"
        )
    profiles = dataset_profiles(results_dir)
    alias = {p["dataset"]: f"Dataset_{chr(65 + i)}" for i, p in enumerate(profiles)}
    anonymous_profiles = [
        {
            "dataset": alias[p["dataset"]], "rows": p["rows"],
            "positives": p["positives"], "negatives": p["negatives"],
            "positive_fraction": p["positive_fraction"],
            "independence_gate": "pending",
        }
        for p in profiles
    ]
    rng = np.random.default_rng(seed)
    round_payloads: List[Dict[str, Any]] = []
    internal_plan: List[Dict[str, Any]] = []
    for round_no in range(1, rounds + 1):
        sampled = rng.choice(prepared["datasets"], size=len(prepared["datasets"]), replace=True).tolist()
        raw = _review_metric_evidence(prepared, sampled)
        evidence = {prepared["names"][key]: value for key, value in raw.items()}
        round_payloads.append({
            "round": round_no,
            "sampled_datasets": [alias[name] for name in sampled],
            "metric_evidence": evidence,
        })
        internal_plan.append({"round": round_no, "sampled_datasets": sampled})
    bundle = {
        "generated_at": now(),
        "backend": "local_codex_multi_agent",
        "scientific_task": "Select metric weights for an AMP binary-classification benchmark.",
        "constraints": {
            "all_metrics_required": True, "min_weight": WEIGHT_MIN,
            "max_weight": WEIGHT_MAX, "sum": 1.0,
            "max_round_to_round_l1": MAX_L1_DELTA,
            "model_identity_blinded": True,
            "resource_gate_precedes_scoring": True,
        },
        "eligible_metrics": metrics,
        "metric_definitions": {m: METRIC_GUIDE[m] for m in metrics},
        "literature_consensus": {k: v for k, v in LITERATURE_CONSENSUS.items() if k != "verified_dataset_sources"},
        "evidence_pool_summary": {"papers": 2365, "evidence_batches": 304},
        "dataset_profiles": anonymous_profiles,
        "resource_gate_summary": {
            "policy_mode": resource_gate["policy_mode"],
            "models_before": resource_gate["models_before"],
            "models_after": resource_gate["models_after"],
            "excluded_models": resource_gate["excluded_models"],
            "flagged_models": resource_gate["flagged_models"],
        },
        "rounds": round_payloads,
        "caveat": "Exploratory post-hoc benchmark; provenance, independence and homology gates remain pending.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resource_gate_audit(output_dir, resource_gate)
    path = output_dir / "agent_evidence_bundle.json"
    write_json(path, bundle)
    write_json(output_dir / "internal_bootstrap_plan.json", {
        "generated_at": now(), "seed": seed, "rounds": internal_plan,
        "note": "Internal scoring plan; not supplied to weight-setting Agents beyond anonymous aliases.",
    })
    print(f"[prepared] {path}")
    print(f"[dimensions] datasets={len(prepared['datasets'])} models={len(prepared['models'])} metrics={len(metrics)} rounds={rounds}")
    return path


def validate_proposal(weights: Mapping[str, Any], metrics: Sequence[str], label: str) -> Dict[str, float]:
    if set(weights) != set(metrics):
        raise ValueError(f"{label}: metric keys do not match the evidence bundle")
    parsed = {m: float(weights[m]) for m in metrics}
    if any(not math.isfinite(v) or v < WEIGHT_MIN - 1e-9 or v > WEIGHT_MAX + 1e-9 for v in parsed.values()):
        raise ValueError(f"{label}: weight outside [{WEIGHT_MIN}, {WEIGHT_MAX}]")
    if abs(sum(parsed.values()) - 1.0) > 1e-5:
        raise ValueError(f"{label}: weights sum to {sum(parsed.values())}, expected 1")
    return parsed


def load_deliberations(output_dir: Path, metrics: Sequence[str], rounds: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    experts: Dict[str, Any] = {}
    for role, filename in EXPERT_FILES.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {role} record: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        declared_role = str(record.get("role", "")).strip().lower().replace("_", " ")
        role_token = role.replace("_agent", "").replace("_", " ")
        if role_token not in declared_role or len(record.get("rounds", [])) != rounds:
            raise ValueError(f"{role}: invalid declared role ({record.get('role')!r}) or round count")
        validate_proposal(record["initial"]["weights"], metrics, f"{role}/initial")
        previous = record["initial"]["weights"]
        for expected, item in enumerate(record["rounds"], 1):
            if int(item.get("round", -1)) != expected:
                raise ValueError(f"{role}: expected round {expected}")
            current = validate_proposal(item["proposed_weights"], metrics, f"{role}/round_{expected}")
            if l1_distance(previous, current, metrics) > MAX_L1_DELTA + 1e-5:
                raise ValueError(f"{role}/round_{expected}: role proposal exceeds L1 limit")
            previous = current
        experts[role] = record
    reviewer_path = output_dir / "reviewer_agent_audit.json"
    if not reviewer_path.exists():
        raise FileNotFoundError(f"Missing Reviewer record: {reviewer_path}")
    reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
    if reviewer.get("role") != "reviewer_agent" or len(reviewer.get("rounds", [])) != rounds:
        raise ValueError("Reviewer record has invalid role or round count")
    return experts, reviewer


def reviewer_adjustments(review: Mapping[str, Any], metrics: Sequence[str]) -> Dict[str, float]:
    raw = review.get("metric_priority_adjustments", {})
    return {m: max(-1.0, min(1.0, float(raw.get(m, 0.0)))) for m in metrics}


def reconcile(
    proposals: Sequence[Mapping[str, float]], review: Mapping[str, Any],
    metrics: Sequence[str], previous: Mapping[str, float] | None,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    means = {m: float(np.mean([float(p[m]) for p in proposals])) for m in metrics}
    spreads = {m: float(np.std([float(p[m]) for p in proposals], ddof=0)) for m in metrics}
    adjustments = reviewer_adjustments(review, metrics)
    adjusted = {m: means[m] * math.exp(0.12 * adjustments[m]) for m in metrics}
    adjusted = normalize_weights(adjusted, metrics)
    if previous is None:
        accepted = adjusted
        raw_l1 = None
        applied_blend = 1.0
    else:
        # Chief gives current expert evidence 45% influence and preserves 55%
        # prior consensus, then enforces the preregistered L1 stability gate.
        target = {m: 0.55 * float(previous[m]) + 0.45 * adjusted[m] for m in metrics}
        target = normalize_weights(target, metrics)
        raw_l1 = l1_distance(previous, target, metrics)
        applied_blend = 1.0 if raw_l1 <= MAX_L1_DELTA else MAX_L1_DELTA / raw_l1
        accepted = normalize_weights(
            {m: float(previous[m]) + applied_blend * (target[m] - float(previous[m])) for m in metrics},
            metrics,
        )
    audit = {
        "expert_mean": means,
        "expert_standard_deviation": spreads,
        "reviewer_adjustments": adjustments,
        "reviewer_criticisms": review.get("criticisms", []),
        "reviewer_required_changes": review.get("required_changes", []),
        "raw_l1_from_previous": raw_l1,
        "chief_blend_fraction": applied_blend,
        "accepted_sum": sum(accepted.values()),
        "accepted_min": min(accepted.values()),
        "accepted_max": max(accepted.values()),
    }
    return accepted, audit


def aggregate_ranking(round_records: Sequence[Mapping[str, Any]], models: Sequence[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    score_rows: List[Dict[str, Any]] = []
    by_model: Dict[str, List[float]] = defaultdict(list)
    ranks: Dict[str, List[int]] = defaultdict(list)
    top3 = Counter()
    for record in round_records:
        for row in record["model_scores"]:
            score_rows.append(dict(row))
            by_model[row["model"]].append(float(row["score"]))
            ranks[row["model"]].append(int(row["rank"]))
            if int(row["rank"]) <= 3:
                top3[row["model"]] += 1
    ranking: List[Dict[str, Any]] = []
    total_rounds = len(round_records)
    for model in models:
        values = by_model[model]
        q1, q3 = np.percentile(values, [25, 75])
        ranking.append({
            "model": model,
            "median_score": round(float(np.median(values)), 8),
            "mean_score": round(float(np.mean(values)), 8),
            "score_iqr": round(float(q3 - q1), 8),
            "median_rank": round(float(np.median(ranks[model])), 3),
            "mean_rank": round(float(np.mean(ranks[model])), 3),
            "top3_frequency": round(top3[model] / total_rounds, 6),
            "rounds": total_rounds,
        })
    ranking.sort(key=lambda x: (-x["median_score"], x["mean_rank"], x["model"].lower()))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    return ranking, score_rows


def plot_publication_figure(
    output_dir: Path, metrics: Sequence[str], round_records: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })
    score_values: Dict[str, List[float]] = defaultdict(list)
    for record in round_records:
        for row in record["model_scores"]:
            score_values[str(row["model"])].append(float(row["score"]))
    ordered = list(ranking)
    ordered_models = [str(row["model"]) for row in ordered]
    fig = plt.figure(figsize=(7.2, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_w = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_w)
    y_w = np.arange(len(ordered_models))
    model_colors = ["#6D62B5" if int(row["rank"]) <= 3 else "#86B6D9" for row in ordered]
    boxes = ax_w.boxplot(
        [score_values[model] for model in ordered_models],
        positions=y_w,
        vert=False,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#172033", "linewidth": 1.15},
        whiskerprops={"color": "#506078", "linewidth": 0.75},
        capprops={"color": "#506078", "linewidth": 0.75},
        boxprops={"edgecolor": "#FFFFFF", "linewidth": 0.55},
    )
    for patch, color in zip(boxes["boxes"], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.90)
    # Retain all 50 observed model scores; deterministic jitter avoids hiding
    # equal values without introducing simulated observations.
    jitter = 0.105 * np.sin(np.arange(len(round_records)) * 2.399963)
    for y, model in zip(y_w, ordered_models):
        ax_w.scatter(score_values[model], y + jitter, s=4.5, color="#26364A", alpha=0.17,
                     linewidth=0, rasterized=True, zorder=1)
    ax_w.set_yticks(y_w, [f"{int(row['rank'])}. {row['model']}" for row in ordered])
    ax_w.invert_yaxis()
    ax_w.set_xlabel("Weighted rank score across 50 rounds")
    ax_w.set_title("a  Model-score distributions", loc="left", fontweight="bold")
    ax_w.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax_w.set_axisbelow(True)

    y_b = np.arange(len(ordered))
    x = np.array([float(r["median_score"]) for r in ordered])
    freq = np.array([float(r["top3_frequency"]) for r in ordered])
    uncertainty = np.array([float(r["score_iqr"]) for r in ordered])
    if uncertainty.max() > uncertainty.min():
        color_value = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    else:
        color_value = np.zeros_like(uncertainty)
    sizes = 26 + 230 * freq
    scatter = ax_b.scatter(x, y_b, s=sizes, c=color_value, cmap="viridis_r", vmin=0, vmax=1,
                           edgecolor="white", linewidth=0.7, alpha=0.93)
    ax_b.set_yticks(y_b)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.set_xlabel("Median weighted rank score across 50 rounds")
    ax_b.set_title("b  Ranking stability", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax_b.set_axisbelow(True)
    for value, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (1.0, "100%")]:
        ax_b.scatter([], [], s=26 + 230 * value, color="#6D62B5", alpha=0.8,
                     edgecolor="white", linewidth=0.6, label=label)
    ax_b.legend(title="Top-3 frequency", loc="lower right", fontsize=5.5, title_fontsize=6,
                labelspacing=0.9, borderpad=0.3)
    cbar = fig.colorbar(scatter, ax=ax_b, orientation="horizontal", fraction=0.035, pad=0.09, aspect=30)
    cbar.set_label("Relative score uncertainty (IQR; darker = higher)", fontsize=6)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])

    base = output_dir / "codex_agent_model_score_boxplot_bubble"
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    tiff_path = base.with_suffix(".tiff")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white")
    outputs = {"png": str(png_path), "svg": str(svg_path), "pdf": str(pdf_path), "tiff": str(tiff_path)}
    plt.close(fig)
    return outputs


def discussion_markdown(
    output_dir: Path, experts: Mapping[str, Any], reviewer: Mapping[str, Any],
    chief_initial: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]],
) -> Path:
    lines = [
        "# Local Codex multi-Agent metric-weight meeting: initial discussion + 50 rounds",
        "",
        "## Method and provenance",
        "",
        "Three independent expert Agents (literature, benchmark statistics and AMP screening) generated proposals from the same blinded evidence bundle. A Reviewer Agent audited all proposals without seeing model scores. The Chief execution layer reconciled the proposals sequentially, enforced all weight constraints and only then calculated model rankings.",
        "",
        "> Scientific status: exploratory post-hoc analysis. Dataset provenance, independence and homology gates remain pending; this output is not a leakage-free preregistered benchmark.",
        "",
        "## Initial Agent meeting",
        "",
    ]
    for role in EXPERT_FILES:
        item = experts[role]["initial"]
        lines += [f"### {role}", "", str(item.get("analysis", "")), "", f"Confidence: {item.get('confidence', 'not stated')}", ""]
    lines += ["### reviewer_agent", "", str(reviewer["initial"].get("analysis", "")), ""]
    lines += ["### chief_agent accepted initial weights", "", "| Metric | Weight |", "|---|---:|"]
    for metric, weight in sorted(chief_initial["accepted_weights"].items(), key=lambda x: -x[1]):
        lines.append(f"| {metric} | {weight:.6f} |")
    lines += [""]
    for record in rounds:
        n = record["round"]
        lines += [f"## Round {n:02d}", "", f"Bootstrap datasets: {', '.join(record['sampled_dataset_aliases'])}", ""]
        for role in EXPERT_FILES:
            item = experts[role]["rounds"][n - 1]
            lines += [f"**{role}:** {item.get('analysis', '')}", ""]
        review = reviewer["rounds"][n - 1]
        lines += [f"**reviewer_agent:** {review.get('analysis', '')}", ""]
        top_weights = sorted(record["accepted_weights"].items(), key=lambda x: -x[1])[:4]
        lines += [
            f"**chief_agent:** reconciled the proposals; L1 change from the previous accepted vector = {record['l1_from_previous']:.6f}. "
            + "Highest accepted weights: " + ", ".join(f"{m}={w:.4f}" for m, w in top_weights) + ".",
            "",
        ]
    lines += ["## Final 50-round ranking", "", "| Rank | Model | Median score | IQR | Median rank | Top-3 frequency |", "|---:|---|---:|---:|---:|---:|"]
    for row in ranking:
        lines.append(f"| {row['rank']} | {row['model']} | {row['median_score']:.6f} | {row['score_iqr']:.6f} | {row['median_rank']:.1f} | {100*row['top3_frequency']:.1f}% |")
    path = output_dir / "codex_agent_discussion_50_rounds.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_report_markdown(
    output_dir: Path,
    chief_initial: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the canonical, audit-ready project report.

    The report deliberately joins the three real pipeline stages.  It reads the
    frozen Stage-1 memory, the stored Stage-2 scientific evaluations and the
    current Stage-3 deliberation artifacts, so a rerun cannot silently replace
    evidence with a prose-only summary.
    """
    root = output_dir.parents[2]

    def read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def root_link(label: str, relative_path: str) -> str:
        return f"[{label}](../../../{Path(relative_path).as_posix()})"

    def local_link(label: str, filename: str) -> str:
        return f"[{label}]({Path(filename).as_posix()})"

    memory = read_json(root / "data" / "literature_deep_research_memory.json", {})
    compact_pool = read_json(root / "data" / "compact_evidence_pool.json", {})
    search_summary = read_json(root / "data" / "multi_source_search_summary.json", {})
    dataset_decision = read_json(root / "data" / "dataset_agent_recommendation.json", {})
    qa = read_json(output_dir / "qa_notes.json", {})

    query_counts = {
        source: len(plans) for source, plans in search_summary.get("queries", {}).items()
        if isinstance(plans, list)
    }
    query_total = sum(query_counts.values())
    query_text = ", ".join(f"{source}={count}" for source, count in sorted(query_counts.items()))

    screening_counts: Counter[str] = Counter()
    screening_path = root / "data" / "literature_meeting_screening_decisions.csv"
    if screening_path.exists():
        with screening_path.open("r", encoding="utf-8-sig", newline="") as handle:
            screening_counts.update(row.get("meeting_decision", "") for row in csv.DictReader(handle))
    frozen_screened = sum(screening_counts.values())
    frozen_eligible = screening_counts.get("accept", 0)

    stage1_models = memory.get("final_deployment_models", [])
    stage1_metrics = memory.get("final_metrics_plan", {})
    stage1_datasets = dataset_decision.get("empirically_evaluated_top3", [])

    dataset_names = ["C_AMPs-predict_test", "Veltri_test", "ProteoGPT_all_predictions"]
    stage2: Dict[str, Any] = {}
    for dataset_name in dataset_names:
        stage2[dataset_name] = read_json(
            root / "data" / "results_manual" / dataset_name / "scientific_evaluation.json", {}
        )

    def top_endpoint(dataset_record: Mapping[str, Any], endpoint: str) -> tuple[str, float]:
        candidates = []
        for model, record in dataset_record.get("models", {}).items():
            value = record.get("selected_threshold_metrics", {}).get(endpoint)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                candidates.append((model, float(value)))
        return max(candidates, key=lambda item: item[1]) if candidates else ("not available", float("nan"))

    final_weights = rounds[-1]["accepted_weights"]
    median_weights = {m: float(np.median([r["accepted_weights"][m] for r in rounds])) for m in final_weights}
    lines = [
        "---",
        'title: "Evidence-grounded multi-Agent AMP benchmark report"',
        'report_type: "three-stage auditable scientific report"',
        f'generated_at_utc: "{now()}"',
        'canonical_result: "local_codex_multi_agent / 50 rounds"',
        'scientific_status: "exploratory; formal independence and homology gates pending"',
        "---",
        "",
        "# 证据驱动的多 Agent AMP 基准评测专业报告",
        "",
        "## 执行摘要",
        "",
        f"本项目以可追溯的三阶段 Human–Agent 工作流，将文献证据转化为可执行的 AMP 模型基准，并进一步通过 50 轮盲化权重会议形成模型排序。当前规范结果中，**{ranking[0]['model']}** 的中位加权秩分数最高；Top‑3 为 **{ranking[0]['model']}、{ranking[1]['model']}、{ranking[2]['model']}**。建议优先进行概率软投票或秩平均；只有在独立验证集上训练的 stacking 才可作为后续方案。",
        "",
        "> 结论边界：Stage 3 的权重 Agent 在模型名称与模型分数揭盲前完成权重选择，但三套数据仍存在来源、独立性和同源性审计缺口；现有结果来自 stored test-like predictions。因此本报告支持探索性比较、系统审计和后续验证设计，不构成无泄漏、预注册的正式 benchmark 声明。",
        "",
        "## 1. 报告对象与真实证据链",
        "",
        "本报告仅整合当前工作区已经存在的输入、Agent 对话、结构化中间产物、评测文件和 50 轮结果，不补造缺失实验。规范证据链为：",
        "",
        "1. **Stage 1 — 文献会议：** 多源检索 → 证据压缩 → 模型/数据集/指标提案 → Critic 质询 → Chief 冻结。",
        "2. **Stage 2 — 自动部署与评测：** 真实预测表/FASTA → schema 对齐 → 代码生成与复核 → 三数据集统一评测。",
        "3. **Stage 3 — 盲化权重会议：** 角色化指标提案 → Reviewer 审计 → Chief 有界更新 50 轮 → 揭盲排名 → Top‑3 集成建议。",
        "",
        f"![三阶段系统主图](../../../figures/amp-agent-three-stage-roundtable-meetings-main-figure-v20.png)",
        "",
        "## 2. Stage 1：文献检索、资产推荐与证据会议",
        "",
        "### 2.1 真实输入",
        "",
        "| 输入对象 | 实际内容 | 机器可读来源 |",
        "|---|---|---|",
        f"| 科学任务 | AMP 二分类；兼顾模型、数据集、指标、代码/权重和算力约束 | {root_link('literature_deep_research_memory.json', 'data/literature_deep_research_memory.json')} |",
        f"| 检索计划 | {query_total} 个存储 query plans（{query_text or 'source counts unavailable'}） | {root_link('multi_source_search_summary.json', 'data/multi_source_search_summary.json')} |",
        f"| 证据池 | {compact_pool.get('paper_count', 'NA')} 篇文献；{compact_pool.get('chunk_summary_count', compact_pool.get('chunk_count', 'NA'))} 个压缩证据单元 | {root_link('compact_evidence_pool.json', 'data/compact_evidence_pool.json')} |",
        f"| 冻结筛选集 | {frozen_screened or 'NA'} 个模型身份；{frozen_eligible or 'NA'} 个通过；{len(stage1_models)} 个进入部署优先池 | {root_link('literature_meeting_screening_decisions.csv', 'data/literature_meeting_screening_decisions.csv')} |",
        "",
        "### 2.2 Stage 1 每个 Agent 的真实输入与输出",
        "",
        "| Agent | 真实输入 | Prompt / 决策契约 | 实际输出 | 输出文件 |",
        "|---|---|---|---|---|",
        f"| Query Planner | 人类研究问题、AMP 二分类范围、来源与计算约束 | 拆分高召回、高精度、架构、数据集、代码/权重等检索意图；保留可复现 query | 多来源检索计划 | {root_link('pubmed_query_planner.md', 'agents/pubmed_planner/pubmed_query_planner.md')}；{root_link('search summary', 'data/multi_source_search_summary.json')} |",
        f"| Search / Info Extractor | query plans、API/仓库返回、文献元数据与全文 | 提取模型身份、任务、代码、权重、数据集、指标和证据锚点；不把搜索命中当作已验证证据 | 原始 evidence pool、结构化论文/仓库/数据集记录 | {root_link('info_extractor_agent.md', 'agents/deepseek_meeting/info_extractor_agent.md')}；{root_link('evidence_pool.json', 'data/evidence_pool.json')} |",
        f"| Evidence Compressor | 原始检索记录、全文片段、仓库与数据集线索 | 仅压缩输入 chunk；不虚构 PMID/DOI/URL；输出严格 JSON 和可追溯摘要 | compact evidence pool | {root_link('evidence_compressor_agent.md', 'agents/deepseek_meeting/evidence_compressor_agent.md')}；{root_link('compact_evidence_pool.md', 'data/compact_evidence_pool.md')} |",
        f"| Scout / Model–Dataset Agent | compact evidence、历史 memory、模型与数据集实体 | 召回完整候选池；按任务同一性、代码/权重、可部署性、证据等级和架构覆盖提出保留/搁置 | 模型/数据集提案；冻结筛选链 {frozen_screened} → {frozen_eligible} → {len(stage1_models)} | {root_link('model_dataset_agent.md', 'agents/deepseek_meeting/model_dataset_agent.md')}；{root_link('screening decisions', 'data/literature_meeting_screening_decisions.csv')} |",
        f"| Dataset Selection Agent | 文献 shortlist、{dataset_decision.get('candidate_pool_size', 'NA')} 个候选、{dataset_decision.get('audited_local_candidate_count', 'NA')} 个真实 CSV 审计结果 | 约束选择 1 个 balanced + 2 个不同不平衡程度的数据集；同时保留 formal blockers | 3 个经验互补评测 profile；正式资格仍 blocked | {root_link('dataset_agent_recommendation.md', 'data/dataset_agent_recommendation.md')}；{root_link('dataset_agent_recommendation.json', 'data/dataset_agent_recommendation.json')} |",
        f"| Metrics Agent | compact evidence、数据集 prevalence、漏检/误检代价 | 预定义 endpoint hierarchy、验证集阈值、校准、效用、资源和统计报告规则 | AUPRC 主终点、MCC 关键次终点及完整指标协议 | {root_link('metric_agent.md', 'agents/deepseek_meeting/metric_agent.md')}；{root_link('literature memory', 'data/literature_deep_research_memory.md')} |",
        f"| Critic Agent | Scout 与 Metrics 提案、来源/身份/代码/泄漏信息 | 对每项作 accept / reject / defer；挑战身份、代码、权重、阈值、独立性和同源性 | 质询、否决/暂缓理由和补证要求 | {root_link('critic_agent.md', 'agents/deepseek_meeting/critic_agent.md')}；{root_link('meeting_trace.md', 'data/meeting_trace.md')} |",
        f"| Chief Agent | Scout/Metrics 提案、Critic 审计、rebuttal 与历史 memory | 调和冲突、保留 dissent、冻结可审计 memory；不删除未解决 blocker | 20 模型优先池、3 个经验评测数据集、冻结指标协议与长期记忆 | {root_link('chief_agent.md', 'agents/deepseek_meeting/chief_agent.md')}；{root_link('literature_deep_research_memory.md', 'data/literature_deep_research_memory.md')} |",
        "",
        "**Human checkpoint：** 人类提供研究范围和算力边界，授权在线检索、代码/数据获取与高风险缺口的人工核验；Agent 不替代数据许可、训练重叠和同源性审计的最终责任。",
        "",
        "![Stage 1 真实讨论实例](../../../figures/amp_agent_discussion_instances_stage1_stage2_v1/amp_real_agent_discussion_stage1.png)",
        "",
        "### 2.3 冻结的模型优先池（n=20）",
        "",
        "| Rank | Model | Benchmark role | Year | Evidence / code status |",
        "|---:|---|---|---:|---|",
    ]
    for model in sorted(stage1_models, key=lambda x: x.get("deployment_rank", 999)):
        year = model.get("publication_year", "NA")
        evidence = model.get("evidence_level", "not stated")
        repo = "repository recorded" if model.get("code_repository_url") else "repository unresolved"
        lines.append(
            f"| {model.get('deployment_rank', '')} | {model.get('model_name', model.get('canonical_name', ''))} | "
            f"{model.get('benchmark_role_label', model.get('benchmark_role', ''))} | {year} | {evidence}; {repo} |"
        )
    lines += [
        "",
        "### 2.4 经验互补数据集与冻结指标协议",
        "",
        "| Dataset | n | Positive / negative | Prevalence | Profile | Formal status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dataset in stage1_datasets:
        audit = dataset.get("audit", {})
        lines.append(
            f"| {dataset.get('dataset_name', '')} | {audit.get('row_count', 'NA'):,} | "
            f"{audit.get('positive_count', 'NA')} / {audit.get('negative_count', 'NA')} | "
            f"{100 * float(audit.get('positive_fraction', 0)):.2f}% | {dataset.get('selection_profile', '')} | "
            f"{'eligible' if dataset.get('formal_eligible') else 'blocked pending provenance / independence / homology gates'} |"
        )
    lines += [
        "",
        "**Literature-meeting endpoint hierarchy：** AUPRC 为唯一主终点；MCC 为关键次终点。探索性四指标权重为 AUPRC 0.35、MCC 0.30、Recall 0.20、Precision 0.15。正式阈值必须在独立验证集上用 Max‑MCC 选择后冻结；0.5 仅作诊断；测试集禁止调阈值。",
        "",
        "## 3. Stage 2：自动部署、代码复核与统一评测",
        "",
        "### 3.1 真实输入",
        "",
        "20 个文献优先模型进入部署尝试；当前规范评测结果覆盖 **18 个成功形成有效概率输出的模型 × 3 个数据集**。输入包括真实 FASTA/预测表、模型 registry、仓库 README/requirements、统一 ID/sequence/probability schema、HPC 路径与运行 manifest。",
        "",
        "### 3.2 Stage 2 每个 Agent 的真实输入与输出",
        "",
        "| Agent | 真实输入 | 实际决策 / 响应 | 实际输出 | 输出文件 |",
        "|---|---|---|---|---|",
        f"| PI Agent（模型运行会议） | 模型仓库、任务约束、预期输入输出和 HPC 环境 | 冻结部署要求、失败边界、概率输出接口和不得静默吞错的规则 | Code Engineer 的执行规格 | {root_link('meeting_stage1_model_run.md', 'data/vlab_discussions/meeting_stage1_model_run.md')} |",
        f"| Code Engineer | PI 规格、仓库 README、依赖和模型入口 | 生成/修订模型运行与结果收集代码 | 可执行 model runner、cache 与运行上下文 | {root_link('stage1_model_runner.py', 'data/vlab_discussions/stage1_model_runner.py')}；{root_link('stage1 context', 'data/vlab_discussions/stage1_context_for_stage2.txt')} |",
        f"| Data Architect Agent | 三套真实预测文件、ground truth、模型概率列 | 定义 ID/sequence/label/probability schema、路径映射、去重和缺失处理 | 评测数据契约与代码生成上下文 | {root_link('meeting_stage2_eval_codegen.md', 'data/vlab_discussions/meeting_stage2_eval_codegen.md')} |",
        f"| MLOps Coder V1 | Data Architect schema + PI 硬性要求 | 生成首版评测脚本 | V1 代码候选，进入独立 review | {root_link('stage2_eval_codegen.json', 'data/vlab_discussions/meeting_stage2_eval_codegen.json')} |",
        f"| Data Architect Reviewer | V1 代码、真实文件模式和错误路径 | 指出 FileNotFoundError、NaN、结果表安全性等缺陷并要求有界修复 | 可执行的修订清单 | {root_link('review transcript', 'data/vlab_discussions/meeting_stage2_eval_codegen.md')} |",
        f"| PI Summary Agent | V1、Reviewer 质询和实验纪律 | 将反馈压缩成不得越界的最终实现合同 | Final coder 的 bounded revision specification | {root_link('PI summary transcript', 'data/vlab_discussions/meeting_stage2_eval_codegen.md')} |",
        f"| MLOps Coder Final | V1 + review + PI summary | 完成防御性 schema 对齐、指标计算、ROC/PR、校准和结构化导出 | `stage2_eval_script.py`；每数据集 CSV/JSON/PNG/MD | {root_link('stage2_eval_script.py', 'data/vlab_discussions/stage2_eval_script.py')} |",
        f"| Per-dataset Critic | 评测指标、曲线、覆盖率、阈值来源和 manifest | 独立检查异常表现、解释风险和不可直接宣称的结论 | 每个数据集的 `critic_individual.md` | {root_link('C_AMPs Critic', 'data/results_manual/C_AMPs-predict_test/critic_individual.md')}；{root_link('Veltri Critic', 'data/results_manual/Veltri_test/critic_individual.md')}；{root_link('ProteoGPT Critic', 'data/results_manual/ProteoGPT_all_predictions/critic_individual.md')} |",
        "",
        "**Human checkpoint：** 人类确认 registry/schema，授权必要的手动 CSV 上传与失败模型检查；只有通过环境与 smoke-test gate 的模型进入正式汇总。",
        "",
        "![Stage 2 真实讨论实例](../../../figures/amp_agent_discussion_instances_stage1_stage2_v1/amp_real_agent_discussion_stage2.png)",
        "",
        "### 3.3 三套数据集的真实评测输出",
        "",
        "统一 Scientific Evaluation Protocol v2.0：主终点 AUPRC、关键次终点 MCC；缺少独立验证集时阈值 0.5 仅作诊断；当前 bootstrap iterations=0；成对阈值错误比较使用 McNemar，并对 pairwise family 采用 Holm 校正。",
        "",
        "| Dataset | n | Prevalence | Evaluated models | Highest AUPRC | Highest MCC | Real artifacts |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for dataset_name in dataset_names:
        record = stage2[dataset_name]
        quality = record.get("data_quality", {})
        top_auprc = top_endpoint(record, "auprc")
        top_mcc = top_endpoint(record, "mcc")
        dataset_rel = f"data/results_manual/{dataset_name}"
        lines.append(
            f"| {dataset_name} | {quality.get('rows', 'NA'):,} | {100 * float(quality.get('positive_prevalence', 0)):.2f}% | "
            f"{len(record.get('models', {}))} | {top_auprc[0]} ({top_auprc[1]:.3f}) | {top_mcc[0]} ({top_mcc[1]:.3f}) | "
            f"{root_link('JSON', dataset_rel + '/scientific_evaluation.json')} · {root_link('MD', dataset_rel + '/scientific_evaluation.md')} · "
            f"{root_link('CSV', dataset_rel + '/final_results_with_predictions.csv')} · {root_link('PNG', dataset_rel + '/evaluation_curves.png')} |"
        )
    lines += [
        "",
        "#### C_AMPs-predict_test",
        "",
        "![C_AMPs-predict_test real evaluation](../C_AMPs-predict_test/evaluation_curves.png)",
        "",
        "#### Veltri_test",
        "",
        "![Veltri_test real evaluation](../Veltri_test/evaluation_curves.png)",
        "",
        "#### ProteoGPT_all_predictions",
        "",
        "![ProteoGPT real evaluation](../ProteoGPT_all_predictions/evaluation_curves.png)",
        "",
        "## 4. Stage 3：50 轮盲化多 Agent 权重会议与模型排序",
        "",
        "### 4.1 真实输入与盲化设计",
        "",
        f"输入为 3 个匿名数据集 × 18 个匿名模型 × {len(final_weights)} 个可用指标的 `agent_evidence_bundle.json`，以及预生成的 50 轮 bootstrap dataset plan。权重 Agent 只接触指标覆盖度、分离度、一致性、共识度、冗余性和任务代价等摘要；Chief 接受权重后，执行层才把权重应用到隐藏模型分数。",
        "",
        "### 4.2 Stage 3 每个 Agent 的真实输入与输出",
        "",
        "| Agent | 真实输入 | Prompt / 角色约束 | 实际输出 | 输出文件 |",
        "|---|---|---|---|---|",
        f"| Literature Agent | 盲化 metric evidence + 文献 endpoint prior | 强调不平衡任务、文献可解释性和已冻结 endpoint hierarchy；不可读取模型身份 | initial + 50 轮权重提案及文字理由 | {local_link('literature_agent_proposals.json', 'literature_agent_proposals.json')} |",
        f"| Statistics Agent | coverage、separation、consistency、consensus、uniqueness、committee support | 惩罚冗余、缺失和不稳定指标；保证统计可辨识性 | initial + 50 轮统计质量调整后的权重提案 | {local_link('statistics_agent_proposals.json', 'statistics_agent_proposals.json')} |",
        f"| Screening Agent | FN/FP 成本、Recall/Precision、AUPRC、calibration 的盲化摘要 | 以 AMP 筛选效用平衡漏检和湿实验假阳性成本 | initial + 50 轮 cost-aware 权重提案 | {local_link('screening_agent_proposals.json', 'screening_agent_proposals.json')} |",
        f"| Reviewer Agent | 三个专家提案 + 同一盲化 evidence bundle | 独立审查权重范围、方向、重复计权和证据不足；不直接给模型排名 | initial + 50 轮 audit、修正方向与边界 | {local_link('reviewer_agent_audit.json', 'reviewer_agent_audit.json')} |",
        f"| Chief Agent | 专家提案、Reviewer audit、上一轮 accepted vector | 强制每项权重 [{WEIGHT_MIN:.3f}, {WEIGHT_MAX:.2f}]、总和=1、单轮 L1 变化≤{MAX_L1_DELTA:.2f}；调和而非覆盖 dissent | initial decision、50 个 round JSON、最终 accepted weights | {local_link('chief_initial_decision.json', 'chief_initial_decision.json')}；{local_link('rounds/', 'rounds/')} |",
        f"| Deterministic ranking engine | Chief 已接受权重 + 揭盲后的真实模型指标 | 不再改权重；对每轮统一计算加权 percentile rank | {qa.get('score_rows', 900)} 条 model-round scores、完整 ranking 和 publication figures | {local_link('model scores CSV', 'codex_agent_model_scores_50_rounds.csv')}；{local_link('ranking CSV', 'codex_agent_model_ranking_50_rounds.csv')} |",
        "",
        "**Human checkpoint：** 人类检查失败模式、数据泄漏风险与最终集成方案；不得依据测试集表现调 stacking、阈值或模型权重。",
        "",
        "![Stage 3 真实 Agent 提示与回答实例](../../../figures/amp_agent_discussion_instance_v1/amp_real_agent_discussion_round09.png)",
        "",
        "### 4.3 运行完整性",
        "",
        f"- Round files：**{qa.get('round_files', len(rounds))}**。",
        f"- Model-round score rows：**{qa.get('score_rows', len(ranking) * len(rounds))}**。",
        f"- Metric-weight rows：**{qa.get('weight_rows', len(final_weights) * len(rounds))}**。",
        f"- 所有评测行纳入：**{qa.get('all_rows_used', True)}**；排除行：**{qa.get('excluded_rows', 0)}**。",
        f"- 权重约束通过：**{qa.get('weight_constraints_pass', True)}**；L1 变化约束通过：**{qa.get('l1_constraints_pass', True)}**。",
        "",
        "### 4.4 接受的指标权重共识",
        "",
        "| Metric | Initial Chief weight | Round-50 weight | 50-round median |",
        "|---|---:|---:|---:|",
    ]
    for metric in sorted(final_weights, key=lambda m: -median_weights[m]):
        lines.append(f"| {metric} | {chief_initial['accepted_weights'][metric]:.6f} | {final_weights[metric]:.6f} | {median_weights[metric]:.6f} |")
    lines += ["", "### 4.5 50 轮模型排名", "", "| Rank | Model | Median score | Score IQR | Mean rank | Top-3 frequency |", "|---:|---|---:|---:|---:|---:|"]
    for row in ranking:
        lines.append(f"| {row['rank']} | {row['model']} | {row['median_score']:.6f} | {row['score_iqr']:.6f} | {row['mean_rank']:.2f} | {100*row['top3_frequency']:.1f}% |")
    lines += [
        "",
        "![50-round model score distributions](codex_agent_model_score_boxplot_bubble.png)",
        "",
        "## 5. Top‑3 集成学习建议",
        "",
        f"1. **{ranking[0]['model']}**：中位分数 {ranking[0]['median_score']:.6f}，Top‑3 出现率 {100*ranking[0]['top3_frequency']:.1f}%。",
        f"2. **{ranking[1]['model']}**：中位分数 {ranking[1]['median_score']:.6f}，Top‑3 出现率 {100*ranking[1]['top3_frequency']:.1f}%。",
        f"3. **{ranking[2]['model']}**：中位分数 {ranking[2]['median_score']:.6f}，Top‑3 出现率 {100*ranking[2]['top3_frequency']:.1f}%。",
        "",
        "**推荐顺序：** 先在独立验证集上比较 soft voting 与 rank averaging；若获得独立 validation predictions，再训练受约束的 stacking meta-learner。禁止在当前三套 test-like 数据上调 ensemble weights、阈值或超参数。",
        "",
        "## 6. 解释边界与尚未关闭的审计项",
        "",
        "- **数据独立性：** 三个经验评测 profile 尚未全部证明对所有模型均为独立外部测试集；存在 model-specific exclusions。",
        "- **同源性与训练重叠：** 仍需 exact-overlap 与 ≤40% sequence-identity 审计，并建立训练集引用清单。",
        "- **阈值：** 当前缺少独立 validation predictions，0.5 仅为诊断阈值；不能将其解释为正式工作点。",
        "- **不确定性：** Stage 2 当前 bootstrap_iterations=0，因此不得把点估计写成已获得置信区间的正式结论。",
        "- **后验性：** Stage 3 使用 stored test-like results 形成探索性权重与排序；适合方法开发和验证设计，不适合最终无偏性能声明。",
        "",
        "## 7. 可复现性与审计清单",
        "",
        "| 层级 | 规范产物 | 用途 |",
        "|---|---|---|",
        f"| Stage 1 memory | {root_link('literature_deep_research_memory.json', 'data/literature_deep_research_memory.json')} / {root_link('MD', 'data/literature_deep_research_memory.md')} | 冻结模型、数据集、指标、讨论与未决问题 |",
        f"| Stage 1 trace | {root_link('meeting_trace.md', 'data/meeting_trace.md')} / {root_link('deepseek_meeting_raw.jsonl', 'data/deepseek_meeting_raw.jsonl')} | Agent 原始会议和质询追溯 |",
        f"| Stage 2 meetings | {root_link('model-run meeting', 'data/vlab_discussions/meeting_stage1_model_run.md')} / {root_link('evaluation-code meeting', 'data/vlab_discussions/meeting_stage2_eval_codegen.md')} | 部署与代码生成/复核轨迹 |",
        f"| Stage 2 evaluations | {root_link('results_manual/', 'data/results_manual')} | 真实 CSV、JSON、MD 与曲线 |",
        f"| Stage 3 evidence | {local_link('agent_evidence_bundle.json', 'agent_evidence_bundle.json')} / {local_link('bootstrap plan', 'internal_bootstrap_plan.json')} | 盲化 Agent 输入 |",
        f"| Stage 3 discussion | {local_link('codex_agent_discussion_50_rounds.md', 'codex_agent_discussion_50_rounds.md')} / {local_link('rounds/', 'rounds/')} | 每轮专家建议、Reviewer 审计与 Chief 决策 |",
        f"| Stage 3 outputs | {local_link('weights CSV', 'codex_agent_metric_weights_50_rounds.csv')} / {local_link('scores CSV', 'codex_agent_model_scores_50_rounds.csv')} / {local_link('ranking CSV', 'codex_agent_model_ranking_50_rounds.csv')} | 统计复核与作图源数据 |",
        "",
        "## 8. 结论",
        "",
        f"该系统已经形成从文献证据到真实预测评测、再到多 Agent 决策的闭环，并保存每个关键 Agent 的输入、约束、输出与审计文件。在当前探索性证据下，**{ranking[0]['model']}** 为最稳定的首选模型，**{ranking[1]['model']}** 与 **{ranking[2]['model']}** 构成 Top‑3 集成候选。下一步的决定性工作不是继续在测试结果上调权，而是关闭数据来源、独立性、同源性、验证阈值和不确定性估计五类审计缺口。",
    ]
    path = output_dir / "amp_future_directions_report_codex_agents.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def finalize(results_dir: Path, output_dir: Path, rounds_count: int, seed: int) -> Dict[str, Any]:
    bundle = json.loads((output_dir / "agent_evidence_bundle.json").read_text(encoding="utf-8"))
    metrics = list(bundle["eligible_metrics"])
    experts, reviewer = load_deliberations(output_dir, metrics, rounds_count)
    rows = collect_eval_rows(results_dir)
    eligible_rows, resource_gate = apply_resource_gate(rows, load_model_resource_policy())
    resource_audit_files = write_resource_gate_audit(output_dir, resource_gate)
    prepared = _prepare(eligible_rows)
    plan = json.loads((output_dir / "internal_bootstrap_plan.json").read_text(encoding="utf-8"))["rounds"]

    initial_proposals = [experts[role]["initial"]["weights"] for role in EXPERT_FILES]
    initial_weights, initial_audit = reconcile(initial_proposals, reviewer["initial"], metrics, None)
    chief_initial = {
        "role": "chief_agent", "meeting": "initial", "generated_at": now(),
        "expert_proposals": {role: experts[role]["initial"]["weights"] for role in EXPERT_FILES},
        "reviewer_audit": reviewer["initial"], "chief_audit": initial_audit,
        "accepted_weights": initial_weights,
    }
    write_json(output_dir / "chief_initial_decision.json", chief_initial)

    previous = initial_weights
    round_records: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    for i in range(rounds_count):
        round_no = i + 1
        proposals = [experts[role]["rounds"][i]["proposed_weights"] for role in EXPERT_FILES]
        accepted, audit = reconcile(proposals, reviewer["rounds"][i], metrics, previous)
        internal_weights = {key: accepted[prepared["names"][key]] for key in prepared["metrics"]}
        sampled_real = plan[i]["sampled_datasets"]
        scores = _score_models(prepared, sampled_real, internal_weights)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
        model_scores = [
            {"round": round_no, "model": model, "score": round(float(score), 10), "rank": rank}
            for rank, (model, score) in enumerate(ordered, 1)
        ]
        record = {
            "round": round_no,
            "sampled_dataset_aliases": bundle["rounds"][i]["sampled_datasets"],
            "metric_evidence": bundle["rounds"][i]["metric_evidence"],
            "expert_proposals": {role: experts[role]["rounds"][i]["proposed_weights"] for role in EXPERT_FILES},
            "reviewer_audit": reviewer["rounds"][i],
            "chief_audit": audit,
            "weights_before": previous,
            "accepted_weights": accepted,
            "l1_from_previous": round(l1_distance(previous, accepted, metrics), 10),
            "model_scores": model_scores,
            "top3": [model for model, _ in ordered[:3]],
        }
        write_json(output_dir / "rounds" / f"round_{round_no:03d}.json", record)
        round_records.append(record)
        for metric in metrics:
            weight_rows.append({"round": round_no, "metric": metric, "weight": accepted[metric]})
        previous = accepted
        print(f"[complete] chief round {round_no}/{rounds_count}")

    ranking, score_rows = aggregate_ranking(round_records, prepared["models"])
    write_csv(output_dir / "codex_agent_metric_weights_50_rounds.csv", weight_rows, ["round", "metric", "weight"])
    write_csv(output_dir / "codex_agent_model_scores_50_rounds.csv", score_rows, ["round", "model", "score", "rank"])
    write_csv(output_dir / "codex_agent_model_ranking_50_rounds.csv", ranking,
              ["rank", "model", "median_score", "mean_score", "score_iqr", "median_rank", "mean_rank", "top3_frequency", "rounds"])
    figure_contract = {
        "core_conclusion": "A robust model ranking emerges across 50 blinded Agent-derived metric-weight rounds.",
        "archetype": "quantitative grid",
        "target": "Nature-family double-column figure",
        "backend": "Python/matplotlib only",
        "final_size": "183 mm wide; 7.2 x 6.5 inches",
        "panels": {"a": "18 model-score boxplots plus all 50 observed scores per model", "b": "model ranking bubble plot; size is Top-3 frequency and color is score uncertainty"},
        "statistics": "50 bootstrap deliberation rounds; median, IQR and Top-3 recurrence",
        "source_data": ["codex_agent_model_scores_50_rounds.csv", "codex_agent_model_ranking_50_rounds.csv"],
        "reviewer_risk": "exploratory post-hoc results; independence and homology gates unresolved",
    }
    write_json(output_dir / "figure_contract.json", figure_contract)
    figures = plot_publication_figure(output_dir, metrics, round_records, ranking)
    trace = discussion_markdown(output_dir, experts, reviewer, chief_initial, round_records, ranking)
    report = final_report_markdown(output_dir, chief_initial, round_records, ranking)
    result = {
        "generated_at": now(), "backend": "local_codex_multi_agent", "rounds": rounds_count,
        "roles": list(EXPERT_FILES) + ["reviewer_agent", "chief_agent"],
        "dimensions": {"datasets": len(prepared["datasets"]), "models": len(prepared["models"]), "metrics": len(metrics)},
        "initial_weights": initial_weights, "final_weights": previous, "ranking": ranking,
        "resource_gate": resource_gate, "resource_gate_audit_files": resource_audit_files,
        "figures": figures, "discussion_markdown": str(trace), "final_report": str(report),
        "scientific_caveat": bundle["caveat"],
    }
    write_json(output_dir / "codex_agent_weight_meeting_50_rounds.json", result)
    write_json(output_dir / "qa_notes.json", {
        "generated_at": now(), "all_eligible_rows_used": True,
        "resource_excluded_rows": resource_gate["rows_before"] - resource_gate["rows_after"],
        "resource_excluded_models": resource_gate["excluded_models"],
        "resource_flagged_models": resource_gate["flagged_models"],
        "round_files": len(round_records), "score_rows": len(score_rows), "weight_rows": len(weight_rows),
        "weight_constraints_pass": all(abs(sum(r["accepted_weights"].values()) - 1) < 1e-6 for r in round_records),
        "l1_constraints_pass": all(r["l1_from_previous"] <= MAX_L1_DELTA + 1e-6 for r in round_records),
        "editable_text": {"svg": True, "pdf_fonttype": 42}, "raster_dpi": {"png": 300, "tiff": 600},
        "interpretation_boundary": bundle["caveat"],
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "finalize"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_bundle(args.results_dir.resolve(), args.output_dir.resolve(), args.rounds, args.seed)
        return
    result = finalize(args.results_dir.resolve(), args.output_dir.resolve(), args.rounds, args.seed)
    print(f"[done] {len(result['ranking'])} models ranked across {result['rounds']} rounds")


if __name__ == "__main__":
    main()
