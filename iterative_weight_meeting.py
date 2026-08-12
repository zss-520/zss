# -*- coding: utf-8 -*-
"""Evidence-driven, reproducible metric-weight deliberation.

The committee starts from literature-meeting weights when available, otherwise
from an equal-weight prior. In every round it bootstraps datasets, reviews metric
coverage, separation, cross-dataset consistency, consensus, and redundancy, then
updates the weights. Model rankings are recomputed after every update.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from model_resource_policy import (
    apply_resource_gate,
    load_model_resource_policy,
    write_resource_gate_audit,
)


LOWER_IS_BETTER = {"brierscore", "ece", "logloss", "nll"}
NON_RANKING_FIELDS = {"threshold"}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _normalize(values: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    cleaned = {key: max(0.0, float(values.get(key, 0.0))) for key in keys}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: cleaned[key] / total for key in keys}


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(aa, bb)[0, 1], -1.0, 1.0))


def _rank_scores(model_values: Mapping[str, float], higher_is_better: bool) -> Dict[str, float]:
    """Return tie-aware percentile rank scores in [0, 1]."""
    items = sorted(model_values.items(), key=lambda item: item[1], reverse=higher_is_better)
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: 1.0}
    result: Dict[str, float] = {}
    index = 0
    while index < len(items):
        end = index + 1
        while end < len(items) and math.isclose(items[end][1], items[index][1], rel_tol=1e-12, abs_tol=1e-12):
            end += 1
        mean_position = (index + end - 1) / 2.0
        score = 1.0 - mean_position / (len(items) - 1)
        for model, _ in items[index:end]:
            result[model] = score
        index = end
    return result


def _prepare(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    names: Dict[str, str] = {}
    models = set()
    datasets = set()
    for row in rows:
        metric_key = str(row.get("metric_key") or row.get("metric") or "").strip().lower()
        metric_name = str(row.get("metric") or metric_key).strip()
        model = str(row.get("model") or "").strip()
        dataset = str(row.get("dataset") or "").strip()
        value = _safe_float(row.get("value"))
        if not metric_key or metric_key in NON_RANKING_FIELDS or not model or not dataset or value is None:
            continue
        values[metric_key][dataset][model] = value
        names.setdefault(metric_key, metric_name)
        models.add(model)
        datasets.add(dataset)

    eligible = []
    expected = max(1, len(models) * len(datasets))
    for metric_key, by_dataset in values.items():
        flat = [v for model_values in by_dataset.values() for v in model_values.values()]
        coverage = len(flat) / expected
        if coverage >= 0.5 and len(flat) >= 2 and np.ptp(flat) > 1e-12:
            eligible.append(metric_key)

    # Remove metrics that induce exactly the same model ordering in every
    # dataset (for example, AUPRC-Lift is a prevalence-scaled AUPRC). Keeping
    # both would count the same evidence twice.
    deduplicated = []
    duplicate_metrics: Dict[str, str] = {}
    signatures: Dict[str, np.ndarray] = {}
    for metric in sorted(eligible):
        signature = []
        for dataset in sorted(datasets):
            rank_map = _rank_scores(values[metric].get(dataset, {}), higher_is_better=metric not in LOWER_IS_BETTER)
            signature.extend(rank_map.get(model, np.nan) for model in sorted(models))
        signature_array = np.asarray(signature, dtype=float)
        matched = None
        for kept_metric, kept_signature in signatures.items():
            valid = np.isfinite(signature_array) & np.isfinite(kept_signature)
            same_missingness = np.array_equal(np.isfinite(signature_array), np.isfinite(kept_signature))
            if same_missingness and valid.any() and np.array_equal(signature_array[valid], kept_signature[valid]):
                matched = kept_metric
                break
        if matched is None:
            deduplicated.append(metric)
            signatures[metric] = signature_array
        else:
            duplicate_metrics[metric] = matched

    return {
        "values": values,
        "names": names,
        "models": sorted(models),
        "datasets": sorted(datasets),
        "metrics": deduplicated,
        "duplicate_metrics": duplicate_metrics,
    }


def _metric_rank_vectors(
    prepared: Mapping[str, Any], sampled_datasets: Sequence[str]
) -> tuple[Dict[str, Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    values = prepared["values"]
    models = prepared["models"]
    metric_means: Dict[str, Dict[str, float]] = {}
    per_dataset: Dict[str, List[Dict[str, float]]] = {}
    for metric in prepared["metrics"]:
        rank_maps = []
        for dataset in sampled_datasets:
            raw = values.get(metric, {}).get(dataset, {})
            rank_maps.append(_rank_scores(raw, higher_is_better=metric not in LOWER_IS_BETTER))
        per_dataset[metric] = rank_maps
        means = {}
        for model in models:
            observed = [rank_map[model] for rank_map in rank_maps if model in rank_map]
            if observed:
                means[model] = float(np.mean(observed))
        metric_means[metric] = means
    return metric_means, per_dataset


def _review_metric_evidence(
    prepared: Mapping[str, Any], sampled_datasets: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    models = prepared["models"]
    metric_means, per_dataset = _metric_rank_vectors(prepared, sampled_datasets)
    model_consensus = {}
    for model in models:
        observed = [metric_means[m][model] for m in prepared["metrics"] if model in metric_means[m]]
        if observed:
            model_consensus[model] = float(np.mean(observed))

    evidence: Dict[str, Dict[str, float]] = {}
    for metric in prepared["metrics"]:
        mean_map = metric_means[metric]
        common = sorted(set(mean_map) & set(model_consensus))
        consensus = (_corr([mean_map[m] for m in common], [model_consensus[m] for m in common]) + 1.0) / 2.0
        separation = min(1.0, 2.0 * float(np.std(list(mean_map.values())))) if mean_map else 0.0

        rank_maps = per_dataset[metric]
        correlations = []
        for i in range(len(rank_maps)):
            for j in range(i + 1, len(rank_maps)):
                overlap = sorted(set(rank_maps[i]) & set(rank_maps[j]))
                if len(overlap) >= 3:
                    correlations.append(_corr([rank_maps[i][m] for m in overlap], [rank_maps[j][m] for m in overlap]))
        consistency = ((float(np.mean(correlations)) if correlations else 0.0) + 1.0) / 2.0

        observed = sum(len(rank_map) for rank_map in rank_maps)
        coverage = observed / max(1, len(sampled_datasets) * len(models))

        redundancy = []
        for other in prepared["metrics"]:
            if other == metric:
                continue
            overlap = sorted(set(mean_map) & set(metric_means[other]))
            if len(overlap) >= 3:
                redundancy.append(abs(_corr([mean_map[m] for m in overlap], [metric_means[other][m] for m in overlap])))
        uniqueness = max(0.10, 1.0 - (float(np.mean(redundancy)) if redundancy else 0.0))

        parts = [max(1e-6, x) for x in (coverage, separation, consistency, consensus, uniqueness)]
        committee_support = float(np.prod(parts) ** (1.0 / len(parts)))
        evidence[metric] = {
            "coverage": round(coverage, 6),
            "separation": round(separation, 6),
            "consistency": round(consistency, 6),
            "consensus": round(consensus, 6),
            "uniqueness": round(uniqueness, 6),
            "committee_support": round(committee_support, 6),
        }
    return evidence


def _score_models(
    prepared: Mapping[str, Any], sampled_datasets: Sequence[str], weights: Mapping[str, float]
) -> Dict[str, float]:
    metric_means, _ = _metric_rank_vectors(prepared, sampled_datasets)
    scores = {}
    for model in prepared["models"]:
        weighted_sum = 0.0
        used_weight = 0.0
        for metric, weight in weights.items():
            if model in metric_means.get(metric, {}):
                weighted_sum += float(weight) * metric_means[metric][model]
                used_weight += float(weight)
        if used_weight > 0:
            scores[model] = weighted_sum / used_weight
    return scores


def _initial_weights(prepared: Mapping[str, Any], initial: Mapping[str, float] | None) -> Dict[str, float]:
    metrics = prepared["metrics"]
    if not initial:
        return _normalize({}, metrics)
    aliases = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "aucpr": "auprc",
        "prauc": "auprc",
        "averageprecision": "auprc",
        "aucroc": "auroc",
        "rocauc": "auroc",
        "f1score": "f1",
    }
    by_canonical = {}
    for key, value in initial.items():
        compact = str(key).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        by_canonical[aliases.get(compact, compact)] = value
    mapped = {}
    for metric in metrics:
        compact = aliases.get(metric.replace("-", "").replace("_", ""), metric)
        if compact in by_canonical:
            mapped[metric] = float(by_canonical[compact])
    return _normalize(mapped, metrics) if mapped else _normalize({}, metrics)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_ranking(
    output_dir: Path, ranking: Sequence[Mapping[str, Any]], score_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, str]:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    ordered = [str(row["model"]) for row in ranking]
    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    for row in score_rows:
        scores_by_model[str(row["model"])].append(float(row["score"]))

    height = max(6.8, 0.34 * len(ordered) + 1.6)
    fig, (ax_box, ax_bubble) = plt.subplots(
        1, 2, figsize=(12.2, height), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True
    )
    palette = plt.get_cmap("tab20")
    colors = [palette(i % 20) for i in range(len(ordered))]
    positions = np.arange(1, len(ordered) + 1)
    boxes = ax_box.boxplot(
        [scores_by_model[m] for m in ordered],
        positions=positions,
        vert=False,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#6B7280", "linewidth": 0.8},
        capprops={"color": "#6B7280", "linewidth": 0.8},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#4B5563")
        patch.set_linewidth(0.7)
    ax_box.set_yticks(positions, [f"{i}. {model}" for i, model in enumerate(ordered, 1)])
    ax_box.invert_yaxis()
    ax_box.set_xlabel("Weighted rank score across 50 deliberation rounds")
    ax_box.set_title("A. Ranking uncertainty across weight updates", loc="left", fontweight="bold")
    ax_box.grid(axis="x", color="#E5E7EB", linewidth=0.7)

    x = np.array([float(row["median_score"]) for row in ranking])
    y = np.array([int(row["rank"]) for row in ranking])
    top3 = np.array([float(row["top3_frequency"]) for row in ranking])
    iqr = np.array([float(row["score_iqr"]) for row in ranking])
    sizes = 65.0 + 620.0 * top3
    scatter = ax_bubble.scatter(
        x, y, s=sizes, c=iqr, cmap="viridis_r", alpha=0.82, edgecolors="#374151", linewidths=0.65
    )
    for row in ranking:
        ax_bubble.annotate(
            str(row["model"]),
            (float(row["median_score"]), int(row["rank"])),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    ax_bubble.invert_yaxis()
    ax_bubble.set_yticks(positions)
    ax_bubble.set_xlabel("Median weighted rank score")
    ax_bubble.set_ylabel("Final rank")
    ax_bubble.set_title("B. Consensus rank and Top3 confidence", loc="left", fontweight="bold")
    ax_bubble.grid(color="#E5E7EB", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=ax_bubble, fraction=0.045, pad=0.03)
    cbar.set_label("Score IQR (lower is more stable)")
    for frequency in (0.25, 0.50, 0.75, 1.0):
        ax_bubble.scatter([], [], s=65.0 + 620.0 * frequency, c="#9CA3AF", alpha=0.65,
                          edgecolors="#374151", linewidths=0.65, label=f"Top3 {frequency:.0%}")
    ax_bubble.legend(title="Bubble size", loc="upper left", frameon=False, fontsize=7, title_fontsize=7)

    png_path = output_dir / "iterative_weight_ranking.png"
    svg_path = output_dir / "iterative_weight_ranking.svg"
    pdf_path = output_dir / "iterative_weight_ranking.pdf"
    tiff_path = output_dir / "iterative_weight_ranking.tiff"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    files = {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "tiff": str(tiff_path),
    }
    plt.close(fig)
    return files


def run_iterative_weight_meeting(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    rounds: int = 50,
    seed: int = 20260716,
    initial_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Run the local evidence committee and persist its complete audit trail."""
    if rounds < 2:
        raise ValueError("rounds must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_rows, resource_gate = apply_resource_gate(list(rows), load_model_resource_policy())
    resource_audit_files = write_resource_gate_audit(output_dir, resource_gate)
    prepared = _prepare(eligible_rows)
    if not prepared["metrics"] or not prepared["models"] or not prepared["datasets"]:
        raise ValueError("No eligible model/metric/dataset evidence for iterative weight meeting")

    rng = np.random.default_rng(seed)
    weights = _initial_weights(prepared, initial_weights)
    round_records = []
    weight_rows = []
    score_rows = []
    rank_rows = []

    for round_index in range(1, rounds + 1):
        sampled = rng.choice(prepared["datasets"], size=len(prepared["datasets"]), replace=True).tolist()
        evidence = _review_metric_evidence(prepared, sampled)
        target = _normalize({metric: evidence[metric]["committee_support"] for metric in prepared["metrics"]}, prepared["metrics"])
        learning_rate = max(0.12, 0.38 / math.sqrt(round_index))
        updated = _normalize(
            {metric: (1.0 - learning_rate) * weights[metric] + learning_rate * target[metric] for metric in prepared["metrics"]},
            prepared["metrics"],
        )
        scores = _score_models(prepared, sampled, updated)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
        rank_map = {model: rank for rank, (model, _) in enumerate(ranked, 1)}

        strongest = max(prepared["metrics"], key=lambda metric: evidence[metric]["committee_support"])
        largest_change = max(prepared["metrics"], key=lambda metric: abs(updated[metric] - weights[metric]))
        round_records.append({
            "round": round_index,
            "sampled_datasets": sampled,
            "learning_rate": round(learning_rate, 6),
            "weights_before": {prepared["names"][m]: round(weights[m], 8) for m in prepared["metrics"]},
            "metric_evidence": {prepared["names"][m]: evidence[m] for m in prepared["metrics"]},
            "weights_after": {prepared["names"][m]: round(updated[m], 8) for m in prepared["metrics"]},
            "discussion": {
                "statistics_expert": f"Bootstrap evidence most strongly supports {prepared['names'][strongest]} in this round.",
                "screening_expert": "Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.",
                "reviewer": f"Largest weight revision: {prepared['names'][largest_change]}; missingness and metric redundancy were penalized.",
            },
            "top3": [model for model, _ in ranked[:3]],
        })
        for metric in prepared["metrics"]:
            weight_rows.append({
                "round": round_index,
                "metric": prepared["names"][metric],
                "weight": round(updated[metric], 10),
                **evidence[metric],
            })
        for model, score in ranked:
            score_rows.append({"round": round_index, "model": model, "score": round(score, 10), "rank": rank_map[model]})
            rank_rows.append((model, rank_map[model]))
        weights = updated

    final_weights = {}
    for metric in prepared["metrics"]:
        metric_values = [row["weight"] for row in weight_rows if row["metric"] == prepared["names"][metric]]
        final_weights[prepared["names"][metric]] = float(np.median(metric_values))
    final_weights = _normalize(final_weights, list(final_weights))

    top3_counts = Counter(model for record in round_records for model in record["top3"])
    ranks_by_model: Dict[str, List[int]] = defaultdict(list)
    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    for row in score_rows:
        ranks_by_model[row["model"]].append(int(row["rank"]))
        scores_by_model[row["model"]].append(float(row["score"]))

    ranking = []
    for model in prepared["models"]:
        values = np.asarray(scores_by_model[model], dtype=float)
        ranks = np.asarray(ranks_by_model[model], dtype=float)
        ranking.append({
            "model": model,
            "median_score": round(float(np.median(values)), 8),
            "mean_score": round(float(np.mean(values)), 8),
            "score_iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 8),
            "median_rank": round(float(np.median(ranks)), 4),
            "mean_rank": round(float(np.mean(ranks)), 4),
            "top3_frequency": round(top3_counts[model] / rounds, 6),
            "rounds": rounds,
        })
    ranking.sort(key=lambda row: (-row["median_score"], row["mean_rank"], row["score_iqr"], row["model"].lower()))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank

    _write_csv(
        output_dir / "metric_weight_50_rounds.csv",
        weight_rows,
        ["round", "metric", "weight", "coverage", "separation", "consistency", "consensus", "uniqueness", "committee_support"],
    )
    _write_csv(output_dir / "model_scores_50_rounds.csv", score_rows, ["round", "model", "score", "rank"])
    ranking_fields = ["rank", "model", "median_score", "mean_score", "score_iqr", "median_rank", "mean_rank", "top3_frequency", "rounds"]
    _write_csv(output_dir / "model_ranking_50_rounds.csv", ranking, ranking_fields)
    figure_files = _plot_ranking(output_dir, ranking, score_rows)

    result = {
        "method": "local evidence-driven multi-role deliberation with dataset bootstrap",
        "rounds": rounds,
        "seed": seed,
        "models": prepared["models"],
        "datasets": prepared["datasets"],
        "eligible_metrics": [prepared["names"][metric] for metric in prepared["metrics"]],
        "excluded_fields": sorted(NON_RANKING_FIELDS),
        "duplicate_metrics_removed": {
            prepared["names"][metric]: prepared["names"][kept]
            for metric, kept in prepared.get("duplicate_metrics", {}).items()
        },
        "resource_gate": resource_gate,
        "resource_gate_audit_files": resource_audit_files,
        "initial_weights": {prepared["names"][m]: round(_initial_weights(prepared, initial_weights)[m], 8) for m in prepared["metrics"]},
        "final_weights_median": {key: round(value, 8) for key, value in final_weights.items()},
        "final_ranking": ranking,
        "figure_files": figure_files,
        "round_records": round_records,
    }
    (output_dir / "weight_meeting_50_rounds.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trace = [
        "# 50-round metric-weight meeting trace",
        "",
        "Each round bootstraps datasets and records three local expert roles: statistics, screening, and review.",
        "No model-specific priority bonus is used.",
        "Model eligibility is decided before scoring by a model-agnostic measured-resource budget gate.",
        f"Resource-excluded models: {', '.join(resource_gate['excluded_models']) or 'none'}.",
        f"Eligible models lacking resource measurements: {', '.join(resource_gate['flagged_models']) or 'none'}.",
        "",
    ]
    for record in round_records:
        trace.extend([
            f"## Round {record['round']}",
            "",
            f"- Sampled datasets: {', '.join(record['sampled_datasets'])}",
            f"- Statistics expert: {record['discussion']['statistics_expert']}",
            f"- Screening expert: {record['discussion']['screening_expert']}",
            f"- Reviewer: {record['discussion']['reviewer']}",
            f"- Top3: {', '.join(record['top3'])}",
            "",
        ])
    (output_dir / "weight_meeting_50_rounds.md").write_text("\n".join(trace), encoding="utf-8")
    return result
