# -*- coding: utf-8 -*-
"""Select a three-model AMP ensemble from stored Stage 2 predictions."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METRICS = [
    "ACC", "AUPRC", "AUROC", "BalancedAccuracy", "BrierScore", "ECE",
    "F1-Score", "MCC", "NPV", "Precision", "Recall", "Specificity",
]
LOWER_IS_BETTER = {"BrierScore", "ECE"}


def _metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    predicted = probability >= 0.5
    positive = y == 1
    negative = ~positive
    tp = float(np.sum(predicted & positive))
    tn = float(np.sum(~predicted & negative))
    fp = float(np.sum(predicted & negative))
    fn = float(np.sum(~predicted & positive))
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1] if index < 9 else probability <= edges[index + 1]
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask])) - float(np.mean(y[mask]))
            )
    return {
        "ACC": (tp + tn) / n,
        "AUPRC": float(average_precision_score(y, probability)),
        "AUROC": float(roc_auc_score(y, probability)),
        "BalancedAccuracy": (recall + specificity) / 2.0,
        "BrierScore": float(np.mean((probability - y) ** 2)),
        "ECE": ece,
        "F1-Score": f1,
        "MCC": mcc,
        "NPV": npv,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
    }


def _load_predictions(results_dir: Path) -> tuple[Dict[str, Any], list[str]]:
    datasets: Dict[str, Any] = {}
    model_sets = []
    for path in sorted(results_dir.glob("*/final_results_with_predictions.csv")):
        frame = pd.read_csv(path)
        if "True_Label" not in frame:
            continue
        models = [column[:-5] for column in frame.columns if column.endswith("_Prob")]
        if not models:
            continue
        model_sets.append(set(models))
        datasets[path.parent.name] = {
            "y": frame["True_Label"].to_numpy(dtype=np.int8),
            "probabilities": {
                model: frame[f"{model}_Prob"].to_numpy(dtype=np.float64) for model in models
            },
        }
    if len(datasets) < 2:
        raise ValueError("At least two Stage 2 prediction tables are required")
    models = sorted(set.intersection(*model_sets), key=str.lower)
    if len(models) < 3:
        raise ValueError("At least three models with complete cross-dataset predictions are required")
    for dataset, data in datasets.items():
        for model in models:
            if not np.all(np.isfinite(data["probabilities"][model])):
                raise ValueError(f"{dataset}/{model} contains missing or non-finite probabilities")
    return datasets, models


def _evaluate_combo(combo: Sequence[str], datasets: Mapping[str, Any]) -> list[Dict[str, Any]]:
    combo_name = " + ".join(combo)
    rows = []
    for dataset, data in datasets.items():
        probability = np.mean(
            np.column_stack([data["probabilities"][model] for model in combo]), axis=1
        )
        rows.append({"combo": combo_name, "dataset": dataset, **_metrics(data["y"], probability)})
    return rows


def _metric_percentile_table(raw: pd.DataFrame) -> pd.DataFrame:
    ranked = raw[["combo", "dataset"]].copy()
    for metric in METRICS:
        # pandas gives the largest percentile to the best value with these directions.
        ascending = metric not in LOWER_IS_BETTER
        ranked[metric] = raw.groupby("dataset")[metric].rank(
            method="average", ascending=ascending, pct=True
        )
    return ranked


def select_top3(
    results_dir: Path,
    output_dir: Path,
    workers: int = 8,
) -> Dict[str, Any]:
    meeting_path = results_dir / "weight_meeting_50_rounds.json"
    meeting = json.loads(meeting_path.read_text(encoding="utf-8"))
    round_records = meeting.get("round_records") or []
    if not round_records:
        raise ValueError(f"No dynamic Agent weight rounds found in {meeting_path}")

    datasets, models = _load_predictions(results_dir)
    combinations = list(itertools.combinations(models, 3))
    raw_rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for rows in executor.map(
            lambda combo: _evaluate_combo(combo, datasets), combinations, chunksize=4
        ):
            raw_rows.extend(rows)
    raw = pd.DataFrame(raw_rows)
    ranked = _metric_percentile_table(raw)

    score_history: Dict[str, list[float]] = defaultdict(list)
    rank_history: Dict[str, list[int]] = defaultdict(list)
    top3_counts: Counter[str] = Counter()
    for record in round_records:
        weights = record["weights_after"]
        per_dataset = ranked[["combo", "dataset"]].copy()
        per_dataset["utility"] = sum(float(weights[m]) * ranked[m] for m in METRICS)
        sampled = [
            per_dataset[per_dataset["dataset"] == dataset][["combo", "utility"]]
            for dataset in record["sampled_datasets"]
        ]
        scores = (
            pd.concat(sampled)
            .groupby("combo", sort=False)["utility"]
            .mean()
            .sort_values(ascending=False)
        )
        for rank, (combo, score) in enumerate(scores.items(), 1):
            score_history[combo].append(float(score))
            rank_history[combo].append(rank)
        top3_counts.update(scores.index[:3])

    summary = []
    for combo in sorted(score_history):
        scores = np.asarray(score_history[combo], dtype=float)
        ranks = np.asarray(rank_history[combo], dtype=float)
        combo_raw = raw[raw["combo"] == combo]
        summary.append({
            "combo": combo,
            "median_score": float(np.median(scores)),
            "mean_score": float(np.mean(scores)),
            "score_iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "top3_frequency": top3_counts[combo] / len(round_records),
            "mean_AUPRC": float(combo_raw["AUPRC"].mean()),
            "mean_MCC": float(combo_raw["MCC"].mean()),
            "mean_Precision": float(combo_raw["Precision"].mean()),
            "mean_Recall": float(combo_raw["Recall"].mean()),
            "worst_dataset_AUPRC": float(combo_raw["AUPRC"].min()),
            "worst_dataset_MCC": float(combo_raw["MCC"].min()),
        })
    summary.sort(
        key=lambda row: (
            -row["median_score"], row["mean_rank"], row["score_iqr"], row["combo"].lower()
        )
    )
    for rank, row in enumerate(summary, 1):
        row["rank"] = rank

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "combo", "median_score", "mean_score", "score_iqr", "median_rank",
        "mean_rank", "top3_frequency", "mean_AUPRC", "mean_MCC", "mean_Precision",
        "mean_Recall", "worst_dataset_AUPRC", "worst_dataset_MCC",
    ]
    csv_path = output_dir / "ensemble_top3_combination_ranking.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    result = {
        "method": (
            "Exhaustive equal-probability soft voting over every three-model combination; "
            "per-dataset metric percentile normalization; existing 50 dynamic Agent weight "
            "rounds and dataset bootstraps reused without model-specific bonuses."
        ),
        "datasets": list(datasets),
        "models": models,
        "combinations": len(combinations),
        "recommended_models": summary[0]["combo"].split(" + "),
        "ranking": summary[:20],
        "scientific_caveat": (
            "Exploratory selection on the same benchmark datasets. Confirm the chosen trio on "
            "an untouched external validation set before a formal performance claim."
        ),
        "outputs": {"ranking_csv": str(csv_path)},
    }
    json_path = output_dir / "ensemble_top3_selection.json"
    result["outputs"]["audit_json"] = str(json_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/results_manual"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = select_top3(
        args.results_dir.resolve(), (args.output_dir or args.results_dir).resolve(), args.workers
    )
    print(json.dumps({
        "recommended_models": result["recommended_models"],
        "top_combinations": result["ranking"][:5],
        "outputs": result["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
