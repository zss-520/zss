"""Deterministic scientific evaluation protocol for prediction tables."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

PROTOCOL_VERSION = "2.0"
DEFAULT_THRESHOLD = 0.5
CALIBRATION_BINS = 10
RANKING_FRACTIONS = (0.01, 0.05, 0.10)


def _normalized_model_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def protocol_config() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "primary_endpoint": "AUPRC",
        "secondary_endpoint": "MCC",
        "composite_score_role": "exploratory ranking only; not a substitute for endpoint-wise reporting",
        "threshold_policy": (
            "maximize MCC on an independent validation prediction table; test-set tuning is forbidden. "
            "Formal runs require validation-derived thresholds; fixed 0.5 is diagnostic only."
        ),
        "require_validation_threshold": os.getenv("SCIENTIFIC_REQUIRE_VALIDATION_THRESHOLD", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        "bootstrap_iterations": int(os.getenv("SCIENTIFIC_BOOTSTRAP_ITERATIONS", "500")),
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": int(os.getenv("SCIENTIFIC_BOOTSTRAP_SEED", "20260714")),
        "bootstrap_unit": "homology/sequence cluster when a cluster column is present; otherwise sequence row",
        "pairwise_tests": [
            "paired bootstrap effect differences on common valid rows/clusters",
            "McNemar with continuity correction for thresholded errors",
        ],
        "multiple_comparison_correction": "Holm family-wise correction across pairwise McNemar tests",
        "calibration": f"Brier score plus {CALIBRATION_BINS}-bin equal-width ECE/MCE",
        "application_utility": "AUPRC lift and precision/recall/enrichment at top 1%, 5% and 10%",
        "resource_reporting": "job-level SLURM elapsed time, MaxRSS and sequence throughput when supplied",
    }


def _finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = CALIBRATION_BINS) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:-1], right=False), bins - 1)
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index in range(bins):
        mask = bin_ids == index
        count = int(np.sum(mask))
        if not count:
            rows.append(
                {
                    "bin": index + 1,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": 0,
                    "mean_probability": None,
                    "observed_positive_fraction": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_probability = float(np.mean(y_prob[mask]))
        observed = float(np.mean(y_true[mask]))
        gap = abs(mean_probability - observed)
        weighted_gap += count * gap
        max_gap = max(max_gap, gap)
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": _finite_or_none(mean_probability),
                "observed_positive_fraction": _finite_or_none(observed),
                "absolute_gap": _finite_or_none(gap),
            }
        )
    return {
        "brier_score": _finite_or_none(np.mean((y_prob - y_true) ** 2)),
        "ece": _finite_or_none(weighted_gap / len(y_true)) if len(y_true) else None,
        "mce": _finite_or_none(max_gap) if len(y_true) else None,
        "binning": "equal_width",
        "bins": rows,
    }


def _ranking_utility(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    n = len(y_true)
    positive_n = int(np.sum(y_true == 1))
    prevalence = positive_n / n if n else float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    rows: dict[str, Any] = {}
    for fraction in RANKING_FRACTIONS:
        k = min(n, max(1, int(math.ceil(n * fraction)))) if n else 0
        hits = int(np.sum(y_true[order[:k]] == 1)) if k else 0
        precision = hits / k if k else float("nan")
        recall = hits / positive_n if positive_n else float("nan")
        enrichment = precision / prevalence if prevalence > 0 else float("nan")
        rows[f"top_{int(fraction * 100)}pct"] = {
            "k": k,
            "true_positives": hits,
            "precision": _finite_or_none(precision),
            "recall": _finite_or_none(recall),
            "enrichment_factor": _finite_or_none(enrichment),
            "number_needed_to_test": _finite_or_none(1.0 / precision) if precision > 0 else None,
        }
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "prevalence": _finite_or_none(prevalence),
        "random_auprc_baseline": _finite_or_none(prevalence),
        "auprc_lift": _finite_or_none(auprc / prevalence) if prevalence > 0 else None,
        "top_fraction_metrics": rows,
    }


def _metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    balanced_accuracy = (recall + specificity) / 2.0 if math.isfinite(specificity) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else 0.0
    two_classes = len(np.unique(y_true)) == 2
    calibration = _calibration_summary(y_true, y_prob)
    ranking = _ranking_utility(y_true, y_prob)
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true == 1)),
        "negative_n": int(np.sum(y_true == 0)),
        "accuracy": _finite_or_none(accuracy_score(y_true, y_pred)),
        "precision": _finite_or_none(precision),
        "recall": _finite_or_none(recall),
        "specificity": _finite_or_none(specificity),
        "balanced_accuracy": _finite_or_none(balanced_accuracy),
        "negative_predictive_value": _finite_or_none(npv),
        "false_positive_rate": _finite_or_none(fpr),
        "false_negative_rate": _finite_or_none(fnr),
        "f1": _finite_or_none(f1),
        "mcc": _finite_or_none(mcc),
        "auroc": _finite_or_none(roc_auc_score(y_true, y_prob)) if two_classes else None,
        "auprc": _finite_or_none(average_precision_score(y_true, y_prob)) if two_classes else None,
        "brier_score": calibration["brier_score"],
        "expected_calibration_error": calibration["ece"],
        "maximum_calibration_error": calibration["mce"],
        "calibration_curve": calibration["bins"],
        "ranking_utility": ranking,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _choose_validation_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    candidates = sorted(set(float(x) for x in y_prob if math.isfinite(float(x))))
    candidates = sorted(set([DEFAULT_THRESHOLD, *candidates]))
    best_threshold = DEFAULT_THRESHOLD
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    for threshold in candidates:
        metrics = _metric_bundle(y_true, y_prob, threshold)
        mcc = metrics["mcc"] if metrics["mcc"] is not None else -1.0
        recall = metrics["recall"] if metrics["recall"] is not None else -1.0
        key = (float(mcc), float(recall), -abs(threshold - DEFAULT_THRESHOLD))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold), float(best_key[0])


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    iterations: int,
    confidence: float,
    seed: int,
    cluster_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    values: dict[str, list[float]] = {k: [] for k in [
        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "mcc", "auroc", "auprc", "brier_score", "expected_calibration_error",
    ]}
    n = len(y_true)
    for _ in range(max(0, iterations)):
        idx = _bootstrap_indices(rng, n, cluster_ids)
        metrics = _metric_bundle(y_true[idx], y_prob[idx], threshold)
        for key in values:
            value = metrics.get(key)
            if value is not None:
                values[key].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {}
    for key, samples in values.items():
        if not samples:
            result[key] = {"lower": None, "upper": None, "median": None, "valid_samples": 0}
            continue
        result[key] = {
            "lower": _finite_or_none(np.quantile(samples, alpha)),
            "upper": _finite_or_none(np.quantile(samples, 1.0 - alpha)),
            "median": _finite_or_none(np.median(samples)),
            "valid_samples": len(samples),
        }
    return result


def _bootstrap_indices(rng: np.random.RandomState, n: int, cluster_ids: np.ndarray | None) -> np.ndarray:
    if cluster_ids is None:
        return rng.randint(0, n, n)
    clusters = np.asarray(cluster_ids, dtype=object)
    unique = list(dict.fromkeys(clusters.tolist()))
    sampled = rng.randint(0, len(unique), len(unique))
    pieces = [np.flatnonzero(clusters == unique[index]) for index in sampled]
    return np.concatenate(pieces) if pieces else np.arange(n)


def _holm_adjust(rows: list[dict[str, Any]], p_key: str = "p_value") -> None:
    valid = [(index, float(row[p_key])) for index, row in enumerate(rows) if row.get(p_key) is not None]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    adjusted: dict[int, float] = {}
    m = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[index] = running
    for index, row in enumerate(rows):
        value = adjusted.get(index)
        row["p_value_holm"] = _finite_or_none(value) if value is not None else None
        row["reject_holm_0_05"] = bool(value is not None and value < 0.05)


def _load_prediction_table(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    # Passing a handle avoids old pandas/Windows failures on non-ASCII paths.
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        df = pd.read_csv(source)
    label_col = next((c for c in ["True_Label", "label", "target", "class"] if c in df.columns), "")
    if not label_col:
        raise ValueError(f"{path} 缺少真值列 True_Label/label/target/class")
    prob_cols = [c for c in df.columns if c.endswith("_Prob")]
    if not prob_cols:
        raise ValueError(f"{path} 没有 *_Prob 模型概率列")
    return df, label_col, prob_cols


def _paired_mcnemar(
    df: pd.DataFrame,
    label_col: str,
    model_cols: list[str],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, col_a in enumerate(model_cols):
        for col_b in model_cols[i + 1 :]:
            labels = pd.to_numeric(df[label_col], errors="coerce")
            probs_a = pd.to_numeric(df[col_a], errors="coerce")
            probs_b = pd.to_numeric(df[col_b], errors="coerce")
            valid = (
                labels.isin([0, 1])
                & probs_a.notna()
                & probs_b.notna()
                & np.isfinite(probs_a)
                & np.isfinite(probs_b)
                & probs_a.between(0.0, 1.0)
                & probs_b.between(0.0, 1.0)
            )
            if not valid.any():
                continue
            y = labels.loc[valid].values.astype(int)
            pa = probs_a.loc[valid].values.astype(float)
            pb = probs_b.loc[valid].values.astype(float)
            name_a, name_b = col_a[:-5], col_b[:-5]
            correct_a = (pa >= thresholds[name_a]).astype(int) == y
            correct_b = (pb >= thresholds[name_b]).astype(int) == y
            b = int(np.sum(correct_a & ~correct_b))
            c = int(np.sum(~correct_a & correct_b))
            if b + c == 0:
                statistic, p_value = 0.0, 1.0
            else:
                statistic = max(abs(b - c) - 1.0, 0.0) ** 2 / (b + c)
                p_value = math.erfc(math.sqrt(statistic / 2.0))
            rows.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "n_common": int(np.sum(valid)),
                    "a_correct_b_wrong": b,
                    "a_wrong_b_correct": c,
                    "chi_square_cc": _finite_or_none(statistic),
                    "p_value": _finite_or_none(p_value),
                }
            )
    _holm_adjust(rows)
    return rows


def _pairwise_bootstrap_differences(
    df: pd.DataFrame,
    label_col: str,
    model_cols: list[str],
    thresholds: dict[str, float],
    *,
    iterations: int,
    confidence: float,
    seed: int,
    cluster_col: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - confidence) / 2.0
    metric_names = ["auprc", "mcc", "balanced_accuracy", "brier_score"]
    for pair_index, col_a in enumerate(model_cols):
        for col_b in model_cols[pair_index + 1 :]:
            labels = pd.to_numeric(df[label_col], errors="coerce")
            probs_a = pd.to_numeric(df[col_a], errors="coerce")
            probs_b = pd.to_numeric(df[col_b], errors="coerce")
            valid = (
                labels.isin([0, 1])
                & probs_a.notna()
                & probs_b.notna()
                & np.isfinite(probs_a)
                & np.isfinite(probs_b)
                & probs_a.between(0.0, 1.0)
                & probs_b.between(0.0, 1.0)
            )
            if not valid.any():
                continue
            y = labels.loc[valid].values.astype(int)
            pa = probs_a.loc[valid].values.astype(float)
            pb = probs_b.loc[valid].values.astype(float)
            name_a, name_b = col_a[:-5], col_b[:-5]
            clusters = None
            if cluster_col:
                raw_clusters = df.loc[valid, cluster_col]
                clusters = np.asarray(
                    [str(value) if pd.notna(value) and str(value).strip() else f"__row_{i}" for i, value in enumerate(raw_clusters)],
                    dtype=object,
                )
            observed_a = _metric_bundle(y, pa, thresholds[name_a])
            observed_b = _metric_bundle(y, pb, thresholds[name_b])
            samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
            rng = np.random.RandomState(seed + len(rows))
            for _ in range(max(0, iterations)):
                idx = _bootstrap_indices(rng, len(y), clusters)
                metrics_a = _metric_bundle(y[idx], pa[idx], thresholds[name_a])
                metrics_b = _metric_bundle(y[idx], pb[idx], thresholds[name_b])
                for metric in metric_names:
                    value_a, value_b = metrics_a.get(metric), metrics_b.get(metric)
                    if value_a is not None and value_b is not None:
                        samples[metric].append(float(value_a) - float(value_b))
            differences: dict[str, Any] = {}
            for metric in metric_names:
                observed = None
                if observed_a.get(metric) is not None and observed_b.get(metric) is not None:
                    observed = float(observed_a[metric]) - float(observed_b[metric])
                values = samples[metric]
                differences[metric] = {
                    "difference_a_minus_b": _finite_or_none(observed),
                    "lower": _finite_or_none(np.quantile(values, alpha)) if values else None,
                    "upper": _finite_or_none(np.quantile(values, 1.0 - alpha)) if values else None,
                    "probability_a_greater": _finite_or_none(np.mean(np.asarray(values) > 0)) if values else None,
                    "valid_samples": len(values),
                    "direction_note": "lower_is_better" if metric == "brier_score" else "higher_is_better",
                }
            rows.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "n_common": int(np.sum(valid)),
                    "resampling_unit": cluster_col or "sequence_row",
                    "differences": differences,
                }
            )
    return rows


def _parse_elapsed_seconds(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        try:
            days = int(day_text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
        elif len(parts) == 2:
            hours, minutes, seconds = 0.0, *map(float, parts)
        else:
            return float(text)
    except ValueError:
        return None
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def _execution_metrics(execution_metadata: dict[str, Any] | None, n_rows: int) -> dict[str, Any] | None:
    if not execution_metadata:
        return None
    slurm = execution_metadata.get("slurm") if isinstance(execution_metadata.get("slurm"), dict) else {}
    elapsed = _parse_elapsed_seconds(slurm.get("elapsed") or execution_metadata.get("elapsed"))
    return {
        "scope": "whole benchmark job; do not interpret as per-model latency when models ran together",
        "status": execution_metadata.get("status"),
        "job_id": execution_metadata.get("job_id") or slurm.get("job_id"),
        "slurm_state": slurm.get("state"),
        "exit_code": slurm.get("exit_code") or execution_metadata.get("exit_code"),
        "elapsed": slurm.get("elapsed") or execution_metadata.get("elapsed"),
        "elapsed_seconds": _finite_or_none(elapsed),
        "max_rss": slurm.get("max_rss") or execution_metadata.get("max_rss"),
        "sequence_rows_per_second": _finite_or_none(n_rows / elapsed) if elapsed and elapsed > 0 else None,
    }


def _length_subgroups(
    df: pd.DataFrame,
    valid: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any] | None:
    sequence_col = next((name for name in ["Sequence", "sequence", "Peptide", "Seq"] if name in df.columns), None)
    if not sequence_col:
        return None
    lengths = df.loc[valid, sequence_col].astype(str).str.replace(r"\s+", "", regex=True).str.len().values
    groups = {
        "10_20_aa": (10, 20),
        "21_30_aa": (21, 30),
        "31_50_aa": (31, 50),
        "51_100_aa": (51, 100),
    }
    output: dict[str, Any] = {}
    for name, (lower, upper) in groups.items():
        mask = (lengths >= lower) & (lengths <= upper)
        if np.any(mask):
            output[name] = _metric_bundle(y_true[mask], y_prob[mask], threshold)
    return output


def evaluate_prediction_table(
    predictions_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_csv: str | Path | None = None,
    expected_models: Iterable[str] | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    require_validation_threshold: bool | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol_config()
    if iterations is not None:
        config["bootstrap_iterations"] = int(iterations)
    if seed is not None:
        config["bootstrap_seed"] = int(seed)
    if require_validation_threshold is not None:
        config["require_validation_threshold"] = bool(require_validation_threshold)

    df, label_col, prob_cols = _load_prediction_table(predictions_csv)
    all_prob_cols = list(prob_cols)
    expected = [str(name) for name in (expected_models or []) if str(name).strip()]
    if expected:
        by_normalized = {
            _normalized_model_name(column[:-5]): column for column in prob_cols
        }
        missing = [
            name for name in expected if _normalized_model_name(name) not in by_normalized
        ]
        if missing:
            raise ValueError(
                "预测文件缺少本轮目标模型概率列: " + ", ".join(missing)
            )
        prob_cols = [by_normalized[_normalized_model_name(name)] for name in expected]
    validation_path = Path(validation_csv) if validation_csv else output_dir / "validation_results_with_predictions.csv"
    validation = None
    validation_label_col = ""
    validation_prob_cols: list[str] = []
    if validation_path.exists():
        validation, validation_label_col, validation_prob_cols = _load_prediction_table(validation_path)

    labels_numeric = pd.to_numeric(df[label_col], errors="coerce")
    cluster_col = next(
        (name for name in ["Homology_Cluster", "Sequence_Cluster", "Cluster_ID", "Family_ID"] if name in df.columns),
        None,
    )
    valid_labels = labels_numeric[labels_numeric.isin([0, 1])]
    report: dict[str, Any] = {
        "protocol": config,
        "predictions_csv": str(predictions_csv),
        "validation_csv": str(validation_path) if validation is not None else None,
        "data_quality": {
            "rows": len(df),
            "invalid_label_rows": int(labels_numeric.isna().sum()),
            "duplicate_id_rows": int(df.duplicated(subset=["Standard_ID"]).sum()) if "Standard_ID" in df.columns else None,
            "positive_rows": int(np.sum(valid_labels == 1)),
            "negative_rows": int(np.sum(valid_labels == 0)),
            "positive_prevalence": _finite_or_none(np.mean(valid_labels == 1)) if len(valid_labels) else None,
            "bootstrap_cluster_column": cluster_col,
            "expected_models": expected or None,
            "ignored_extra_probability_columns": [
                column for column in all_prob_cols if column not in prob_cols
            ],
        },
        "models": {},
        "pairwise_mcnemar": [],
        "pairwise_bootstrap_differences": [],
        "execution_metrics": _execution_metrics(execution_metadata, len(df)),
    }
    thresholds: dict[str, float] = {}
    standardized: dict[str, dict[str, Any]] = {}

    for index, prob_col in enumerate(prob_cols):
        model_name = prob_col[:-5]
        numeric_probs = pd.to_numeric(df[prob_col], errors="coerce")
        valid = (
            labels_numeric.isin([0, 1])
            & numeric_probs.notna()
            & np.isfinite(numeric_probs)
            & numeric_probs.between(0.0, 1.0)
        )
        y_true = labels_numeric.loc[valid].values.astype(int)
        y_prob = numeric_probs.loc[valid].values.astype(float)
        if not len(y_true):
            report["models"][model_name] = {"status": "no_valid_predictions", "coverage": 0.0}
            standardized[model_name] = {}
            thresholds[model_name] = DEFAULT_THRESHOLD
            continue

        threshold = DEFAULT_THRESHOLD
        threshold_source = "diagnostic_fixed_0.5_no_validation_data"
        validation_mcc = None
        if validation is not None and prob_col in validation_prob_cols:
            val_labels = pd.to_numeric(validation[validation_label_col], errors="coerce")
            val_probs = pd.to_numeric(validation[prob_col], errors="coerce")
            val_valid = (
                val_labels.isin([0, 1])
                & val_probs.notna()
                & np.isfinite(val_probs)
                & val_probs.between(0.0, 1.0)
            )
            if val_valid.any() and len(np.unique(val_labels.loc[val_valid])) == 2:
                threshold, validation_mcc = _choose_validation_threshold(
                    val_labels.loc[val_valid].values.astype(int),
                    val_probs.loc[val_valid].values.astype(float),
                )
                threshold_source = "validation_max_mcc"
        if config["require_validation_threshold"] and threshold_source != "validation_max_mcc":
            raise ValueError(
                f"Formal scientific evaluation requires an independent validation-derived threshold for {model_name}; "
                "fixed 0.5 is diagnostic only."
            )
        thresholds[model_name] = threshold
        selected_metrics = _metric_bundle(y_true, y_prob, threshold)
        fixed_metrics = _metric_bundle(y_true, y_prob, DEFAULT_THRESHOLD)
        cluster_ids = None
        if cluster_col:
            raw_clusters = df.loc[valid, cluster_col]
            cluster_ids = np.asarray(
                [str(value) if pd.notna(value) and str(value).strip() else f"__row_{i}" for i, value in enumerate(raw_clusters)],
                dtype=object,
            )
        ci = _bootstrap_ci(
            y_true,
            y_prob,
            threshold,
            iterations=int(config["bootstrap_iterations"]),
            confidence=float(config["bootstrap_confidence"]),
            seed=int(config["bootstrap_seed"]) + index,
            cluster_ids=cluster_ids,
        )
        report["models"][model_name] = {
            "status": "evaluated",
            "coverage": _finite_or_none(len(y_true) / len(df)),
            "invalid_prediction_rows": int((~valid).sum()),
            "out_of_range_probability_rows": int(
                (numeric_probs.notna() & ~numeric_probs.between(0.0, 1.0)).sum()
            ),
            "threshold": threshold,
            "threshold_source": threshold_source,
            "validation_mcc_at_selected_threshold": validation_mcc,
            "selected_threshold_metrics": selected_metrics,
            "fixed_0_5_metrics": fixed_metrics,
            "bootstrap_95_ci": ci,
            "bootstrap_resampling_unit": cluster_col or "sequence_row",
            "length_subgroup_metrics": _length_subgroups(df, valid, y_true, y_prob, threshold),
        }
        standardized[model_name] = {
            "ACC": selected_metrics["accuracy"],
            "Precision": selected_metrics["precision"],
            "Recall": selected_metrics["recall"],
            "Specificity": selected_metrics["specificity"],
            "BalancedAccuracy": selected_metrics["balanced_accuracy"],
            "NPV": selected_metrics["negative_predictive_value"],
            "F1-Score": selected_metrics["f1"],
            "MCC": selected_metrics["mcc"],
            "AUROC": selected_metrics["auroc"],
            "AUPRC": selected_metrics["auprc"],
            "BrierScore": selected_metrics["brier_score"],
            "ECE": selected_metrics["expected_calibration_error"],
            "AUPRC-Lift": (selected_metrics.get("ranking_utility") or {}).get("auprc_lift"),
            "Threshold": threshold,
            "Coverage": _finite_or_none(len(y_true) / len(df)),
        }

    report["pairwise_mcnemar"] = _paired_mcnemar(df, label_col, prob_cols, thresholds)
    report["pairwise_bootstrap_differences"] = _pairwise_bootstrap_differences(
        df,
        label_col,
        prob_cols,
        thresholds,
        iterations=int(config["bootstrap_iterations"]),
        confidence=float(config["bootstrap_confidence"]),
        seed=int(config["bootstrap_seed"]),
        cluster_col=cluster_col,
    )
    (output_dir / "scientific_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "eval_result.json").write_text(
        json.dumps(standardized, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(report, output_dir / "scientific_evaluation.md")
    return {"report": report, "eval_result": standardized}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Scientific Evaluation Protocol",
        "",
        f"- Protocol version: {report['protocol']['version']}",
        f"- Threshold policy: {report['protocol']['threshold_policy']}",
        f"- Bootstrap iterations: {report['protocol']['bootstrap_iterations']}",
        f"- Bootstrap unit: {report['data_quality'].get('bootstrap_cluster_column') or 'sequence row'}",
        f"- Validation predictions: {report.get('validation_csv') or 'not provided'}",
        f"- Multiple comparisons: {report['protocol']['multiple_comparison_correction']}",
        "",
        "| Model | Coverage | Threshold | Source | AUPRC | AUPRC lift | AUROC | MCC | Balanced ACC | Recall | Precision | Brier | ECE |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, item in report["models"].items():
        if item.get("status") != "evaluated":
            lines.append(f"| {model} | {item.get('coverage', 0)} |  | {item.get('status')} |  |  |  |  |  |  |  |  |  |")
            continue
        m = item["selected_threshold_metrics"]
        lines.append(
            "| " + " | ".join(
                [
                    model,
                    f"{item['coverage']:.4f}",
                    f"{item['threshold']:.6g}",
                    item["threshold_source"],
                    str(m.get("auprc")),
                    str((m.get("ranking_utility") or {}).get("auprc_lift")),
                    str(m.get("auroc")),
                    str(m.get("mcc")),
                    str(m.get("balanced_accuracy")),
                    str(m.get("recall")),
                    str(m.get("precision")),
                    str(m.get("brier_score")),
                    str(m.get("expected_calibration_error")),
                ]
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
