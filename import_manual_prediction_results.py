# -*- coding: utf-8 -*-
"""Import manually generated AMP prediction CSVs into the benchmark result flow.

The input CSVs are expected to contain:
  - Sequence
  - label
  - one or more model probability columns in [0, 1]

For each dataset this script writes the same core artifacts used by the normal
pipeline:
  - final_results_with_predictions.csv
  - eval_result.json
  - scientific_evaluation.json / .md
  - evaluation_curves.png
  - critic_individual.md (deterministic short summary)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from scientific_evaluation import DEFAULT_THRESHOLD, evaluate_prediction_table

DEFAULT_INPUTS = {
    "C_AMPs-predict_test": Path(r"D:/google/Downloads/C_AMPs-predict_test_out0309_corrected.csv"),
    "Veltri_test": Path(r"D:/google/Downloads/Veltri_test_out0309_corrected.csv"),
    "ProteoGPT_all_predictions": Path(r"D:/google/Downloads/ProteoGPT_all_predictions-0324_corrected.csv"),
}

DEFAULT_EXCLUDED_MODELS = {
    "C_soft_vote",
    "AMPidentifier",
    "bert",
    "att",
    "C_hard_vote",
}

MODEL_NAME_ALIASES = {
    "C_AMP-predict": "C_AMPs-predict",
}


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip())
    return out.strip("._") or "unnamed"


def _standard_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(">", "", regex=False)
    )


def _load_manual_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    if "Sequence" not in df.columns:
        raise ValueError(f"{path} is missing required column: Sequence")
    if "label" not in df.columns:
        raise ValueError(f"{path} is missing required column: label")
    return df


def _split_names(values: list[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def _model_name_key(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def standardize_manual_predictions(path: Path, exclude_models: set[str] | None = None) -> pd.DataFrame:
    df = _load_manual_csv(path)
    out = pd.DataFrame(
        {
            "Standard_ID": _standard_id(df["Sequence"]),
            "True_Label": pd.to_numeric(df["label"], errors="coerce"),
        }
    )

    skip = {"sequence", "label", "standard_id", "true_label"}
    excluded = {_model_name_key(x) for x in (exclude_models or set())}
    used: set[str] = set()
    for col in df.columns:
        if col.strip().lower() in skip:
            continue
        if _model_name_key(col) in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        model_name = _safe_name(MODEL_NAME_ALIASES.get(col, col))
        base = f"{model_name}_Prob"
        prob_col = base
        i = 2
        while prob_col in used or prob_col in out.columns:
            prob_col = f"{model_name}_{i}_Prob"
            i += 1
        used.add(prob_col)
        out[prob_col] = values

    if not any(c.endswith("_Prob") for c in out.columns):
        raise ValueError(f"{path} has no numeric model probability columns")
    out = out.drop_duplicates(subset=["Standard_ID"], keep="first")
    return out


def _finite(value: Any) -> str:
    try:
        x = float(value)
        return f"{x:.4f}" if np.isfinite(x) else "NA"
    except Exception:
        return "NA"


def _precision_recall_at_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _model_color_map(names: list[str]) -> dict[str, Any]:
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        colors.extend([cmap(i) for i in range(cmap.N)])
    return {name: colors[i % len(colors)] for i, name in enumerate(names)}


def _annotate_bars(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 0.015, 1.03),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def write_curves(predictions: pd.DataFrame, output_path: Path) -> None:
    label = pd.to_numeric(predictions["True_Label"], errors="coerce")
    model_cols = [c for c in predictions.columns if c.endswith("_Prob")]
    model_names = [c[:-5] for c in model_cols]
    colors = _model_color_map(model_names)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_roc, ax_pr, ax_precision, ax_recall = axes.ravel()

    precision_names: list[str] = []
    precision_values: list[float] = []
    recall_names: list[str] = []
    recall_values: list[float] = []

    for col in model_cols:
        name = col[:-5]
        prob = pd.to_numeric(predictions[col], errors="coerce")
        valid = label.isin([0, 1]) & prob.notna() & np.isfinite(prob) & prob.between(0.0, 1.0)
        if not valid.any() or len(np.unique(label.loc[valid])) < 2:
            continue
        y_true = label.loc[valid].values.astype(int)
        y_prob = prob.loc[valid].values.astype(float)
        y_pred = (y_prob >= DEFAULT_THRESHOLD).astype(int)

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)
            ax_roc.plot(fpr, tpr, label=f"{name} ({auroc:.3f})", color=colors[name], linewidth=1.8)
        except Exception:
            pass
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            ax_pr.plot(recall, precision, label=f"{name} ({auprc:.3f})", color=colors[name], linewidth=1.8)
        except Exception:
            pass
        p_at_threshold, r_at_threshold = _precision_recall_at_threshold(y_true, y_pred)
        precision_names.append(name)
        precision_values.append(float(p_at_threshold))
        recall_names.append(name)
        recall_values.append(float(r_at_threshold))

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_title("A. ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)

    ax_pr.set_title("B. Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)

    order = np.argsort(precision_values)[::-1] if precision_values else []
    precision_order_names = [precision_names[i] for i in order]
    precision_bars = ax_precision.bar(
        precision_order_names,
        [precision_values[i] for i in order],
        color=[colors[name] for name in precision_order_names],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_precision.set_title("C. Precision at Threshold 0.5")
    ax_precision.set_ylim(0, 1.05)
    ax_precision.tick_params(axis="x", labelrotation=75)
    _annotate_bars(ax_precision, precision_bars)

    order = np.argsort(recall_values)[::-1] if recall_values else []
    recall_order_names = [recall_names[i] for i in order]
    recall_bars = ax_recall.bar(
        recall_order_names,
        [recall_values[i] for i in order],
        color=[colors[name] for name in recall_order_names],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_recall.set_title("D. Recall at Threshold 0.5")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.tick_params(axis="x", labelrotation=75)
    _annotate_bars(ax_recall, recall_bars)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_markdown(eval_result: dict[str, Any], output_path: Path) -> None:
    rows = []
    for model, metrics in eval_result.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "model": model,
                "AUPRC": metrics.get("AUPRC"),
                "AUROC": metrics.get("AUROC"),
                "MCC": metrics.get("MCC"),
                "Recall": metrics.get("Recall"),
                "Precision": metrics.get("Precision"),
            }
        )
    rows.sort(
        key=lambda r: (
            float(r["AUPRC"]) if r["AUPRC"] is not None else -1,
            float(r["MCC"]) if r["MCC"] is not None else -1,
            float(r["Recall"]) if r["Recall"] is not None else -1,
        ),
        reverse=True,
    )
    lines = [
        "# Manual Prediction Evaluation Summary",
        "",
        "This file was generated from manually supplied prediction CSVs.",
        "",
        "| Rank | Model | AUPRC | AUROC | MCC | Recall | Precision |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows[:10], 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row["model"]),
                    _finite(row["AUPRC"]),
                    _finite(row["AUROC"]),
                    _finite(row["MCC"]),
                    _finite(row["Recall"]),
                    _finite(row["Precision"]),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_one(dataset_name: str, input_path: Path, results_dir: Path, archive_dir: Path, exclude_models: set[str]) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    safe_dataset = _safe_name(dataset_name)
    raw_dir = archive_dir / safe_dataset
    out_dir = results_dir / safe_dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    archived = raw_dir / input_path.name
    shutil.copy2(input_path, archived)

    predictions = standardize_manual_predictions(input_path, exclude_models=exclude_models)
    predictions_path = out_dir / "final_results_with_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    evaluated = evaluate_prediction_table(predictions_path, out_dir)
    write_curves(predictions, out_dir / "evaluation_curves.png")
    write_summary_markdown(evaluated["eval_result"], out_dir / "critic_individual.md")

    print(f"[OK] {safe_dataset}")
    print(f"     raw: {archived}")
    print(f"     standardized: {predictions_path}")
    print(f"     eval: {out_dir / 'eval_result.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manual AMP prediction CSVs into data/results.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="DATASET=CSV",
        help="Dataset name and CSV path. Can be repeated. If omitted, the three known Downloads files are used.",
    )
    parser.add_argument("--results-dir", default=str(ROOT / "data" / "results_manual"), help="Where evaluation result folders are written.")
    parser.add_argument("--archive-dir", default=str(ROOT / "data" / "manual_predictions"), help="Where raw manual CSV copies are archived.")
    parser.add_argument("--exclude-model", action="append", default=[], help="Model column name to exclude. Can be repeated or comma-separated.")
    parser.add_argument("--include-default-excluded", action="store_true", help="Do not exclude the current default skipped models.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items: dict[str, Path] = {}
    if args.input:
        for item in args.input:
            if "=" not in item:
                raise ValueError(f"--input must be DATASET=CSV, got: {item}")
            name, path = item.split("=", 1)
            items[name.strip()] = Path(path.strip())
    else:
        items = dict(DEFAULT_INPUTS)

    results_dir = Path(args.results_dir)
    archive_dir = Path(args.archive_dir)
    exclude_models = set() if args.include_default_excluded else set(DEFAULT_EXCLUDED_MODELS)
    exclude_models.update(_split_names(args.exclude_model))
    for dataset_name, input_path in items.items():
        import_one(dataset_name, input_path, results_dir, archive_dir, exclude_models)

    manifest = {
        "results_dir": str(results_dir),
        "archive_dir": str(archive_dir),
        "datasets": list(items.keys()),
        "excluded_models": sorted(exclude_models),
        "ranking_policy": "Metric weights are generated by the downstream 50-round evidence meeting; no model-specific priority bonus is applied.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manual_import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Manual results imported under: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
