# -*- coding: utf-8 -*-
"""Create a publication-grade figure from the three stage-1 evaluation tables."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


RESULTS_DIR = ROOT / "data" / "results_manual"
OUTPUT_DIR = RESULTS_DIR / "publication_figures"

DATASETS = [
    ("C_AMPs-predict_test", "C_AMPs-predict test", "#3C6E8F"),
    ("Veltri_test", "Veltri test", "#D98555"),
    ("ProteoGPT_all_predictions", "ProteoGPT test", "#4C956C"),
]

# These eight outcomes cover ranking, discrimination, threshold behaviour, and
# calibration without double-counting AUPRC-Lift or configuration fields.
METRICS = [
    ("AUPRC", "AUPRC", True),
    ("AUROC", "AUROC", True),
    ("MCC", "MCC", True),
    ("BalancedAccuracy", "BAcc", True),
    ("Recall", "Recall", True),
    ("Precision", "Precision", True),
    ("BrierScore", "Brier", False),
    ("ECE", "ECE", False),
]

MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def _find_label_column(frame: pd.DataFrame) -> str:
    aliases = {"label", "y_true", "true_label", "amp_label"}
    for column in frame.columns:
        if str(column).strip().lower() in aliases:
            return str(column)
    raise ValueError("No ground-truth label column found")


def load_source_data() -> tuple[pd.DataFrame, dict[str, dict[str, float]], list[str], dict[str, dict[str, float]]]:
    long_rows = []
    sample_stats: dict[str, dict[str, float]] = {}
    metric_tables: dict[str, dict[str, float]] = {}

    for dataset_key, dataset_label, _ in DATASETS:
        eval_path = RESULTS_DIR / dataset_key / "eval_result.json"
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        prediction_path = RESULTS_DIR / dataset_key / "final_results_with_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        label_col = _find_label_column(predictions)
        labels_all = pd.to_numeric(predictions[label_col], errors="coerce")
        valid_label_mask = labels_all.notna()
        labels = labels_all.loc[valid_label_mask]
        sample_stats[dataset_key] = {
            "total_rows": int(len(labels_all)),
            "missing_labels": int((~valid_label_mask).sum()),
            "n": int(len(labels)),
            "positive": int((labels == 1).sum()),
            "negative": int((labels == 0).sum()),
            "prevalence": float((labels == 1).mean()),
        }

        frame = pd.DataFrame.from_dict(eval_data, orient="index")
        frame.index.name = "model"
        for metric_key, metric_label, higher_is_better in METRICS:
            values = pd.to_numeric(frame[metric_key], errors="coerce")
            percentiles = values.rank(method="average", pct=True, ascending=higher_is_better)
            metric_tables[f"{dataset_key}|{metric_key}"] = values.to_dict()
            for model in frame.index:
                raw_value = values.loc[model]
                percentile = percentiles.loc[model]
                long_rows.append(
                    {
                        "dataset": dataset_key,
                        "dataset_label": dataset_label,
                        "model": model,
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "higher_is_better": higher_is_better,
                        "raw_value": float(raw_value) if pd.notna(raw_value) else np.nan,
                        "performance_percentile": float(percentile) if pd.notna(percentile) else np.nan,
                    }
                )

    source = pd.DataFrame(long_rows)
    models = sorted(source["model"].unique())
    ranking_path = RESULTS_DIR / "model_ranking_50_rounds.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        ranked_models = [model for model in ranking["model"].astype(str) if model in models]
        models = ranked_models + [model for model in models if model not in ranked_models]

    rank_summary: dict[str, dict[str, float]] = {}
    for dataset_key, _, _ in DATASETS:
        subset = source[source["dataset"] == dataset_key]
        consensus = subset.groupby("model")["performance_percentile"].median()
        ranks = consensus.rank(method="average", ascending=False)
        rank_summary[dataset_key] = ranks.to_dict()

    return source, sample_stats, models, rank_summary


def _build_heatmap(source: pd.DataFrame, models: list[str]) -> np.ndarray:
    columns = []
    for dataset_key, _, _ in DATASETS:
        subset = source[source["dataset"] == dataset_key]
        pivot = subset.pivot(index="model", columns="metric", values="performance_percentile")
        columns.extend(pivot.reindex(index=models, columns=[m[0] for m in METRICS]).to_numpy().T)
    return np.asarray(columns).T


def plot_figure(
    source: pd.DataFrame,
    sample_stats: dict[str, dict[str, float]],
    models: list[str],
    rank_summary: dict[str, dict[str, float]],
) -> dict[str, Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 9.25), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0], width_ratios=[1.0, 1.0])
    ax_heat = fig.add_subplot(grid[0, :])
    ax_rank = fig.add_subplot(grid[1, 0])
    ax_pr = fig.add_subplot(grid[1, 1])

    heat = _build_heatmap(source, models)
    image = ax_heat.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    metric_labels = [metric[1] for _ in DATASETS for metric in METRICS]
    ax_heat.set_xticks(np.arange(len(metric_labels)), metric_labels, rotation=55, ha="right", rotation_mode="anchor")
    ax_heat.set_yticks(np.arange(len(models)), models)
    ax_heat.set_title("a   Cross-dataset performance landscape", loc="left", fontweight="bold", pad=34)
    ax_heat.tick_params(length=0)
    for boundary in range(1, len(DATASETS)):
        x = boundary * len(METRICS) - 0.5
        ax_heat.axvline(x, color="white", linewidth=2.2)
        ax_heat.axvline(x, color="#4B5563", linewidth=0.45)

    for dataset_index, (dataset_key, dataset_label, color) in enumerate(DATASETS):
        start = dataset_index * len(METRICS) - 0.5
        width = len(METRICS)
        stats = sample_stats[dataset_key]
        ax_heat.add_patch(
            Rectangle((start, -1.40), width, 0.22, facecolor=color, edgecolor="none", clip_on=False)
        )
        ax_heat.text(
            start + width / 2,
            -1.73,
            f"{dataset_label}\nn={stats['n']:,}; prevalence={stats['prevalence']:.1%}",
            ha="center",
            va="bottom",
            color="#1F2937",
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )

    # White points identify the best model for each metric within each dataset.
    for column_index in range(heat.shape[1]):
        column = heat[:, column_index]
        if np.isfinite(column).any():
            best = np.flatnonzero(np.isclose(column, np.nanmax(column)))
            ax_heat.scatter(
                np.full(len(best), column_index),
                best,
                s=8,
                c="white",
                edgecolors="#263238",
                linewidths=0.35,
                zorder=3,
            )

    cbar = fig.colorbar(image, ax=ax_heat, orientation="horizontal", fraction=0.035, pad=0.10, aspect=45)
    cbar.set_label("Within-dataset performance percentile (higher is better)")
    cbar.outline.set_linewidth(0.5)

    ranking_table = pd.read_csv(RESULTS_DIR / "model_ranking_50_rounds.csv")
    highlight_models = [m for m in ranking_table["model"].astype(str).head(3) if m in models]
    x_positions = np.arange(len(DATASETS))
    for model in models:
        y_values = [rank_summary[key].get(model, np.nan) for key, _, _ in DATASETS]
        if model in highlight_models:
            index = highlight_models.index(model)
            ax_rank.plot(
                x_positions,
                y_values,
                color=MODEL_COLORS[index],
                marker="o",
                markersize=4.2,
                linewidth=1.8,
                zorder=3,
            )
            ax_rank.text(2.08, y_values[-1], model, color=MODEL_COLORS[index], va="center", fontsize=6.5)
        else:
            ax_rank.plot(
                x_positions,
                y_values,
                color="#9CA3AF",
                marker="o",
                markersize=2.2,
                linewidth=0.65,
                alpha=0.46,
                zorder=1,
            )
    ax_rank.set_xticks(x_positions, ["C_AMPs", "Veltri", "ProteoGPT"])
    ax_rank.set_xlim(-0.12, 2.72)
    ax_rank.set_ylim(len(models) + 0.6, 0.4)
    ax_rank.set_yticks([1, 3, 6, 9, 12, 15, 18])
    ax_rank.set_ylabel("Consensus rank across eight metrics")
    ax_rank.set_title("b   Model rank shifts across test sets", loc="left", fontweight="bold")
    ax_rank.grid(axis="y", color="#E5E7EB", linewidth=0.55)

    dataset_handles = []
    for dataset_key, dataset_label, color in DATASETS:
        subset = source[source["dataset"] == dataset_key]
        wide = subset.pivot(index="model", columns="metric", values="raw_value")
        sizes = 16 + 72 * wide["AUPRC"].clip(0, 1)
        ax_pr.scatter(
            wide["Recall"],
            wide["Precision"],
            s=sizes,
            color=color,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.45,
            label=dataset_label,
        )
        dataset_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white",
                   markersize=5.5, label=dataset_label)
        )

    # Connect each final Top3 model across datasets to expose its operating-point shift.
    for model_index, model in enumerate(highlight_models):
        xs, ys = [], []
        for dataset_key, _, _ in DATASETS:
            point = source[(source["dataset"] == dataset_key) & (source["model"] == model)]
            raw = point.set_index("metric")["raw_value"]
            xs.append(float(raw["Recall"]))
            ys.append(float(raw["Precision"]))
        ax_pr.plot(xs, ys, color=MODEL_COLORS[model_index], linewidth=1.15, alpha=0.85, zorder=3)
        ax_pr.scatter(xs, ys, s=29, facecolors="none", edgecolors=MODEL_COLORS[model_index], linewidths=1.0, zorder=4)
        annotation_layout = {
            0: {"xytext": (-8, 10), "ha": "right"},
            1: {"xytext": (8, -12), "ha": "left"},
            2: {"xytext": (7, 10), "ha": "left"},
        }[model_index]
        ax_pr.annotate(
            model,
            (xs[-1], ys[-1]),
            xytext=annotation_layout["xytext"],
            textcoords="offset points",
            color=MODEL_COLORS[model_index],
            fontsize=6.2,
            fontweight="bold",
            ha=annotation_layout["ha"],
            va="center",
        )

    ax_pr.set_xlim(-0.02, 1.03)
    ax_pr.set_ylim(-0.02, 1.03)
    ax_pr.set_xlabel("Recall at diagnostic threshold 0.5")
    ax_pr.set_ylabel("Precision at diagnostic threshold 0.5")
    ax_pr.set_title("c   Precision–recall operating-point trade-off", loc="left", fontweight="bold")
    ax_pr.grid(color="#E5E7EB", linewidth=0.55)
    ax_pr.legend(handles=dataset_handles, loc="lower left", title="Test set", title_fontsize=6.5)
    ax_pr.text(
        0.99,
        0.02,
        "Point area scales with AUPRC",
        transform=ax_pr.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#4B5563",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "stage1_cross_dataset_performance.png"
    svg_path = OUTPUT_DIR / "stage1_cross_dataset_performance.svg"
    pdf_path = OUTPUT_DIR / "stage1_cross_dataset_performance.pdf"
    tiff_path = OUTPUT_DIR / "stage1_cross_dataset_performance.tiff"
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
    plt.close(fig)
    return {"png": png_path, "svg": svg_path, "pdf": pdf_path, "tiff": tiff_path}


def write_supporting_files(
    source: pd.DataFrame,
    sample_stats: dict[str, dict[str, float]],
    models: list[str],
    rank_summary: dict[str, dict[str, float]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_out = source.copy()
    source_out["dataset_consensus_rank"] = source_out.apply(
        lambda row: rank_summary[row["dataset"]].get(row["model"], np.nan), axis=1
    )
    source_out.to_csv(OUTPUT_DIR / "stage1_cross_dataset_performance_source_data.csv", index=False, encoding="utf-8-sig")

    caption = """# Figure legend

**Figure X | AMP model performance is strongly dataset dependent across three independent test sets.**
**a,** Performance landscape for 18 AMP prediction models. Values are converted to within-dataset percentiles so that metrics with different scales and datasets with markedly different class prevalence can be displayed together. Brier score and expected calibration error (ECE) are direction-reversed before percentile ranking; higher colour intensity therefore consistently denotes better performance. White points mark the best-performing model for each metric within each test set. **b,** Model-rank trajectories based on the median percentile across the eight displayed metrics. The three models ranked highest by the independent 50-round weight deliberation are highlighted; all remaining models are shown in grey. **c,** Precision–recall operating points at the diagnostic threshold of 0.5. Point area scales with AUPRC, and highlighted trajectories connect each final Top3 model across the three test sets. The test sets contained 59,311 sequences (1.75% positive), 1,203 sequences (51.04% positive), and 1,796 sequences (40.37% positive), respectively. Because no validation-derived threshold or bootstrap confidence intervals were available, threshold-dependent results are diagnostic and no inferential significance claims are made.
"""
    (OUTPUT_DIR / "stage1_cross_dataset_performance_caption.md").write_text(caption, encoding="utf-8")

    qa = f"""# Figure QA notes

- Core conclusion: AMP model advantages are strongly dataset dependent.
- Archetype: quantitative grid with one hero heatmap and two robustness panels.
- Backend: Python/matplotlib only.
- Final width: 183 mm (double-column); editable SVG/PDF plus 300 dpi PNG and compressed 600 dpi TIFF.
- Included observations: {len(models)} models x {len(DATASETS)} datasets x {len(METRICS)} displayed metrics = {len(source)} source rows.
- No model or dataset was excluded.
- Main-figure metrics: AUPRC, AUROC, MCC, Balanced Accuracy, Recall, Precision, Brier score, and ECE.
- Fields not drawn in the main figure: Coverage (constant), Threshold and Source (configuration/provenance), AUPRC-Lift (dataset-prevalence transform of AUPRC), and secondary/redundant threshold metrics. They remain in the unmodified supplementary evaluation tables.
- Brier score and ECE were reversed only for percentile colour direction; raw values are preserved in the source-data CSV.
- Bootstrap iterations in the source evaluation were 0; no confidence intervals or significance annotations were added.
- Sample statistics: {json.dumps(sample_stats, ensure_ascii=False)}
"""
    (OUTPUT_DIR / "stage1_cross_dataset_performance_qa.md").write_text(qa, encoding="utf-8")


def main() -> int:
    source, sample_stats, models, rank_summary = load_source_data()
    write_supporting_files(source, sample_stats, models, rank_summary)
    outputs = plot_figure(source, sample_stats, models, rank_summary)
    print("Publication figure written:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
