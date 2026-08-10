# -*- coding: utf-8 -*-
"""Build the filtered-cohort cross-dataset publication figure."""
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
RANKING_PATH = RESULTS_DIR / "posthoc_filtered_view" / "posthoc_filtered_model_ranking.csv"
OUTPUT_DIR = RESULTS_DIR / "publication_figures_filtered"

DATASETS = [
    ("C_AMPs-predict_test", "C_AMPs-predict test", "#3C6E8F"),
    ("Veltri_test", "Veltri test", "#D98555"),
    ("ProteoGPT_all_predictions", "ProteoGPT test", "#4C956C"),
]

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

HIGHLIGHT_MODELS = ["HMD-AMP", "C_AMPs-predict", "AMPsorter"]
MODEL_COLORS = {
    "HMD-AMP": "#0072B2",
    "C_AMPs-predict": "#D55E00",
    "AMPsorter": "#009E73",
}


def _find_label_column(frame: pd.DataFrame) -> str:
    aliases = {"label", "y_true", "true_label", "amp_label"}
    for column in frame.columns:
        if str(column).strip().lower() in aliases:
            return str(column)
    raise ValueError("No ground-truth label column found")


def load_filtered_models() -> tuple[list[str], pd.DataFrame]:
    ranking = pd.read_csv(RANKING_PATH)
    required = {"filtered_rank", "global_rank", "model"}
    missing = required.difference(ranking.columns)
    if missing:
        raise ValueError(f"Filtered ranking is missing columns: {sorted(missing)}")
    ranking = ranking.sort_values("filtered_rank", kind="stable").reset_index(drop=True)
    models = ranking["model"].astype(str).tolist()
    missing_targets = [model for model in HIGHLIGHT_MODELS if model not in models]
    if missing_targets:
        raise ValueError(f"Highlighted models are missing: {missing_targets}")
    return models, ranking


def load_source_data(
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    long_rows: list[dict[str, object]] = []
    sample_stats: dict[str, dict[str, float]] = {}

    for dataset_key, dataset_label, _ in DATASETS:
        eval_path = RESULTS_DIR / dataset_key / "eval_result.json"
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        frame = pd.DataFrame.from_dict(eval_data, orient="index")
        frame.index = frame.index.astype(str)
        frame.index.name = "model"
        missing_models = [model for model in models if model not in frame.index]
        if missing_models:
            raise ValueError(f"{dataset_key} is missing models: {missing_models}")
        frame = frame.reindex(models)

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

        for metric_key, metric_label, higher_is_better in METRICS:
            if metric_key not in frame.columns:
                raise ValueError(f"{dataset_key} is missing metric: {metric_key}")
            values = pd.to_numeric(frame[metric_key], errors="coerce")
            if values.isna().any():
                missing_value_models = values.index[values.isna()].tolist()
                raise ValueError(
                    f"{dataset_key}/{metric_key} has missing values for: {missing_value_models}"
                )
            percentiles = values.rank(
                method="average", pct=True, ascending=higher_is_better
            )
            for model in models:
                long_rows.append(
                    {
                        "dataset": dataset_key,
                        "dataset_label": dataset_label,
                        "model": model,
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "higher_is_better": higher_is_better,
                        "raw_value": float(values.loc[model]),
                        "performance_percentile": float(percentiles.loc[model]),
                    }
                )

    source = pd.DataFrame(long_rows)
    rank_summary: dict[str, dict[str, float]] = {}
    for dataset_key, _, _ in DATASETS:
        subset = source[source["dataset"] == dataset_key]
        consensus = subset.groupby("model", sort=False)["performance_percentile"].median()
        ranks = consensus.rank(method="average", ascending=False)
        rank_summary[dataset_key] = ranks.to_dict()
    return source, sample_stats, rank_summary


def build_heatmap(source: pd.DataFrame, models: list[str]) -> np.ndarray:
    blocks = []
    for dataset_key, _, _ in DATASETS:
        subset = source[source["dataset"] == dataset_key]
        pivot = subset.pivot(index="model", columns="metric", values="performance_percentile")
        block = pivot.reindex(index=models, columns=[metric[0] for metric in METRICS]).to_numpy()
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _highlight_annotations(model: str) -> dict[str, object]:
    layouts = {
        "HMD-AMP": {"xytext": (-8, -13), "ha": "right"},
        "C_AMPs-predict": {"xytext": (8, 11), "ha": "left"},
        "AMPsorter": {"xytext": (8, -2), "ha": "left"},
    }
    return layouts[model]


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

    heat = build_heatmap(source, models)
    image = ax_heat.imshow(
        heat,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    metric_labels = [metric[1] for _ in DATASETS for metric in METRICS]
    ax_heat.set_xticks(
        np.arange(len(metric_labels)),
        metric_labels,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )
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

    for column_index in range(heat.shape[1]):
        column = heat[:, column_index]
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
    cbar.set_label("Within-filtered-cohort performance percentile (higher is better)")
    cbar.outline.set_linewidth(0.5)

    x_positions = np.arange(len(DATASETS))
    for model in models:
        y_values = [rank_summary[key].get(model, np.nan) for key, _, _ in DATASETS]
        if model in HIGHLIGHT_MODELS:
            color = MODEL_COLORS[model]
            ax_rank.plot(
                x_positions,
                y_values,
                color=color,
                marker="o",
                markersize=4.2,
                linewidth=1.8,
                zorder=3,
            )
            ax_rank.text(2.08, y_values[-1], model, color=color, va="center", fontsize=6.5)
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
    ax_rank.set_yticks([1, 3, 6, 9, 12, 15])
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
        )
        dataset_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=5.5,
                label=dataset_label,
            )
        )

    for model in HIGHLIGHT_MODELS:
        xs, ys = [], []
        for dataset_key, _, _ in DATASETS:
            point = source[(source["dataset"] == dataset_key) & (source["model"] == model)]
            raw = point.set_index("metric")["raw_value"]
            xs.append(float(raw["Recall"]))
            ys.append(float(raw["Precision"]))
        color = MODEL_COLORS[model]
        ax_pr.plot(xs, ys, color=color, linewidth=1.15, alpha=0.85, zorder=3)
        ax_pr.scatter(
            xs,
            ys,
            s=29,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            zorder=4,
        )
        layout = _highlight_annotations(model)
        ax_pr.annotate(
            model,
            (xs[-1], ys[-1]),
            xytext=layout["xytext"],
            textcoords="offset points",
            color=color,
            fontsize=6.2,
            fontweight="bold",
            ha=layout["ha"],
            va="center",
        )

    ax_pr.set_xlim(-0.02, 1.03)
    ax_pr.set_ylim(-0.02, 1.03)
    ax_pr.set_xlabel("Recall at diagnostic threshold 0.5")
    ax_pr.set_ylabel("Precision at diagnostic threshold 0.5")
    ax_pr.set_title("c   Precision-recall operating-point trade-off", loc="left", fontweight="bold")
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
    stem = OUTPUT_DIR / "filtered_cross_dataset_performance"
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "tiff": stem.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


def write_supporting_files(
    source: pd.DataFrame,
    sample_stats: dict[str, dict[str, float]],
    models: list[str],
    ranking: pd.DataFrame,
    rank_summary: dict[str, dict[str, float]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_out = source.copy()
    source_out["dataset_consensus_rank"] = source_out.apply(
        lambda row: rank_summary[row["dataset"]].get(row["model"], np.nan), axis=1
    )
    source_out = source_out.merge(
        ranking[["model", "filtered_rank", "global_rank"]], on="model", how="left", validate="many_to_one"
    )
    source_out.to_csv(
        OUTPUT_DIR / "filtered_cross_dataset_performance_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    caption = """# Figure legend

**Figure X | Cross-dataset performance of the current filtered AMP-model cohort.**
**a,** Performance landscape for 15 retained AMP prediction models across three independent test sets. Raw evaluation values were converted to percentiles within the displayed 15-model cohort so metrics with different scales could be shown together. Brier score and expected calibration error (ECE) were direction-reversed only for percentile ranking; darker colour consistently denotes better performance. White points mark the best displayed model for each metric and test set. **b,** Dataset-specific model-rank trajectories based on the median percentile across the eight displayed metrics. HMD-AMP, C_AMPs-predict and AMPsorter are highlighted; all other retained models are grey. **c,** Precision-recall operating points at threshold 0.5. Point area scales with AUPRC, and highlighted trajectories connect each focal model across the three test sets. The test sets contained 59,311 sequences (1.75% positive), 1,203 sequences (51.04% positive), and 1,796 sequences (40.37% positive), respectively. The display is post-hoc filtered: pepnet_standard, amplify_imb and amplify_bal were omitted according to the previously recorded ranking-view rule. Raw metric values are unchanged, but this panel must not be interpreted as an unbiased global Top3 comparison.
"""
    (OUTPUT_DIR / "filtered_cross_dataset_performance_caption.md").write_text(caption, encoding="utf-8")

    qa = f"""# Figure QA notes

- Core conclusion: the three focal models show distinct, dataset-dependent strengths within the current filtered cohort.
- Archetype: quantitative grid with one hero heatmap and two supporting robustness panels.
- Backend: Python/matplotlib only.
- Final size: 183 mm double-column; editable SVG/PDF, 300 dpi PNG and compressed 600 dpi TIFF.
- Input cohort: 18 complete benchmark models.
- Displayed cohort: {len(models)} models.
- Explicitly excluded from this display: pepnet_standard, amplify_imb, amplify_bal.
- Exclusion rule: retain the three focal models and all models originally ranked below the lowest-ranked focal model.
- Scientific boundary: this is a post-hoc, result-conditioned display and is not valid evidence of an unbiased global Top3.
- Raw evaluation values were not changed. Percentiles and dataset-specific consensus ranks were recalculated within the displayed cohort.
- Included observations: {len(models)} models x {len(DATASETS)} datasets x {len(METRICS)} metrics = {len(source)} source rows.
- No missing model-metric values were accepted by the plotting script.
- Brier score and ECE were reversed only for percentile direction; raw values remain in source data.
- No confidence intervals or significance claims are shown because bootstrap iterations were unavailable.
- Sample statistics: {json.dumps(sample_stats, ensure_ascii=False)}
"""
    (OUTPUT_DIR / "filtered_cross_dataset_performance_qa.md").write_text(qa, encoding="utf-8")

    contract = {
        "core_conclusion": "The three focal AMP models have complementary, dataset-dependent operating profiles within the current filtered cohort.",
        "evidence_chain": {
            "a": "Eight raw metrics summarized as within-filtered-cohort percentiles across three test sets.",
            "b": "Median-percentile rank shifts expose dataset dependence.",
            "c": "Raw precision-recall operating points expose threshold-level trade-offs.",
        },
        "archetype": "quantitative grid",
        "backend": "python",
        "displayed_models": models,
        "highlighted_models": HIGHLIGHT_MODELS,
        "excluded_models": ["pepnet_standard", "amplify_imb", "amplify_bal"],
        "posthoc_result_conditioned_filter": True,
        "valid_for_unbiased_global_top3_claim": False,
        "exports": ["png", "svg", "pdf", "tiff", "csv", "caption", "qa"],
    }
    (OUTPUT_DIR / "filtered_cross_dataset_figure_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    models, ranking = load_filtered_models()
    source, sample_stats, rank_summary = load_source_data(models)
    write_supporting_files(source, sample_stats, models, ranking, rank_summary)
    outputs = plot_figure(source, sample_stats, models, rank_summary)
    print("Filtered publication figure written:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
