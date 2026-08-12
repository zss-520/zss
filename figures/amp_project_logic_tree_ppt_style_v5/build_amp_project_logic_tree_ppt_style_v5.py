from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

V4_SCRIPT = (
    ROOT / "figures" / "amp_project_logic_tree_v4"
    / "build_amp_project_logic_tree_v4.py"
)
SPEC = importlib.util.spec_from_file_location("amp_project_tree_v4", V4_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import preserved project tree: {V4_SCRIPT}")
V4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V4)

WIDTH_MM, HEIGHT_MM = 210.0, 297.0
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4

NAVY = "#111C2E"
BLUE = "#0874A6"
BLUE_DARK = "#0B486B"
BLUE_LIGHT = "#EAF2F7"
BLUE_PALE = "#F5F9FC"
GREEN = "#16844A"
ORANGE = "#F18727"
PURPLE = "#7542B5"
GRAY = "#617184"
LINE = "#C7D3DD"
WHITE = "#FFFFFF"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 5.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})


ROW_HEIGHTS = [0.080, 0.160, 0.110, 0.150, 0.160, 0.095, 0.070]
ROW_GAP = 0.010


def compact_lines(stage: str, layer: str, lines: list[str]) -> list[str]:
    if stage == "0":
        return lines
    if stage == "1" and layer == "agents":
        return [
            "agents/shared/ + agents/deepseek_meeting/",
            "evidence_compressor | info_extractor",
            "model_dataset | metric | critic -> chief (.md)",
        ]
    if stage == "1" and layer == "data":
        return [
            "evidence_pool.json -> compact_evidence_pool.json",
            "-> screening_decisions.csv -> literature memory.json/.md",
            "dataset_candidate_pool.json -> dataset_plan.json",
            "exports/: recommended_models / datasets / metrics.csv",
        ]
    if stage == "2" and layer == "agents":
        return [
            "agents/model_onboarding/",
            "repository_inspector_system.md",
            "repository_inspector_task.md",
            "agents/model_execution/",
            "self_heal_system.md | self_heal_task.md",
        ]
    if stage == "2" and layer == "engines":
        return [
            "new_model_onboarding.py -> hpc_model_ops.py",
            "README -> registry -> upload -> Conda/deps",
            "-> smoke/self-heal",
        ]
    if stage == "2" and layer == "data":
        return [
            "data/local_registry.json",
            "repository + environment + command + status",
            "data/hpc_self_heal/",
            "failure logs + bounded repair records",
        ]
    if stage == "3" and layer == "agents":
        return [
            "agents/runtime_prompts/",
            "benchmark_architect | coder | critic | pi (.md)",
            "dataset_etl_agent.md",
        ]
    if stage == "3" and layer == "engines":
        return [
            "AUTO",
            "main.py -> run_meeting.py -> workflow_utils.py",
            "MANUAL",
            "import_manual_prediction_results.py",
            "COMMON -> scientific_evaluation.py",
        ]
    if stage == "3" and layer == "data":
        return [
            "data/runs/{run_id}/",
            "manifest + artifacts + results/{dataset}/",
            "data/results_manual/{dataset}/",
            "standardized predictions.csv + eval_result.json",
            "scientific_evaluation.json/.md + Critic report",
        ]
    if stage == "4" and layer == "agents":
        return [
            "agents/weight_meeting/: shared_system.md",
            "literature | statistics | screening",
            "-> reviewer -> chief",
            "research_advisor.md",
        ]
    if stage == "4" and layer == "engines":
        return [
            "codex_agent_weight_meeting.py",
            "or llm_agent_weight_meeting.py",
            "-> iterative_weight_meeting.py",
            "blind proposals -> audit -> Chief",
            "-> 50 bounded updates",
        ]
    if stage == "4" and layer == "data":
        return [
            "agent_evidence_bundle -> three expert proposals",
            "-> reviewer_agent_audit -> chief_initial_decision",
            "-> rounds/round_001 ... round_050.json",
            "-> metric weights / model scores",
            "-> final ranking + report",
        ]
    if stage == "5" and layer == "agents":
        return ["Consumes Stage 4 Chief decision", "Research Advisor synthesis"]
    if stage == "6" and layer == "agents":
        return ["Presentation only; never feeds values back into scoring"]
    return lines


def short_arrow(ax, x1, x2, y, color=ORANGE):
    patch = FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.75, color=color, zorder=5,
    )
    ax.add_patch(patch)


def draw_metric_band(fig, state):
    ax = fig.add_axes([0.070, 0.790, 0.860, 0.055])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, 1], [0.96, 0.96], color=BLUE, lw=0.75)
    ax.plot([0, 1], [0.04, 0.04], color=LINE, lw=0.55)
    metrics = [
        (f"{state['evidence_papers']:,}", "literature records"),
        (f"{state['priority_models']}", "recommended models"),
        (f"{state['dataset_count']}", "test datasets"),
        (f"{state['round_files']}", "decision rounds"),
    ]
    for index, (value, label) in enumerate(metrics):
        x0, x1 = index / 4, (index + 1) / 4
        if index:
            ax.plot([x0, x0], [0.08, 0.92], color=LINE, lw=0.55)
        ax.text((x0 + x1) / 2, 0.60, value, fontsize=8.0, weight="bold",
                color=NAVY, ha="center", va="center")
        ax.text((x0 + x1) / 2, 0.28, label, fontsize=5.0, color=GRAY,
                ha="center", va="center")


def draw_tree(fig, state):
    ax = fig.add_axes([0.070, 0.105, 0.860, 0.665])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_x = 0.000
    agent_x = 0.205
    engine_x = 0.470
    data_x = 0.715
    trunk_x = 0.028

    ax.text(stage_x, 1.012, "PROJECT TREE", fontsize=5.0, weight="bold", color=BLUE_DARK, va="bottom")
    ax.text(agent_x, 1.012, "AGENT CONTRACTS", fontsize=5.0, weight="bold", color=BLUE_DARK, va="bottom")
    ax.text(engine_x, 1.012, "EXECUTION", fontsize=5.0, weight="bold", color=BLUE_DARK, va="bottom")
    ax.text(data_x, 1.012, "PERSISTED EVIDENCE", fontsize=5.0, weight="bold", color=BLUE_DARK, va="bottom")
    ax.plot([0, 1], [1.002, 1.002], color=BLUE, lw=0.75, clip_on=False)

    ax.plot([trunk_x, trunk_x], [0.012, 0.988], color=BLUE_DARK, lw=0.85, zorder=2)
    ax.add_patch(Rectangle((0.000, 0.952), 0.162, 0.036, facecolor=BLUE_LIGHT,
                           edgecolor="none", zorder=1))
    ax.text(0.010, 0.970, "zss/", fontsize=6.5, weight="bold", color=NAVY, va="center")
    ax.text(0.070, 0.970, "repository root", fontsize=5.0, color=GRAY, va="center")

    y_top = 0.935
    for index, (row, height) in enumerate(zip(V4.ROWS, ROW_HEIGHTS)):
        y_bottom = y_top - height
        center = (y_top + y_bottom) / 2
        shade = BLUE_PALE if index % 2 else WHITE
        ax.add_patch(Rectangle((0.000, y_bottom), 1.000, height,
                               facecolor=shade, edgecolor="none", zorder=0))
        ax.plot([0, 1], [y_top, y_top], color=LINE, lw=0.55, zorder=3)
        ax.plot([trunk_x, 0.060], [center, center], color=BLUE, lw=0.75, zorder=3)
        ax.scatter([trunk_x], [center], s=9, color=BLUE, edgecolors=WHITE,
                   linewidths=0.45, zorder=4)

        ax.text(0.064, y_top - 0.022, f"S{row['stage']}", fontsize=5.45,
                weight="bold", color=BLUE, va="top")
        ax.text(0.098, y_top - 0.022, row["title"], fontsize=5.25,
                weight="bold", color=NAVY, va="top")

        for x in [0.190, 0.455, 0.700]:
            ax.plot([x, x], [y_bottom + 0.012, y_top - 0.012], color=LINE, lw=0.50)

        agents = compact_lines(row["stage"], "agents", row["agents"])
        engines = compact_lines(row["stage"], "engines", row["engines"])
        data = compact_lines(row["stage"], "data", row["data"])
        body_y = y_top - 0.048
        ax.text(agent_x, body_y, "\n".join(agents), fontsize=5.0, color=NAVY,
                va="top", linespacing=1.03)
        ax.text(engine_x, body_y, "\n".join(engines), fontsize=5.0, color=NAVY,
                va="top", linespacing=1.03)
        ax.text(data_x, body_y, "\n".join(data), fontsize=5.0, color=NAVY,
                va="top", linespacing=1.03)

        short_arrow(ax, 0.445, 0.462, center)
        short_arrow(ax, 0.690, 0.707, center)
        if row["stage"] not in {"0", "6"}:
            ax.text(0.985, y_bottom + 0.012, f"VERIFIED HANDOFF  {row['count']}",
                    fontsize=5.0, color=GREEN, weight="bold", ha="right", va="bottom")
        y_top = y_bottom - ROW_GAP

    ax.plot([0, 1], [0.002, 0.002], color=LINE, lw=0.55)


def draw_callout(fig):
    ax = fig.add_axes([0.070, 0.045, 0.860, 0.044])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.002, 0.002], [0.12, 0.88], color=ORANGE, lw=2.0)
    ax.text(0.022, 0.69, "Repository invariant", fontsize=5.2, weight="bold",
            color=NAVY, va="center")
    ax.text(
        0.022, 0.34,
        "Automatic HPC evaluation and manual prediction import remain separate until the common scientific evaluator; figures and tests never feed values back into scoring.",
        fontsize=5.0, color=NAVY, va="center",
    )


def write_source_data(state):
    with (OUT / "source_data_ppt_style_tree.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "stage_title", "agent_contracts", "execution", "persisted_evidence", "verified_handoff"])
        for row in V4.ROWS:
            writer.writerow([
                row["stage"], row["title"], " | ".join(compact_lines(row["stage"], "agents", row["agents"])),
                " | ".join(compact_lines(row["stage"], "engines", row["engines"])),
                " | ".join(compact_lines(row["stage"], "data", row["data"])), row["count"],
            ])
    with (OUT / "source_data_ppt_style_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ("literature_records", state["evidence_papers"]),
            ("recommended_models", state["priority_models"]),
            ("test_datasets", state["dataset_count"]),
            ("decision_rounds", state["round_files"]),
            ("sequences", state["sequence_count"]),
            ("usable_models", state["evaluated_models"]),
            ("eligible_metrics", state["eligible_metrics"]),
            ("weight_rows", state["weight_rows"]),
            ("score_rows", state["score_rows"]),
        ])


def write_qa(state):
    qa = {
        "backend": "Python/matplotlib",
        "final_canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "archetype": "A4 portrait supplementary-methods tree",
        "style_reference": "user-provided eight-page A4 portrait supplementary-methods deck",
        "reuse_level": "style-only inheritance; project structure and evidence remain native to the AMP benchmark",
        "style_tokens": {
            "background": WHITE,
            "title": NAVY,
            "section_rule": BLUE,
            "verified_output": GREEN,
            "branch_callout": ORANGE,
            "table_header": BLUE_LIGHT,
        },
        "preserved_stages": [row["stage"] for row in V4.ROWS],
        "automatic_and_manual_branches_preserved": True,
        "counts_are_direct_artifact_totals": True,
        "displayed_metrics": {
            "literature_records": state["evidence_papers"],
            "recommended_models": state["priority_models"],
            "test_datasets": state["dataset_count"],
            "decision_rounds": state["round_files"],
        },
        "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi LZW"],
    }
    (OUT / "qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    V4.verify_tree_source()
    state = V4.V1.load_real_state()
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    fig.text(0.070, 0.982, "SUPPLEMENTARY METHODS · FILE-BACKED MULTI-AGENT BENCHMARK",
             fontsize=5.2, color=BLUE_DARK, va="top")
    fig.text(0.070, 0.958, "Repository logic tree and auditable evidence handoffs",
             fontsize=11.0, weight="bold", color=NAVY, va="top")
    rule_ax = fig.add_axes([0.070, 0.9235, 0.860, 0.003])
    rule_ax.axhline(0.5, color=BLUE, lw=1.15)
    rule_ax.axis("off")
    fig.text(
        0.070, 0.902,
        "Seven ordered stages connect versioned Agent contracts, explicit execution files and durable intermediate results without allowing presentation artifacts to alter model scoring.",
        fontsize=5.5, color=NAVY, va="top",
    )

    draw_metric_band(fig, state)
    draw_tree(fig, state)
    draw_callout(fig)
    fig.text(0.070, 0.018,
             "PROJECT_LOGIC_TREE_FOR_FIGURE.md · source_data_ppt_style_tree.csv · qa_notes.json",
             fontsize=5.0, color=GRAY, va="bottom")
    fig.text(0.930, 0.018, "01 / 01", fontsize=5.0, color=GRAY, ha="right", va="bottom")

    outputs = {
        "png": OUT / "amp_project_logic_tree_ppt_style_v5.png",
        "pdf": OUT / "amp_project_logic_tree_ppt_style_v5.pdf",
        "svg": OUT / "amp_project_logic_tree_ppt_style_v5.svg",
        "tiff": OUT / "amp_project_logic_tree_ppt_style_v5.tiff",
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    write_source_data(state)
    write_qa(state)
    print("\n".join(str(path) for path in outputs.values()))


if __name__ == "__main__":
    main()
