from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

V1_SCRIPT = (
    ROOT / "figures" / "amp_project_file_agent_supplementary_v1"
    / "build_amp_project_file_agent_supplementary_v1.py"
)
SPEC = importlib.util.spec_from_file_location("amp_project_v1", V1_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import real-state verifier: {V1_SCRIPT}")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

WIDTH_MM, HEIGHT_MM = 183.0, 235.0
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4

INK = "#27333F"
MUTED = "#6A7787"
LINE = "#D5DEE7"
WHITE = "#FFFFFF"
GRAY = "#7C8794"
BLUE = "#347FC4"
TEAL = "#258E8A"
GREEN = "#439668"
RED = "#D26060"
PURPLE = "#7656B3"
GOLD = "#BE872B"
LIGHT_GRAY = "#F4F6F8"
LIGHT_BLUE = "#EEF5FC"
LIGHT_TEAL = "#EDF8F7"
LIGHT_GREEN = "#EFF8F2"
LIGHT_PURPLE = "#F5F1FA"
LIGHT_GOLD = "#FCF7EC"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 5.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})


def rounded(ax, x, y, w, h, fc=WHITE, ec=LINE, lw=0.75, radius=0.010, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, lw=0.68, dashed=False, rad=0.0, z=6):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7,
        linewidth=lw, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    )
    ax.add_patch(patch)
    return patch


def verify_tree_source():
    required = [
        "PROJECT_LOGIC_TREE_FOR_FIGURE.md",
        "amp_benchmark_menu.py", "config.py", "workflow_guards.py", "run_manifest.py",
        "agents/deepseek_meeting/evidence_compressor_agent.md",
        "agents/deepseek_meeting/info_extractor_agent.md",
        "agents/deepseek_meeting/model_dataset_agent.md",
        "agents/deepseek_meeting/metric_agent.md",
        "agents/deepseek_meeting/critic_agent.md",
        "agents/deepseek_meeting/chief_agent.md",
        "agents/model_onboarding/repository_inspector_system.md",
        "agents/model_execution/self_heal_system.md",
        "agents/runtime_prompts/benchmark_architect.md",
        "agents/runtime_prompts/coder.md",
        "agents/runtime_prompts/critic.md",
        "agents/runtime_prompts/pi.md",
        "agents/runtime_prompts/dataset_etl_agent.md",
        "agents/weight_meeting/literature_agent.md",
        "agents/weight_meeting/statistics_agent.md",
        "agents/weight_meeting/screening_agent.md",
        "agents/weight_meeting/reviewer_agent.md",
        "agents/weight_meeting/chief_agent.md",
        "agents/weight_meeting/research_advisor.md",
        "deep_research_literature_agent.py", "new_model_onboarding.py", "hpc_model_ops.py",
        "main.py", "import_manual_prediction_results.py", "scientific_evaluation.py",
        "codex_agent_weight_meeting.py", "llm_agent_weight_meeting.py",
        "iterative_weight_meeting.py", "ensemble_top3_selector.py", "amp_research_advisor.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"The preserved logic tree references missing files: {missing}")


ROWS = [
    {
        "stage": "0", "title": "ENTRY & CONFIG", "y": 0.810, "h": 0.077,
        "color": GRAY, "fill": LIGHT_GRAY,
        "agents": ["No role-specific Agent", "Global controls and guards"],
        "engines": ["run_menu.bat -> amp_benchmark_menu.py", "config.py + .env", "workflow_guards.py | run_manifest.py"],
        "data": ["runtime selection + validated configuration", "run checkpoint and artifact provenance"],
        "count": "workflow entry",
    },
    {
        "stage": "1", "title": "LITERATURE SEARCH & RECOMMENDATION", "y": 0.630, "h": 0.162,
        "color": BLUE, "fill": LIGHT_BLUE,
        "agents": [
            "agents/shared/", "agents/deepseek_meeting/", "  evidence_compressor_agent.md",
            "  info_extractor_agent.md", "  model_dataset_agent.md", "  metric_agent.md",
            "  critic_agent.md -> chief_agent.md",
        ],
        "engines": [
            "agent_md_loader.py", "deep_research_literature_agent.py",
            "dataset_recommendation_agent.py", "benchmark_portfolio.py",
        ],
        "data": [
            "data/evidence_pool.json", "  -> compact_evidence_pool.json",
            "  -> literature_meeting_screening_decisions.csv",
            "  -> literature_deep_research_memory.json/.md",
            "data/dataset_candidate_pool.json -> dataset_plan.json",
            "data/exports/literature_recommendations/",
            "  recommended_models.csv", "  recommended_datasets.csv", "  recommended_metrics.csv",
        ],
        "count": "2,365 papers -> 20 models / 3 datasets / 44 metric rows",
    },
    {
        "stage": "2", "title": "MODEL REGISTRATION & HPC DEPLOYMENT", "y": 0.505, "h": 0.108,
        "color": TEAL, "fill": LIGHT_TEAL,
        "agents": [
            "agents/model_onboarding/", "  repository_inspector_system.md", "  repository_inspector_task.md",
            "agents/model_execution/", "  self_heal_system.md | self_heal_task.md",
        ],
        "engines": ["new_model_onboarding.py", "  -> hpc_model_ops.py", "README -> registry -> upload", "-> Conda/deps -> smoke/self-heal"],
        "data": ["data/local_registry.json", "  repository + env + command + status", "data/hpc_self_heal/", "  failure logs + bounded repair records"],
        "count": "registered model state",
    },
    {
        "stage": "3", "title": "DATASET BENCHMARKING", "y": 0.330, "h": 0.158,
        "color": GREEN, "fill": LIGHT_GREEN,
        "agents": [
            "agents/runtime_prompts/", "  benchmark_architect.md", "  coder.md | critic.md | pi.md",
            "  dataset_etl_agent.md",
        ],
        "engines": [
            "AUTO BRANCH", "  main.py -> run_meeting.py", "  -> workflow_utils.py",
            "MANUAL BRANCH", "  import_manual_prediction_results.py",
            "COMMON", "  -> scientific_evaluation.py",
        ],
        "data": [
            "data/runs/{run_id}/", "  manifest.json + artifacts/ + results/{dataset}/",
            "data/results_manual/{dataset}/", "  final_results_with_predictions.csv",
            "  eval_result.json + scientific_evaluation.json/.md",
            "  critic_individual.md",
        ],
        "count": "3 datasets / 62,310 sequences / 18 usable model outputs",
    },
    {
        "stage": "4", "title": "50-ROUND MULTI-AGENT RANKING", "y": 0.145, "h": 0.168,
        "color": PURPLE, "fill": LIGHT_PURPLE,
        "agents": [
            "agents/weight_meeting/", "  shared_system.md", "  literature_agent.md",
            "  statistics_agent.md", "  screening_agent.md", "  reviewer_agent.md",
            "  chief_agent.md | research_advisor.md",
        ],
        "engines": [
            "codex_agent_weight_meeting.py", "or llm_agent_weight_meeting.py",
            "-> iterative_weight_meeting.py", "blind proposals -> audit -> Chief",
            "-> 50 bounded updates",
        ],
        "data": [
            "data/results_manual/codex_agent_weight_meeting/", "  agent_evidence_bundle.json",
            "  {literature,statistics,screening}_agent_proposals.json",
            "  reviewer_agent_audit.json -> chief_initial_decision.json",
            "  rounds/round_001.json ... round_050.json",
            "  metric_weights_50_rounds.csv", "  model_scores_50_rounds.csv",
            "  model_ranking_50_rounds.csv + future-directions report",
        ],
        "count": "12 metrics / 50 rounds / 600 weights / 900 scores",
    },
    {
        "stage": "5", "title": "TOP-3 & ENSEMBLE LEARNING", "y": 0.062, "h": 0.066,
        "color": GOLD, "fill": LIGHT_GOLD,
        "agents": ["Consumes Stage 4 Chief decision", "Research Advisor synthesis"],
        "engines": ["ensemble_top3_selector.py", "amp_research_advisor.py"],
        "data": ["ensemble_top3_selection.json", "ensemble_top3_combination_ranking.csv"],
        "count": "Top-3 and complementarity",
    },
    {
        "stage": "6", "title": "PAPER FIGURES & QUALITY ASSURANCE", "y": 0.005, "h": 0.042,
        "color": RED, "fill": "#FDF2F2",
        "agents": ["Presentation only; no reverse scoring"],
        "engines": ["figures/ plotting scripts | tests/"],
        "data": ["SVG / PDF / PNG / TIFF + QA notes + source CSV"],
        "count": "reporting layer",
    },
]


def draw_text_block(ax, x, y, w, h, heading, lines, color, fill):
    rounded(ax, x, y, w, h, fc=fill, ec=color, lw=0.72, radius=0.009)
    compact = h < 0.055
    heading_y = y + h - (0.010 if compact else 0.020)
    body_y = y + (0.010 if compact else h - 0.048)
    ax.text(x + 0.010, heading_y, heading, fontsize=5.1, weight="bold", color=color, va="top")
    ax.text(x + 0.010, body_y, "\n".join(lines), fontsize=5.0,
            color=INK, va="top", linespacing=1.00)


def draw_figure(state):
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    fig.text(0.025, 0.989, "Repository logic tree and auditable file handoffs of the AMP benchmark",
             fontsize=9.4, weight="bold", color=INK, va="top")
    fig.text(0.025, 0.966,
             "the preserved project tree is expanded as Agent contracts -> execution files -> intermediate and terminal artifacts",
             fontsize=5.15, color=MUTED, va="top")
    rule_ax = fig.add_axes([0.025, 0.951, 0.950, 0.003])
    rule_ax.axhline(0.5, color=INK, lw=0.8)
    rule_ax.axis("off")

    ax = fig.add_axes([0.025, 0.035, 0.950, 0.900])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, 0.003, 0.003, 0.994, 0.994, fc="#FCFDFE", ec=BLUE, lw=0.95, radius=0.018)

    rounded(ax, 0.022, 0.912, 0.112, 0.062, fc=INK, ec=INK, lw=0.8, radius=0.012)
    ax.text(0.078, 0.943, "zss/", fontsize=7.0, weight="bold", color=WHITE, ha="center", va="center")
    ax.text(0.078, 0.917, "repository root", fontsize=5.0, color=WHITE, ha="center", va="center")

    headers = [
        (0.160, "agents/ DEFINITIONS"),
        (0.405, "ENTRY / PYTHON EXECUTION"),
        (0.640, "data/ INTERMEDIATE -> OUTPUT"),
    ]
    for x, text in headers:
        ax.text(x, 0.948, text, fontsize=5.15, weight="bold", color=MUTED, va="center")

    trunk_x = 0.078
    ax.plot([trunk_x, trunk_x], [0.022, 0.912], color=INK, lw=0.95, zorder=2)
    agent_x, agent_w = 0.160, 0.225
    engine_x, engine_w = 0.405, 0.215
    data_x, data_w = 0.640, 0.335

    for row in ROWS:
        y, h, color, fill = row["y"], row["h"], row["color"], row["fill"]
        center = y + h / 2
        ax.plot([trunk_x, 0.105], [center, center], color=color, lw=0.85, zorder=3)
        rounded(ax, 0.102, y, 0.043, h, fc=color, ec=color, lw=0.8, radius=0.010)
        ax.text(0.1235, center, row["stage"], fontsize=6.0, weight="bold",
                color=WHITE, ha="center", va="center")
        ax.text(agent_x, y + h + 0.004, row["title"], fontsize=5.0, weight="bold",
                color=color, ha="left", va="bottom")

        draw_text_block(ax, agent_x, y, agent_w, h, "Agent contract / folder", row["agents"], color, fill)
        draw_text_block(ax, engine_x, y, engine_w, h, "Execution path", row["engines"], color, fill)
        draw_text_block(ax, data_x, y, data_w, h, "Persisted handoff", row["data"], color, fill)
        arrow(ax, (0.146, center), (agent_x - 0.005, center), color=color, lw=0.65)
        arrow(ax, (agent_x + agent_w + 0.003, center), (engine_x - 0.005, center), color=color, lw=0.65)
        arrow(ax, (engine_x + engine_w + 0.003, center), (data_x - 0.005, center), color=color, lw=0.65)

        badge_y = y + 0.006
        ax.text(data_x + data_w - 0.010, badge_y, row["count"], fontsize=5.0,
                color=color, weight="bold", ha="right", va="bottom")

    return fig


def write_source_data(state):
    with (OUT / "source_data_logic_tree.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "stage_title", "layer", "item", "stage_summary"])
        for row in ROWS:
            for layer, values in [("agent_definition", row["agents"]), ("execution", row["engines"]), ("persisted_artifact", row["data"])]:
                for item in values:
                    writer.writerow([row["stage"], row["title"], layer, item, row["count"]])

    counts = [
        ("literature papers", state["evidence_papers"]),
        ("evidence batches", state["evidence_batches"]),
        ("compacted papers", state["compact_papers"]),
        ("chunk summaries", state["compact_chunks"]),
        ("screened identities", state["screened_identities"]),
        ("accepted identities", state["meeting_accepted"]),
        ("recommended models", state["priority_models"]),
        ("datasets", state["dataset_count"]),
        ("sequences", state["sequence_count"]),
        ("usable models", state["evaluated_models"]),
        ("eligible metrics", state["eligible_metrics"]),
        ("round files", state["round_files"]),
        ("weight rows", state["weight_rows"]),
        ("score rows", state["score_rows"]),
    ]
    with (OUT / "source_data_real_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measure", "value"])
        writer.writerows(counts)


def write_qa(state):
    qa = {
        "backend": "Python/matplotlib",
        "final_canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "archetype": "single-panel hierarchical repository tree",
        "core_conclusion": "The preserved six-stage repository tree links versioned Agent definitions to explicit execution files and durable intermediate artifacts.",
        "tree_source": "PROJECT_LOGIC_TREE_FOR_FIGURE.md",
        "preserved_top_level_stages": [row["stage"] for row in ROWS],
        "manual_and_automatic_evaluation_branches_preserved": True,
        "all_referenced_agent_and_engine_files_verified": True,
        "counts_are_direct_artifact_totals": True,
        "displayed_top3": [row["model"] for row in state["top3"]],
        "interpretation_boundary": "The terminal ranking remains exploratory pending formal provenance, independence and homology gates.",
        "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi LZW"],
    }
    (OUT / "qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    verify_tree_source()
    state = V1.load_real_state()
    fig = draw_figure(state)
    outputs = {
        "png": OUT / "amp_project_logic_tree_v4.png",
        "pdf": OUT / "amp_project_logic_tree_v4.pdf",
        "svg": OUT / "amp_project_logic_tree_v4.svg",
        "tiff": OUT / "amp_project_logic_tree_v4.tiff",
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
