from __future__ import annotations

import csv
import importlib.util
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MEETING = DATA / "results_manual" / "codex_agent_weight_meeting"

V1_SCRIPT = (
    ROOT / "figures" / "amp_project_file_agent_supplementary_v1"
    / "build_amp_project_file_agent_supplementary_v1.py"
)
SPEC = importlib.util.spec_from_file_location("amp_project_v1", V1_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import real-state verifier: {V1_SCRIPT}")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

WIDTH_MM, HEIGHT_MM = 183.0, 205.0
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4

INK = "#25313D"
MUTED = "#6A7787"
LINE = "#D6DEE6"
WHITE = "#FFFFFF"
BLUE = "#347FC4"
TEAL = "#258E8A"
GREEN = "#42956A"
RED = "#D35E5E"
PURPLE = "#7655B3"
GOLD = "#BE862A"
LIGHT_BLUE = "#EEF5FC"
LIGHT_GREEN = "#EFF8F2"
LIGHT_PURPLE = "#F5F1FA"
LIGHT_GOLD = "#FCF7EC"
LIGHT_GRAY = "#F4F6F8"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 5.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})


def rounded(ax, x, y, w, h, fc=WHITE, ec=LINE, lw=0.8, radius=0.014, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, lw=0.72, dashed=False, rad=0.0, z=5):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7,
        linewidth=lw, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(fig, rect, label, title, subtitle, edge, face=WHITE, title_x=0.068):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, 0.005, 0.005, 0.990, 0.990, fc=face, ec=edge, lw=1.0, radius=0.020)
    ax.text(0.025, 0.965, label, fontsize=8.0, weight="bold", color=INK, va="top")
    ax.text(title_x, 0.965, title, fontsize=7.4, weight="bold", color=INK, va="top")
    ax.text(title_x, 0.916, subtitle, fontsize=5.0, color=MUTED, va="top")
    return ax


def verify_agent_files():
    paths = [row[3] for row in AGENT_ROWS if row[3]]
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Agent definitions missing: {missing}")


AGENT_ROWS = [
    ("Evidence", "Evidence compressor", "traceable evidence compression", "agents/deepseek_meeting/evidence_compressor_agent.md"),
    ("Evidence", "Information extractor", "structured paper metadata", "agents/deepseek_meeting/info_extractor_agent.md"),
    ("Evidence", "Model/dataset scout", "candidate model and dataset proposal", "agents/deepseek_meeting/model_dataset_agent.md"),
    ("Evidence", "Metrics specialist", "endpoint and metric protocol", "agents/deepseek_meeting/metric_agent.md"),
    ("Evidence", "Independent critic", "challenge unsupported evidence", "agents/deepseek_meeting/critic_agent.md"),
    ("Evidence", "Chief decision", "reconcile and freeze memory", "agents/deepseek_meeting/chief_agent.md"),
    ("Execute", "Repository inspector", "README-to-registry contract", "agents/model_onboarding/repository_inspector_system.md"),
    ("Execute", "Self-heal agent", "bounded deployment repair", "agents/model_execution/self_heal_system.md"),
    ("Execute", "Benchmark architect", "model invocation and schema", "agents/runtime_prompts/benchmark_architect.md"),
    ("Execute", "Coder", "generate executable evaluator", "agents/runtime_prompts/coder.md"),
    ("Execute", "Evaluation critic", "audit code and outputs", "agents/runtime_prompts/critic.md"),
    ("Execute", "Principal investigator", "approve or request revision", "agents/runtime_prompts/pi.md"),
    ("Decide", "Literature expert", "evidence relevance proposal", "agents/weight_meeting/literature_agent.md"),
    ("Decide", "Statistics expert", "validity and uncertainty proposal", "agents/weight_meeting/statistics_agent.md"),
    ("Decide", "Screening expert", "operational robustness proposal", "agents/weight_meeting/screening_agent.md"),
    ("Decide", "Independent reviewer", "blind proposal audit", "agents/weight_meeting/reviewer_agent.md"),
    ("Decide", "Chief decision", "accept bounded weight vector", "agents/weight_meeting/chief_agent.md"),
    ("Decide", "Research advisor", "Top-3 and ensemble synthesis", "agents/weight_meeting/research_advisor.md"),
]


def short_lines(lines, width=34):
    result = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        result.extend(wrapped or [line])
    return result


def file_box(ax, x, y, w, h, heading, lines, color, fill, heading_size=5.25, body_size=5.0):
    rounded(ax, x, y, w, h, fc=fill, ec=color, lw=0.78, radius=0.012)
    ax.text(x + 0.012, y + h - 0.025, heading, fontsize=heading_size,
            weight="bold", color=color, va="top")
    ax.text(x + 0.012, y + h - 0.061, "\n".join(lines), fontsize=body_size,
            color=INK, va="top", linespacing=1.03)


def draw_repository_spine(ax):
    rounded(ax, 0.030, 0.795, 0.940, 0.105, fc=LIGHT_GRAY, ec=LINE, lw=0.75, radius=0.014)
    nodes = [
        (0.047, 0.142, "zss/ root", "entry + configuration", BLUE),
        (0.225, 0.158, "agents/", "Markdown role contracts", TEAL),
        (0.420, 0.178, "*.py engines", "orchestration + execution", GOLD),
        (0.635, 0.145, "data/", "persisted state", PURPLE),
        (0.817, 0.135, "figures/ + tests/", "reporting + QA", GREEN),
    ]
    for i, (x, w, title, subtitle, color) in enumerate(nodes):
        rounded(ax, x, 0.817, w, 0.058, fc=WHITE, ec=color, lw=0.75, radius=0.011)
        ax.text(x + 0.010, 0.853, title, fontsize=5.05, weight="bold", color=color, va="center")
        ax.text(x + 0.010, 0.831, subtitle, fontsize=5.0, color=MUTED, va="center")
        if i < len(nodes) - 1:
            arrow(ax, (x + w + 0.005, 0.846), (nodes[i + 1][0] - 0.006, 0.846), lw=0.65)


def draw_panel_a(fig):
    ax = panel(
        fig, [0.025, 0.470, 0.950, 0.475], "a",
        "Repository-ordered execution graph",
        "folder hierarchy is read left-to-right; each stage maps Agent definitions to Python engines and persisted files",
        BLUE, "#FCFDFF",
    )
    draw_repository_spine(ax)

    columns = [0.030, 0.125, 0.388, 0.645]
    widths = [0.078, 0.242, 0.236, 0.325]
    headers = ["STAGE", "agents/ ROLE DEFINITIONS", "ROOT PYTHON ENGINES", "data/ PERSISTED HANDOFF"]
    for x, title in zip(columns, headers):
        ax.text(x + 0.004, 0.758, title, fontsize=5.0, color=MUTED, weight="bold")

    rows = [
        {
            "y": 0.520, "color": BLUE, "fill": LIGHT_BLUE, "stage": "1\nEVIDENCE",
            "agents": ["shared/*.md + deepseek_meeting/*.md", "compressor | extractor | scout | metrics", "critic -> chief -> frozen memory"],
            "engines": ["deep_research_literature_agent.py", "dataset_recommendation_agent.py", "benchmark_portfolio.py"],
            "data": ["evidence_pool.json -> compact_evidence_pool.json", "literature_deep_research_memory.json / .md", "exports/.../recommended_{models,datasets,metrics}.csv"],
        },
        {
            "y": 0.285, "color": GREEN, "fill": LIGHT_GREEN, "stage": "2\nEXECUTE",
            "agents": ["model_onboarding/*.md", "model_execution/*.md", "runtime_prompts/{architect,coder,critic,pi}.md"],
            "engines": ["new_model_onboarding.py -> hpc_model_ops.py", "main.py -> run_meeting.py -> workflow_utils.py", "scientific_evaluation.py | run_manifest.py"],
            "data": ["local_registry.json -> data/runs/{run_id}/manifest.json", "results/{dataset}/final_results_with_predictions.csv", "eval_result.json + scientific_evaluation.json + critic.md"],
        },
        {
            "y": 0.050, "color": PURPLE, "fill": LIGHT_PURPLE, "stage": "3\nDECIDE",
            "agents": ["weight_meeting/shared_system.md", "literature | statistics | screening", "reviewer -> chief -> research advisor"],
            "engines": ["codex_agent_weight_meeting.py", "llm_agent_weight_meeting.py", "iterative_weight_meeting.py", "ensemble_top3_selector.py"],
            "data": ["agent_evidence_bundle.json -> *_agent_proposals.json", "reviewer_agent_audit.json -> chief_initial_decision.json", "rounds/round_001-050.json -> weights/scores/ranking CSV"],
        },
    ]

    for row in rows:
        y, color, fill = row["y"], row["color"], row["fill"]
        h = 0.185
        rounded(ax, columns[0], y, widths[0], h, fc=color, ec=color, lw=0.8, radius=0.012)
        ax.text(columns[0] + widths[0] / 2, y + h / 2, row["stage"], fontsize=5.2,
                color=WHITE, weight="bold", ha="center", va="center")
        file_box(ax, columns[1], y, widths[1], h, "Markdown contracts", row["agents"], color, fill)
        file_box(ax, columns[2], y, widths[2], h, "Execution", row["engines"], color, fill)
        file_box(ax, columns[3], y, widths[3], h, "Intermediate -> output", row["data"], color, fill)
        for i in range(3):
            arrow(ax, (columns[i] + widths[i] + 0.003, y + h / 2),
                  (columns[i + 1] - 0.005, y + h / 2), color=color, lw=0.70)

    arrow(ax, (0.980, 0.520), (0.980, 0.476), color=MUTED, lw=0.65, dashed=True)
    arrow(ax, (0.980, 0.285), (0.980, 0.241), color=MUTED, lw=0.65, dashed=True)
    ax.text(0.989, 0.405, "persisted handoff", fontsize=5.0, color=MUTED,
            rotation=90, ha="center", va="center")


def role_card(ax, x, y, w, h, role, filename, color, fill):
    rounded(ax, x, y, w, h, fc=fill, ec=color, lw=0.72, radius=0.010)
    ax.text(x + w / 2, y + h - 0.022, role, fontsize=5.0, weight="bold",
            color=color, ha="center", va="top")
    ax.text(x + w / 2, y + 0.018, filename, fontsize=5.0, color=INK,
            ha="center", va="bottom", linespacing=0.98)


def role_lane(ax, y, stage, color, fill, roles, filenames, relation):
    rounded(ax, 0.030, y, 0.128, 0.152, fc=color, ec=color, lw=0.8, radius=0.014)
    ax.text(0.094, y + 0.095, stage, fontsize=5.05, color=WHITE,
            weight="bold", ha="center", va="center")
    ax.text(0.094, y + 0.055, "Agent files", fontsize=5.0, color=WHITE,
            ha="center", va="center")
    x0, x1 = 0.185, 0.965
    gap = 0.010
    w = (x1 - x0 - gap * (len(roles) - 1)) / len(roles)
    for i, (role, filename) in enumerate(zip(roles, filenames)):
        x = x0 + i * (w + gap)
        role_card(ax, x, y + 0.035, w, 0.115, role, filename, color, fill)
        if i < len(roles) - 1:
            arrow(ax, (x + w + 0.002, y + 0.092),
                  (x + w + gap - 0.002, y + 0.092), color=color, lw=0.58)
    ax.text(0.575, y + 0.012, relation, fontsize=5.0, color=color,
            weight="bold", ha="center", va="bottom")


def draw_panel_b(fig):
    ax = panel(
        fig, [0.025, 0.045, 0.585, 0.405], "b",
        "Agent definitions and collaboration",
        "every role is a versionable Markdown contract loaded at runtime; filenames are shown below functional roles",
        TEAL, "#FCFEFE",
    )
    role_lane(
        ax, 0.655, "STAGE 1", BLUE, LIGHT_BLUE,
        ["Compress", "Extract", "Scout", "Metrics", "Critic", "Chief"],
        ["evidence_\ncompressor_\nagent.md", "info_\nextractor_\nagent.md", "model_\ndataset_\nagent.md",
         "metric_agent.md", "critic_agent.md", "chief_agent.md"],
        "retrieve -> structure -> propose -> challenge -> reconcile -> write literature memory",
    )
    role_lane(
        ax, 0.395, "STAGE 2", GREEN, LIGHT_GREEN,
        ["Inspect", "Architect", "Coder", "Self-heal", "Critic", "PI"],
        ["repository_\ninspector_\nsystem.md", "benchmark_\narchitect.md", "coder.md",
         "self_heal_\nsystem.md", "critic.md", "pi.md"],
        "README -> registry/schema -> code -> bounded repair -> audit -> approved evaluation bundle",
    )
    role_lane(
        ax, 0.135, "STAGE 3", PURPLE, LIGHT_PURPLE,
        ["Literature", "Statistics", "Screening", "Reviewer", "Chief", "Advisor"],
        ["literature_\nagent.md", "statistics_\nagent.md", "screening_\nagent.md",
         "reviewer_\nagent.md", "chief_\nagent.md", "research_\nadvisor.md"],
        "three blind proposals -> independent audit -> bounded consensus -> Top-3/ensemble report",
    )


def ledger_card(ax, y, h, color, fill, title, lines, footer):
    rounded(ax, 0.045, y, 0.910, h, fc=fill, ec=color, lw=0.82, radius=0.014)
    ax.text(0.065, y + h - 0.027, title, fontsize=5.25, weight="bold", color=color, va="top")
    ax.text(0.065, y + h - 0.067, "\n".join(lines), fontsize=5.0,
            color=INK, va="top", linespacing=1.06)
    ax.text(0.065, y + 0.020, footer, fontsize=5.0, weight="bold", color=color, va="bottom")


def draw_panel_c(fig, state):
    ax = panel(
        fig, [0.625, 0.045, 0.350, 0.405], "c",
        "Persisted intermediate-result ledger",
        "real artifact counts and terminal handoffs from the current project state",
        GOLD, "#FFFEFB", title_x=0.098,
    )
    ledger_card(
        ax, 0.660, 0.230, BLUE, LIGHT_BLUE,
        "STAGE 1 | evidence -> frozen recommendations",
        [
            f"evidence_pool.json: {state['evidence_papers']:,} papers / {state['evidence_batches']} batches",
            f"compact_evidence_pool.json: {state['compact_papers']:,} papers / {state['compact_chunks']} chunks",
            f"screening decisions: {state['screened_identities']} identities / {state['meeting_accepted']} accepted",
            f"exports: {state['priority_models']} models / 3 datasets / 44 metric rows",
        ],
        "handoff: memory.json/.md + three recommendation CSV files",
    )
    ledger_card(
        ax, 0.385, 0.245, GREEN, LIGHT_GREEN,
        "STAGE 2 | registered models -> evaluation bundles",
        [
            "local_registry.json: repository + environment + command + smoke status",
            f"3 datasets / {state['sequence_count']:,} sequences / {state['evaluated_models']} usable models",
            "manual branch: raw prediction CSV -> standardized prediction table",
            "per dataset: eval_result + scientific_evaluation + Critic report",
        ],
        "handoff: anonymous 3-dataset x 18-model metric evidence",
    )
    ledger_card(
        ax, 0.135, 0.220, PURPLE, LIGHT_PURPLE,
        "STAGE 3 | proposals -> 50-round decision state",
        [
            "3 expert proposal files -> reviewer_agent_audit.json",
            "chief_initial_decision.json -> rounds/round_001 ... round_050.json",
            f"{state['eligible_metrics']} metrics / {state['weight_rows']} weight rows / {state['score_rows']} score rows",
        ],
        "handoff: ranking CSV + bubble/box plots + future-directions report",
    )
    rounded(ax, 0.045, 0.025, 0.910, 0.080, fc=LIGHT_GRAY, ec=INK, lw=0.78, radius=0.014)
    top3 = "  |  ".join(f"{row['rank']} {row['model']}" for row in state["top3"])
    ax.text(0.065, 0.078, "CURRENT TERMINAL TOP-3", fontsize=5.0, color=INK, weight="bold", va="center")
    ax.text(0.935, 0.050, top3, fontsize=5.0, color=PURPLE, weight="bold", ha="right", va="center")


def write_source_tables(state):
    with (OUT / "source_data_agent_definitions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "agent_role", "responsibility", "markdown_definition"])
        writer.writerows(AGENT_ROWS)

    handoffs = [
        ("Stage 1", "agents/deepseek_meeting/*.md", "deep_research_literature_agent.py", "evidence_pool.json", "compact_evidence_pool.json"),
        ("Stage 1", "agents/deepseek_meeting/*.md", "dataset_recommendation_agent.py", "literature_deep_research_memory.json", "recommended_models/datasets/metrics.csv"),
        ("Stage 2", "agents/model_onboarding/*.md", "new_model_onboarding.py", "README + repository", "local_registry.json"),
        ("Stage 2", "agents/model_execution/*.md", "hpc_model_ops.py", "registry + HPC log", "smoke-test state"),
        ("Stage 2", "agents/runtime_prompts/*.md", "main.py + run_meeting.py", "FASTA + registry", "predictions.csv"),
        ("Stage 2", "agents/runtime_prompts/*.md", "scientific_evaluation.py", "prediction table", "eval_result.json + scientific_evaluation.json"),
        ("Stage 3", "agents/weight_meeting/*.md", "codex_agent_weight_meeting.py", "agent_evidence_bundle.json", "round_001-050.json"),
        ("Stage 3", "agents/weight_meeting/*.md", "iterative_weight_meeting.py", "accepted weights", "model scores + final ranking"),
        ("Stage 3", "agents/weight_meeting/research_advisor.md", "amp_research_advisor.py", "ranking + critics", "future-directions report"),
    ]
    with (OUT / "source_data_folder_handoffs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "agent_folder", "execution_engine", "input_artifact", "persisted_output"])
        writer.writerows(handoffs)

    artifacts = [
        ("Stage 1", "evidence_pool.json", state["evidence_papers"], "papers"),
        ("Stage 1", "evidence_pool.json", state["evidence_batches"], "evidence batches"),
        ("Stage 1", "compact_evidence_pool.json", state["compact_papers"], "papers"),
        ("Stage 1", "compact_evidence_pool.json", state["compact_chunks"], "chunks"),
        ("Stage 1", "literature_meeting_screening_decisions.csv", state["screened_identities"], "identities"),
        ("Stage 1", "recommended_models.csv", state["priority_models"], "models"),
        ("Stage 2", "evaluated datasets", state["dataset_count"], "datasets"),
        ("Stage 2", "evaluated sequences", state["sequence_count"], "sequences"),
        ("Stage 2", "usable model outputs", state["evaluated_models"], "models"),
        ("Stage 3", "eligible metrics", state["eligible_metrics"], "metrics"),
        ("Stage 3", "round JSON", state["round_files"], "rounds"),
        ("Stage 3", "metric-weight CSV", state["weight_rows"], "rows"),
        ("Stage 3", "model-score CSV", state["score_rows"], "rows"),
    ]
    with (OUT / "source_data_intermediate_artifacts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "artifact", "value", "unit"])
        writer.writerows(artifacts)


def write_qa(state):
    qa = {
        "backend": "Python/matplotlib",
        "final_canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "archetype": "schematic-led composite",
        "core_conclusion": "Versioned Markdown Agent contracts are invoked by explicit Python engines and every stage writes named, auditable intermediate artifacts.",
        "reuse_level": "structural adaptation of the prior repository-trace figure; all content was remapped to the current request",
        "all_agent_paths_verified": True,
        "all_source_rows_used_for_counts": True,
        "displayed_top3": [row["model"] for row in state["top3"]],
        "statistics": "No inferential statistics are plotted; numbers are direct artifact counts.",
        "interpretation_boundary": "The terminal ranking is the current stored 50-round result and remains exploratory pending formal provenance, independence and homology gates.",
        "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi LZW"],
    }
    (OUT / "qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    verify_agent_files()
    state = V1.load_real_state()
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    fig.text(0.025, 0.988, "Repository-ordered, file-backed multi-Agent workflow for AMP benchmarking",
             fontsize=9.6, weight="bold", color=INK, va="top")
    fig.text(0.025, 0.963,
             "real folder hierarchy | versioned Agent definitions | explicit execution engines | persisted intermediate results",
             fontsize=5.2, color=MUTED, va="top")
    rule_ax = fig.add_axes([0.025, 0.9465, 0.950, 0.003])
    rule_ax.axhline(0.5, color=INK, lw=0.8)
    rule_ax.axis("off")

    draw_panel_a(fig)
    draw_panel_b(fig)
    draw_panel_c(fig, state)

    outputs = {
        "png": OUT / "amp_project_folder_agent_trace_v3.png",
        "pdf": OUT / "amp_project_folder_agent_trace_v3.pdf",
        "svg": OUT / "amp_project_folder_agent_trace_v3.svg",
        "tiff": OUT / "amp_project_folder_agent_trace_v3.tiff",
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    write_source_tables(state)
    write_qa(state)
    print("\n".join(str(path) for path in outputs.values()))


if __name__ == "__main__":
    main()
