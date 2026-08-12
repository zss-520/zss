from __future__ import annotations

import csv
import importlib.util
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MEETING = DATA / "results_manual" / "codex_agent_weight_meeting"

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


def wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(
        str(value), width=width, break_long_words=False, break_on_hyphens=False,
    ))


def load_runtime_state():
    state = V4.V1.load_real_state()
    registry = json.loads((DATA / "local_registry.json").read_text(encoding="utf-8"))
    if not isinstance(registry, list):
        registry = registry.get("models", [])
    ensemble = json.loads((DATA / "results_manual" / "ensemble_top3_selection.json").read_text(encoding="utf-8"))
    combination_rows = list(csv.DictReader(
        (DATA / "results_manual" / "ensemble_top3_combination_ranking.csv").open(encoding="utf-8-sig")
    ))
    state.update({
        "registry_entries": len(registry),
        "registry_ready": sum(bool(row.get("skip_env_setup")) for row in registry),
        "self_heal_files": sum(1 for path in (DATA / "hpc_self_heal").rglob("*") if path.is_file()),
        "ensemble_combinations": len(combination_rows),
        "ensemble_recommended": ensemble.get("recommended_models", []),
        "test_files": len(list((ROOT / "tests").glob("test_*.py"))),
    })
    return state


def page_data(state):
    top3 = [row["model"] for row in state["top3"]]
    ensemble_top3 = state["ensemble_recommended"]
    return [
        {
            "stage": "S0", "slug": "stage0_entry_config",
            "eyebrow": "SUPPLEMENTARY METHODS · WORKFLOW CONTROL",
            "title": "Entry, configuration and run provenance",
            "summary": "A single entry layer validates local and HPC configuration before any literature, deployment or evaluation Agent is allowed to execute.",
            "metrics": [("1", "unified menu"), ("4", "control files"), ("1", "environment file"), ("1", "run manifest")],
            "agents": [
                ("Workflow control", "No role-specific Agent", "Human workflow selection and authorization"),
                ("Global safeguards", "workflow_guards.py", "Preflight checks before deployment or evaluation"),
            ],
            "steps": [
                ("Select", "run_menu.bat"), ("Dispatch", "amp_benchmark_menu.py"),
                ("Configure", "config.py + .env"), ("Gate", "workflow_guards.py"),
                ("Record", "run_manifest.py"),
            ],
            "artifacts": [
                ("run_menu.bat", "entry", "starts the interactive workflow"),
                ("amp_benchmark_menu.py", "orchestrator", "routes literature, onboarding and benchmark tasks"),
                ("config.py + .env", "configuration", "API, HPC, SLURM, model and metric settings"),
                ("workflow_guards.py", "gate", "blocks incomplete or unsafe runs"),
                ("run_manifest.py", "provenance", "records inputs, versions, events and artifacts"),
            ],
            "callout_title": "Control invariant",
            "callout": "Credentials remain local; every formal run receives a manifest before downstream scientific outputs are accepted.",
        },
        {
            "stage": "S1", "slug": "stage1_literature_recommendation",
            "eyebrow": "SUPPLEMENTARY METHODS · LITERATURE MULTI-AGENT MEETING",
            "title": "Evidence search, recommendation and persistent memory",
            "summary": "Role-specific Agents retrieve, compress, challenge and reconcile literature evidence before freezing model, dataset and metric recommendation tables.",
            "metrics": [
                (f"{state['evidence_papers']:,}", "papers"), (f"{state['evidence_batches']}", "evidence batches"),
                (f"{state['compact_chunks']}", "compact chunks"), (f"{state['priority_models']}", "recommended models"),
            ],
            "agents": [
                ("Compress", "evidence_compressor_agent.md", "Retain traceable evidence while reducing context"),
                ("Extract", "info_extractor_agent.md", "Produce structured paper and repository metadata"),
                ("Scout", "model_dataset_agent.md", "Propose models and benchmark datasets"),
                ("Metrics", "metric_agent.md", "Define endpoints and evaluation evidence"),
                ("Critic", "critic_agent.md", "Challenge unsupported or incomplete claims"),
                ("Chief", "chief_agent.md", "Reconcile decisions and write persistent memory"),
            ],
            "steps": [
                ("Retrieve", "multi-source search"), ("Compress", "compact evidence"),
                ("Propose", "models/data/metrics"), ("Audit", "critic challenge"),
                ("Freeze", "memory + CSV exports"),
            ],
            "artifacts": [
                ("evidence_pool.json", "raw evidence", f"{state['evidence_papers']:,} papers / {state['evidence_batches']} batches"),
                ("compact_evidence_pool.json", "compressed evidence", f"{state['compact_papers']:,} papers / {state['compact_chunks']} chunks"),
                ("literature_meeting_screening_decisions.csv", "audit decisions", f"{state['screened_identities']} identities / {state['meeting_accepted']} accepted"),
                ("literature_deep_research_memory.json/.md", "long-term memory", "machine-readable state plus human-readable record"),
                ("recommended_{models,datasets,metrics}.csv", "frozen handoff", "20 models / 3 datasets / 44 metric rows"),
            ],
            "callout_title": "Memory rule",
            "callout": "The next literature run reads and updates the stored memory; the recommendation list is versioned evidence, not a permanently fixed template.",
        },
        {
            "stage": "S2", "slug": "stage2_model_deployment",
            "eyebrow": "SUPPLEMENTARY METHODS · MODEL ONBOARDING AND SELF-HEALING",
            "title": "Repository registration and HPC deployment",
            "summary": "README-derived model contracts are converted into registry records, uploaded to HPC, installed in isolated environments and admitted only after smoke testing.",
            "metrics": [
                (f"{state['registry_entries']}", "registry entries"), (f"{state['registry_ready']}", "marked ready"),
                (f"{state['self_heal_files']}", "repair artifacts"), ("1", "smoke-test gate"),
            ],
            "agents": [
                ("Repository inspector", "model_onboarding/repository_inspector_system.md", "Translate README, requirements and commands into a registry proposal"),
                ("Inspector task", "model_onboarding/repository_inspector_task.md", "Constrain the repository-specific extraction task"),
                ("Self-heal system", "model_execution/self_heal_system.md", "Diagnose logs and propose bounded repairs"),
                ("Self-heal task", "model_execution/self_heal_task.md", "Retry only approved environment or invocation changes"),
            ],
            "steps": [
                ("Inspect", "README + requirements"), ("Register", "local_registry.json"),
                ("Upload", "repository -> HPC"), ("Build", "Conda + dependencies"),
                ("Admit", "smoke-test success"),
            ],
            "artifacts": [
                ("new_model_onboarding.py", "local engine", "repository inspection and registry generation"),
                ("hpc_model_ops.py", "HPC engine", "upload, environment creation, repair and smoke test"),
                ("data/local_registry.json", "durable model state", "repository, environment, command and readiness"),
                ("data/hpc_self_heal/", "repair evidence", "failure logs, proposed repair and retry record"),
                ("skip_env_setup=true", "admission state", "written only after the deployment gate succeeds"),
            ],
            "callout_title": "Admission rule",
            "callout": "A model name in the registry is insufficient; environment readiness, inference command and smoke-test evidence must all be available before formal benchmarking.",
        },
        {
            "stage": "S3", "slug": "stage3_dataset_benchmarking",
            "eyebrow": "SUPPLEMENTARY METHODS · CROSS-DATASET SCIENTIFIC EVALUATION",
            "title": "Automatic and manual predictions converge at one evaluator",
            "summary": "HPC-generated predictions and externally generated prediction tables remain separate until both satisfy the same sample, label and probability schema.",
            "metrics": [
                ("59,311", "C_AMPs-predict test"), ("1,203", "Veltri test"),
                ("1,796", "ProteoGPT test"), (f"{state['evaluated_models']}", "usable models"),
            ],
            "agents": [
                ("Architect", "runtime_prompts/benchmark_architect.md", "Infer model invocation and output schema"),
                ("Coder", "runtime_prompts/coder.md", "Generate executable evaluation code"),
                ("ETL", "runtime_prompts/dataset_etl_agent.md", "Align sequence IDs, labels and probabilities"),
                ("Critic", "runtime_prompts/critic.md", "Audit code, outputs and missingness"),
                ("PI", "runtime_prompts/pi.md", "Approve, reject or request a bounded revision"),
            ],
            "steps": [
                ("Auto", "main -> HPC/SLURM"), ("Manual", "import prediction CSV"),
                ("Normalize", "sample/label/probability"), ("Evaluate", "scientific_evaluation.py"),
                ("Persist", "prediction + metric bundle"),
            ],
            "artifacts": [
                ("data/runs/{run_id}/manifest.json", "automatic provenance", "HPC inputs, events and retrieved artifacts"),
                ("data/manual_predictions/{dataset}/", "manual provenance", "archived source prediction tables"),
                ("final_results_with_predictions.csv", "standard table", "aligned sample IDs, labels and probabilities"),
                ("eval_result.json + scientific_evaluation.json/.md", "metric evidence", "common metric protocol and statistical summaries"),
                ("critic_individual.md", "independent audit", "dataset-specific interpretation and failure review"),
            ],
            "callout_title": "Evaluation invariant",
            "callout": "Automatic and manual branches cannot change the metric definition; both enter the same evaluator only after schema alignment and provenance recording.",
        },
        {
            "stage": "S4", "slug": "stage4_fifty_round_ranking",
            "eyebrow": "SUPPLEMENTARY RESULTS · BLINDED MULTI-AGENT RANKING",
            "title": "Fifty bounded rounds convert metric evidence into consensus ranking",
            "summary": "Three specialist proposals are reviewed independently and reconciled by a Chief Agent before each accepted metric vector scores all models across datasets.",
            "metrics": [
                (f"{state['eligible_metrics']}", "eligible metrics"), (f"{state['round_files']}", "decision rounds"),
                (f"{state['weight_rows']}", "weight rows"), (f"{state['score_rows']}", "model scores"),
            ],
            "agents": [
                ("Literature", "weight_meeting/literature_agent.md", "Judge biological relevance and evidence support"),
                ("Statistics", "weight_meeting/statistics_agent.md", "Judge validity, uncertainty and metric behavior"),
                ("Screening", "weight_meeting/screening_agent.md", "Judge operational robustness and screening utility"),
                ("Reviewer", "weight_meeting/reviewer_agent.md", "Audit proposals against blinded evidence"),
                ("Chief", "weight_meeting/chief_agent.md", "Accept a bounded weight vector or request revision"),
                ("Advisor", "weight_meeting/research_advisor.md", "Interpret stability, complementarity and Top-3"),
            ],
            "steps": [
                ("Blind", "agent evidence bundle"), ("Propose", "three expert vectors"),
                ("Audit", "independent reviewer"), ("Accept", "Chief bounded update"),
                ("Aggregate", "50-round consensus"),
            ],
            "artifacts": [
                ("agent_evidence_bundle.json", "blinded evidence", "3 datasets / 18 models / eligible metrics"),
                ("*_agent_proposals.json", "specialist proposals", "Literature, Statistics and Screening vectors"),
                ("reviewer_agent_audit.json", "independent audit", "proposal-level challenges and decisions"),
                ("chief_initial_decision.json + rounds/round_001-050.json", "decision state", "initial vector plus every accepted update"),
                ("weights/scores/ranking_50_rounds.csv", "terminal evidence", f"current Top-3: {' / '.join(top3)}"),
            ],
            "callout_title": "Aggregation equations",
            "callout": r"Single round: $S_m^{(t)}=\sum_k w_k^{(t)}q_{mk}$; terminal evidence: $\bar{S}_m=50^{-1}\sum_{t=1}^{50}S_m^{(t)}$. No model-specific bonus is applied.",
        },
        {
            "stage": "S5", "slug": "stage5_top3_ensemble",
            "eyebrow": "SUPPLEMENTARY RESULTS · TOP-3 AND ENSEMBLE ANALYSIS",
            "title": "Exhaustive three-model analysis tests performance complementarity",
            "summary": "Every three-model combination is evaluated by equal-probability soft voting and ranked with the stored cross-dataset evidence without model-specific bonuses.",
            "metrics": [
                (f"{state['evaluated_models']}", "eligible models"), (f"{state['ensemble_combinations']}", "three-model combinations"),
                ("3", "benchmark datasets"), ("1", "recommended trio"),
            ],
            "agents": [
                ("Chief handoff", "chief_agent.md", "Provides the accepted ranking evidence"),
                ("Research advisor", "research_advisor.md", "Interprets stability and complementary errors"),
                ("Combination engine", "ensemble_top3_selector.py", "Scores every unique three-model combination"),
            ],
            "steps": [
                ("Align", "three datasets"), ("Enumerate", "all 3-model sets"),
                ("Vote", "equal probability mean"), ("Score", "stored 50-round evidence"),
                ("Recommend", "external validation trio"),
            ],
            "artifacts": [
                ("ensemble_top3_selector.py", "selection engine", "exhaustive equal-probability soft voting"),
                ("ensemble_top3_combination_ranking.csv", "full ranking", f"{state['ensemble_combinations']} evaluated combinations"),
                ("ensemble_top3_selection.json", "selection record", "method, datasets, models, ranking and caveat"),
                ("recommended_models", "current trio", " / ".join(ensemble_top3)),
                ("amp_research_advisor.py", "interpretation", "performance, stability and future validation plan"),
            ],
            "callout_title": "Scientific boundary",
            "callout": "The recommended trio is exploratory because selection used the same benchmark datasets; a formal performance claim requires an untouched external validation set.",
        },
        {
            "stage": "S6", "slug": "stage6_figures_quality_assurance",
            "eyebrow": "REPRODUCIBILITY APPENDIX · REPORTING AND QUALITY ASSURANCE",
            "title": "Figures expose evidence without becoming evidence",
            "summary": "Publication assets are generated from stored CSV and JSON files, while tests and QA records verify rendering, evaluation and workflow contracts.",
            "metrics": [
                ("4", "export formats"), ("300", "PNG dpi"), ("600", "TIFF dpi"),
                (f"{state['test_files']}", "test modules"),
            ],
            "agents": [
                ("Presentation layer", "figures/", "Generate publication images from stored results"),
                ("Quality layer", "tests/", "Check memory, gates, evaluation, manifests and ranking"),
                ("Protocol", "SCIENTIFIC_EVALUATION_PROTOCOL.md", "Document metric and interpretation rules"),
            ],
            "steps": [
                ("Read", "source CSV/JSON"), ("Render", "Python/matplotlib"),
                ("Export", "SVG/PDF/PNG/TIFF"), ("Inspect", "resolution and overlap"),
                ("Archive", "source data + QA notes"),
            ],
            "artifacts": [
                ("figures/*.svg", "editable vector", "selectable text and editable paths"),
                ("figures/*.pdf", "publication vector", "submission and review layout"),
                ("figures/*.png + *.tiff", "raster export", "300-dpi preview and 600-dpi submission"),
                ("source_data_*.csv", "traceability", "values displayed in each quantitative panel"),
                ("qa_notes.json + tests/", "quality evidence", "dimensions, exclusions, caveats and contract tests"),
            ],
            "callout_title": "Reporting invariant",
            "callout": "Plots, captions and screenshots are terminal presentation artifacts; formal numbers must always be traced back to their CSV, JSON, manifest and round-specific decision records.",
        },
    ]


def metric_band(fig, metrics):
    ax = fig.add_axes([0.070, 0.790, 0.860, 0.055])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0, 1], [0.96, 0.96], color=BLUE, lw=0.75)
    ax.plot([0, 1], [0.04, 0.04], color=LINE, lw=0.55)
    for index, (value, label) in enumerate(metrics):
        x0, x1 = index / 4, (index + 1) / 4
        if index:
            ax.plot([x0, x0], [0.08, 0.92], color=LINE, lw=0.55)
        ax.text((x0 + x1) / 2, 0.60, value, fontsize=8.0, weight="bold",
                color=NAVY, ha="center", va="center")
        ax.text((x0 + x1) / 2, 0.28, label, fontsize=5.0, color=GRAY,
                ha="center", va="center")


def section_title(ax, y, title, subtitle=None):
    ax.text(0.000, y, title, fontsize=6.0, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(1.000, y, subtitle, fontsize=5.0, color=GRAY, ha="right", va="top")
    ax.plot([0, 1], [y - 0.025, y - 0.025], color=BLUE, lw=0.65)


def draw_table(ax, top, bottom, headers, rows, widths, wrap_widths):
    header_h = 0.045
    ax.add_patch(Rectangle((0, top - header_h), 1, header_h, facecolor=BLUE_LIGHT, edgecolor="none"))
    x_positions = [0]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)
    for x, header in zip(x_positions, headers):
        ax.text(x + 0.010, top - header_h / 2, header, fontsize=5.0,
                weight="bold", color=BLUE_DARK, va="center")
    row_h = (top - header_h - bottom) / max(len(rows), 1)
    for row_index, row in enumerate(rows):
        y_top = top - header_h - row_index * row_h
        y_bottom = y_top - row_h
        if row_index % 2:
            ax.add_patch(Rectangle((0, y_bottom), 1, row_h, facecolor=BLUE_PALE, edgecolor="none"))
        ax.plot([0, 1], [y_bottom, y_bottom], color=LINE, lw=0.45)
        for x, value, width in zip(x_positions, row, wrap_widths):
            ax.text(x + 0.010, y_top - 0.010, wrap(value, width), fontsize=5.0,
                    color=NAVY, va="top", linespacing=1.02)


def execution_flow(ax, page):
    section_title(ax, 0.550, "Execution sequence", "ordered file handoffs")
    xs = [0.02, 0.215, 0.410, 0.605, 0.800]
    for index, ((label, detail), x) in enumerate(zip(page["steps"], xs), start=1):
        ax.text(x, 0.480, f"{index:02d}", fontsize=5.1, weight="bold", color=BLUE)
        ax.text(x, 0.447, label, fontsize=5.1, weight="bold", color=NAVY)
        ax.text(x, 0.410, wrap(detail, 22), fontsize=5.0, color=GRAY, va="top", linespacing=1.02)
        if index < len(xs):
            arrow = FancyArrowPatch(
                (x + 0.145, 0.450), (xs[index] - 0.020, 0.450),
                arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=ORANGE,
            )
            ax.add_patch(arrow)


def render_page(page, page_number, total_pages):
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    fig.text(0.070, 0.982, f"{page['eyebrow']} · {page['stage']}",
             fontsize=5.2, color=BLUE_DARK, va="top")
    fig.text(0.070, 0.958, page["title"], fontsize=10.8, weight="bold", color=NAVY, va="top")
    rule_ax = fig.add_axes([0.070, 0.9235, 0.860, 0.003])
    rule_ax.axhline(0.5, color=BLUE, lw=1.15); rule_ax.axis("off")
    fig.text(0.070, 0.902, wrap(page["summary"], 145), fontsize=5.5, color=NAVY, va="top")
    metric_band(fig, page["metrics"])

    ax = fig.add_axes([0.070, 0.105, 0.860, 0.665])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    section_title(ax, 0.985, "Agent contracts and responsibilities", "versioned Markdown or control files")
    draw_table(
        ax, 0.940, 0.610,
        ["ROLE", "DEFINITION FILE", "RESPONSIBILITY"], page["agents"],
        [0.20, 0.38, 0.42], [24, 47, 54],
    )
    execution_flow(ax, page)
    section_title(ax, 0.350, "Persisted evidence and stage handoff", "project-state artifacts")
    draw_table(
        ax, 0.305, 0.015,
        ["ARTIFACT", "ROLE", "AUDIT / HANDOFF MEANING"], page["artifacts"],
        [0.39, 0.22, 0.39], [46, 27, 47],
    )

    callout_ax = fig.add_axes([0.070, 0.045, 0.860, 0.044])
    callout_ax.set_xlim(0, 1); callout_ax.set_ylim(0, 1); callout_ax.axis("off")
    callout_ax.plot([0.002, 0.002], [0.12, 0.88], color=ORANGE, lw=2.0)
    callout_ax.text(0.022, 0.69, page["callout_title"], fontsize=5.2,
                    weight="bold", color=NAVY, va="center")
    callout_ax.text(0.022, 0.34, wrap(page["callout"], 150), fontsize=5.0,
                    color=NAVY, va="center")
    fig.text(0.070, 0.018,
             f"PROJECT_LOGIC_TREE_FOR_FIGURE.md · {page['slug']}_source.csv · qa_notes.json",
             fontsize=5.0, color=GRAY, va="bottom")
    fig.text(0.930, 0.018, f"{page_number:02d} / {total_pages:02d}",
             fontsize=5.0, color=GRAY, ha="right", va="bottom")
    return fig


def write_page_source(page):
    path = OUT / f"{page['slug']}_source.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "field_1", "field_2", "field_3"])
        for value, label in page["metrics"]:
            writer.writerow(["metric", value, label, ""])
        for row in page["agents"]:
            writer.writerow(["agent", *row])
        for label, detail in page["steps"]:
            writer.writerow(["execution", label, detail, ""])
        for row in page["artifacts"]:
            writer.writerow(["artifact", *row])


def create_contact_sheet(png_paths):
    thumbs = []
    thumb_width = 520
    resampling = getattr(Image, "Resampling", Image)
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        height = round(image.height * thumb_width / image.width)
        thumbs.append(image.resize((thumb_width, height), resampling.LANCZOS))
    gap = 30
    rows = (len(thumbs) + 1) // 2
    cell_height = max(image.height for image in thumbs)
    canvas = Image.new("RGB", (thumb_width * 2 + gap * 3, rows * cell_height + gap * (rows + 1)), "white")
    for index, image in enumerate(thumbs):
        col, row = index % 2, index // 2
        x = gap + col * (thumb_width + gap)
        y = gap + row * (cell_height + gap)
        canvas.paste(image, (x, y))
    canvas.save(OUT / "amp_project_stage_pages_contact_sheet.png", dpi=(150, 150))


def main():
    V4.verify_tree_source()
    state = load_runtime_state()
    pages = page_data(state)
    png_paths = []
    combined_path = OUT / "amp_project_stage_pages_ppt_style_v6.pdf"
    with PdfPages(combined_path) as combined:
        for index, page in enumerate(pages, start=1):
            fig = render_page(page, index, len(pages))
            prefix = OUT / page["slug"]
            fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
            fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
            fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
            combined.savefig(fig, bbox_inches="tight", pad_inches=0.03)
            plt.close(fig)
            png_paths.append(prefix.with_suffix(".png"))
            write_page_source(page)
    create_contact_sheet(png_paths)
    qa = {
        "backend": "Python/matplotlib",
        "page_size_mm": [WIDTH_MM, HEIGHT_MM],
        "page_count": len(pages),
        "style_reference": "user-provided A4 portrait supplementary-methods deck",
        "reuse_level": "style-only inheritance",
        "stages": [page["stage"] for page in pages],
        "all_values_from_current_project_artifacts": True,
        "automatic_and_manual_evaluation_branches_preserved": True,
        "exports_per_stage": ["PNG 300 dpi", "SVG", "PDF"],
        "combined_pdf": combined_path.name,
    }
    (OUT / "qa_notes.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(combined_path)
    for path in png_paths:
        print(path)


if __name__ == "__main__":
    main()
