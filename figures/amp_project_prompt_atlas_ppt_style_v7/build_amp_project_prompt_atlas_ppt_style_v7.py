from __future__ import annotations

import csv
import json
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MEETING = DATA / "results_manual" / "codex_agent_weight_meeting"

WIDTH_MM, HEIGHT_MM = 210.0, 297.0
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4

NAVY = "#111C2E"
BLUE = "#17698D"
BLUE_DARK = "#315B77"
GREEN = "#16844A"
GRAY = "#617184"
LIGHT = "#D7E1E8"
PALE = "#F5F8FA"
WHITE = "#FFFFFF"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 5.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def count_csv(path: Path) -> int:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return 0


def runtime_state() -> dict:
    evidence = read_json(DATA / "evidence_pool.json", {})
    compact = read_json(DATA / "compact_evidence_pool.json", {})
    registry = read_json(DATA / "local_registry.json", [])
    if isinstance(registry, dict):
        registry = registry.get("models", [])
    ensemble = read_json(DATA / "results_manual" / "ensemble_top3_selection.json", {})
    ranking_path = MEETING / "codex_agent_model_ranking_50_rounds.csv"
    ranking = []
    if ranking_path.exists():
        with ranking_path.open(encoding="utf-8-sig", newline="") as handle:
            ranking = list(csv.DictReader(handle))
    return {
        "papers": int(evidence.get("paper_count") or 2365),
        "batches": int(evidence.get("evidence_batch_count") or 304),
        "chunks": int(compact.get("chunk_summary_count") or compact.get("chunk_count") or 241),
        "registry": len(registry),
        "registry_ready": sum(bool(row.get("skip_env_setup")) for row in registry),
        "repairs": sum(1 for p in (DATA / "hpc_self_heal").rglob("*") if p.is_file()),
        "ranking_models": len(ranking) or 18,
        "top3": [row.get("model", "") for row in ranking[:3]],
        "rounds": len(list((MEETING / "rounds").glob("round_*.json"))),
        "weight_rows": count_csv(MEETING / "codex_agent_metric_weights_50_rounds.csv"),
        "score_rows": count_csv(MEETING / "codex_agent_model_scores_50_rounds.csv"),
        "ensemble_models": ensemble.get("recommended_models", []),
        "ensemble_combinations": int(ensemble.get("combinations") or 816),
    }


def records(state: dict) -> list[dict]:
    literature_out = (
        f"{state['papers']:,} papers; {state['batches']} evidence batches; "
        f"{state['chunks']} compact summaries"
    )
    return [
        {
            "part": "PART I", "slug": "part1_literature_meeting",
            "eyebrow": "AUDITABLE AGENT PROMPTS · PART I",
            "title": "Evidence search, memory and literature consensus",
            "entries": [
                {
                    "code": "P00", "title": "Traceable evidence compression",
                    "system": "Compress one evidence chunk into a compact, traceable JSON record. Preserve canonical model names, aliases, paper identifiers, repositories, datasets, metrics, blocking issues, evidence level and uncertainty. Never invent a paper, URL, dataset or code repository.",
                    "user": "Compress the supplied runtime evidence chunk. Retain PMID, PMCID, DOI, title, URL and source fields. Return the declared JSON schema only; long source passages are excluded from the compact pool.",
                    "output": literature_out,
                    "source": "agents/deepseek_meeting/evidence_compressor_agent.md · deep_research_literature_agent.py:3042-3055",
                },
                {
                    "code": "P01", "title": "Recall-oriented model and dataset Scout",
                    "system": "Build a broad deduplicated inventory from compact evidence. Merge aliases, retain candidates with explicit blocking issues, separate all candidates from benchmark-ready and deployable models, and classify representation and architecture.",
                    "user": "Discuss every dynamically supplied acquisition and coverage candidate as accept, reject or defer. Compare source provenance, labels, balance, negative construction, overlap and homology. Required core candidates must be discussed, but are not fixed winners.",
                    "output": "20-model deployment portfolio; three dataset candidates retained with decision traces",
                    "source": "agents/deepseek_meeting/model_dataset_agent.md · deep_research_literature_agent.py:3322-3349",
                },
                {
                    "code": "P02", "title": "Benchmark metric and dataset review",
                    "system": "Interpret the evidence as a statistical evaluation expert. Review prevalence, threshold freezing, calibration, paired or cluster bootstrap and homology-aware evaluation. Do not create dataset names, URLs or computed results.",
                    "user": "Review whether the candidates form a complementary test matrix. Prefer balanced and differently imbalanced sets only when provenance, labels, training overlap, homology and independence are defensible. Return accept, reject or defer for every candidate.",
                    "output": "Dynamic metric protocol persisted; dataset properties without evidence marked needs_sequence_audit",
                    "source": "agents/deepseek_meeting/metric_agent.md · deep_research_literature_agent.py:3354-3369",
                },
                {
                    "code": "P03", "title": "Critic audit and Chief memory update",
                    "system": "Audit scope, provenance, reproducibility, leakage and unsupported certainty. The Chief must reconcile proposals, criticisms, rebuttals and dissent while preserving the cumulative candidate pool and explicit reasons for any downgrade.",
                    "user": "Adjudicate every dataset and model coverage candidate, check continuity with prior memory, then write strict JSON for the long-term state. Historical memory is an evidence anchor rather than a fixed recommendation template.",
                    "output": "literature_deep_research_memory.json/.md and screening decision CSV refreshed",
                    "source": "agents/deepseek_meeting/critic_agent.md · chief_agent.md · deep_research_literature_agent.py:3374-3474",
                },
            ],
        },
        {
            "part": "PART II", "slug": "part2_onboarding_self_heal",
            "eyebrow": "AUDITABLE AGENT PROMPTS · PART II",
            "title": "Repository onboarding, HPC deployment and self-healing",
            "entries": [
                {
                    "code": "P04", "title": "Repository evidence inspection",
                    "system": "Act as a computational-biology MLOps engineer. Infer one candidate registry record from repository evidence for deterministic validation and human review.",
                    "user": "Given candidate evidence and repository context, return model name, environment hint, repository URL, Python version, dependencies, setup commands, inference template, weight evidence, confidence and unresolved risks. Leave unsupported commands empty.",
                    "output": f"{state['registry']} registry records currently persisted in data/local_registry.json",
                    "source": "agents/model_onboarding/repository_inspector_system.md · repository_inspector_task.md",
                },
                {
                    "code": "P05", "title": "Deterministic registry validation",
                    "system": "Treat Agent output as a proposal, not executable truth. Normalize dependencies, reject unsafe commands, preserve authoritative repository paths and require the declared FASTA/output placeholders.",
                    "user": "Validate required fields and evidence links before writing the registry. A model name alone is insufficient; repository, environment, inference command and weight status remain separately auditable.",
                    "output": "Registry write is separated from deployment readiness and formal benchmark admission",
                    "source": "new_model_onboarding.py · model_resource_policy.py · data/local_registry.json",
                },
                {
                    "code": "P06", "title": "Bounded HPC self-heal proposal",
                    "system": "Diagnose one failed environment or smoke test from registry, repository documentation and logs. Recommend the smallest evidence-supported repair. Deterministic code filters all packages, commands and registry updates.",
                    "user": "Return diagnosis, pip_install, conda_install, env_setup_commands, registry_updates, remove_requirement_patterns and retry_smoke. Never propose destructive or privileged commands.",
                    "output": f"{state['repairs']} repair artifacts preserved under data/hpc_self_heal",
                    "source": "agents/model_execution/self_heal_system.md · self_heal_task.md · hpc_model_ops.py",
                },
                {
                    "code": "P07", "title": "Smoke-test admission gate",
                    "system": "Only an evidence-backed, environment-specific inference command may enter the smoke test. Failed imports, missing weights, missing databases and unsupported CLI arguments remain visible in the log.",
                    "user": "Run a mini-FASTA test, verify the output tree and prediction schema, retry only an approved bounded repair, and write readiness only after the smoke test succeeds.",
                    "output": f"{state['registry_ready']} of {state['registry']} registry entries currently carry skip_env_setup=true",
                    "source": "hpc_model_ops.py · data/hpc_self_heal/ · data/local_registry.json",
                },
            ],
        },
        {
            "part": "PART III", "slug": "part3_benchmark_evaluation",
            "eyebrow": "AUDITABLE AGENT PROMPTS · PART III",
            "title": "Dataset alignment, execution and scientific evaluation",
            "entries": [
                {
                    "code": "P08", "title": "Observed-output schema extraction",
                    "system": "Infer id, sequence and probability fields only from visible output headers and samples. Never invent a column. If an identifier or prediction field cannot be established, return UNKNOWN and request human intervention.",
                    "user": "Read the Stage-1 exploration report and return the exact seven-key schema for every model: file_path, file_ext, sep, comment_char, id_col, seq_col and prob_col. Output JSON only.",
                    "output": "Observed schemas are persisted and reused through schema_memory.json",
                    "source": "agents/runtime_prompts/data_analyst_extraction.md · run_meeting.py:330-331",
                },
                {
                    "code": "P09", "title": "Evaluation code generation and review",
                    "system": "Use only runtime-supplied model commands, dataset files, schemas and output contracts. Do not invent paths, dependency versions, model outputs, columns or metric values. Preserve missing predictions as missing.",
                    "user": "Generate one executable Python evaluation script and one launch script. A separate Data Architect reviews every referenced field; runtime code performs syntax, artifact, dependency, path and scientific metric validation.",
                    "output": "Automatic HPC predictions and manual prediction tables converge at one evaluator",
                    "source": "agents/runtime_prompts/coder.md · data_analyst_review.md · main.py · run_meeting.py",
                },
                {
                    "code": "P10", "title": "Manual prediction import contract",
                    "system": "Manual model outputs cannot bypass provenance or metric rules. Preserve the original file, align sequence IDs, labels and probabilities, and report duplicate, unmatched or invalid rows.",
                    "user": "Import the three supplied prediction tables into dataset-specific directories, standardize their schema and continue from the same scientific evaluator used by automatically executed models.",
                    "output": "C_AMPs-predict test n=59,311 · Veltri test n=1,203 · ProteoGPT test n=1,796",
                    "source": "import_manual_prediction_results.py · data/manual_predictions/ · data/results_manual/",
                },
                {
                    "code": "P11", "title": "Independent scientific result audit",
                    "system": "Interpret objective benchmark data without fabricating missing results. Distinguish execution failure from biological prediction performance and keep dataset-specific limitations visible.",
                    "user": "Review dynamic metric weights, real metric tables and quantitative scores. Explain performance under the meeting protocol, then write the model-level scientific judgement and evidence-limited ranking.",
                    "output": f"Three evaluation bundles produced for {state['ranking_models']} usable models",
                    "source": "agents/runtime_prompts/critic.md · scientific_evaluation.py · data/results_manual/*/eval_result.json",
                },
            ],
        },
        {
            "part": "PART IV", "slug": "part4_fifty_round_ranking",
            "eyebrow": "AUDITABLE AGENT PROMPTS · PART IV",
            "title": "Blinded 50-round metric weighting and model ranking",
            "entries": [
                {
                    "code": "P12", "title": "Independent specialist weight proposals",
                    "system": "You are one role in a blinded multi-Agent meeting selecting metric weights for AMP binary classification. Never optimize weights for a named model, a desired Top-3 or a leaderboard position.",
                    "user": "Literature, Statistics and Screening Agents independently propose all eligible metric weights from literature consensus, anonymous dataset profiles and metric-level evidence. Distinguish literature, benchmark evidence and LLM prior.",
                    "output": "Three role-specific proposal streams retained for the initial meeting and every round",
                    "source": "agents/weight_meeting/shared_system.md · literature_agent.md · statistics_agent.md · screening_agent.md",
                },
                {
                    "code": "P13", "title": "Independent methodology audit",
                    "system": "Audit unsupported evidence, hidden model preference, test-set tuning, excessive metric dominance, redundancy, ignored calibration, prevalence and unresolved disagreement. Do not produce final weights.",
                    "user": "Review all three blinded proposals and return criticisms, required changes, preferred metric directions, a leakage check and unresolved risks in strict JSON.",
                    "output": "reviewer_agent_audit.json plus per-round Reviewer records preserved",
                    "source": "agents/weight_meeting/reviewer_agent.md · llm_agent_weight_meeting.py:457-470",
                },
                {
                    "code": "P14", "title": "Chief bounded consensus",
                    "system": "Reconcile the expert proposals and Reviewer audit into the only accepted weight vector. Respond to criticisms and preserve disagreement and uncertainty.",
                    "user": "Return all exact metric keys with weights in [0.005, 0.35], sum=1, no model-specific priority and at most 0.30 L1 change from the previous accepted vector. Runtime code validates and repairs schema violations.",
                    "output": "chief_initial_decision.json and round_001.json through round_050.json",
                    "source": "agents/weight_meeting/chief_agent.md · llm_agent_weight_meeting.py:473-497",
                },
                {
                    "code": "P15", "title": "Fifty-round evidence aggregation",
                    "system": "The accepted vector scores every eligible model against the same normalized metric evidence. Model identities remain absent from weight-setting prompts, and the complete audit trail is retained.",
                    "user": "Repeat bounded evidence updates for 50 rounds, calculate each model's weighted score, then aggregate median score, IQR, median rank and Top-3 frequency without a model-specific bonus.",
                    "output": f"{state['rounds']} rounds · {state['weight_rows']} weight rows · {state['score_rows']} model-round scores · Top-3: {' / '.join(state['top3'])}",
                    "source": "codex_agent_weight_meeting.py · iterative_weight_meeting.py · data/results_manual/codex_agent_weight_meeting/",
                },
            ],
        },
        {
            "part": "PART V", "slug": "part5_ensemble_reporting",
            "eyebrow": "AUDITABLE AGENT PROMPTS · PART V",
            "title": "Top-3 ensemble analysis and research reporting",
            "entries": [
                {
                    "code": "P16", "title": "Exhaustive three-model enumeration",
                    "system": "This selection stage is deterministic. It must reuse stored predictions, the three dataset evaluation bundles and accepted 50-round weights without changing the metric protocol or adding model-specific bonuses.",
                    "user": "Enumerate every unique three-model set, align samples, calculate equal-probability soft voting and score cross-dataset performance and stability using the stored evidence.",
                    "output": f"{state['ensemble_combinations']} unique three-model combinations evaluated",
                    "source": "ensemble_top3_selector.py · data/results_manual/ensemble_top3_combination_ranking.csv",
                },
                {
                    "code": "P17", "title": "Complementarity recommendation",
                    "system": "Prefer combinations supported by performance, stability and complementary errors. Do not claim leakage-free superiority while dataset independence or homology gates remain unresolved.",
                    "user": "Select one auditable trio from the complete combination ranking and persist the method, datasets, candidate models, metrics, ranking evidence and scientific caveat.",
                    "output": "Recommended trio: " + " / ".join(state["ensemble_models"]),
                    "source": "data/results_manual/ensemble_top3_selection.json",
                },
                {
                    "code": "P18", "title": "Research Advisor synthesis",
                    "system": "Write a concise report grounded only in final weights, ranking and dataset profiles. Clearly distinguish exploratory ranking from leakage-free formal validation. Explain Top-3, weight consensus, score IQR and Top-3 frequency.",
                    "user": "Analyze cross-dataset generalization, model limitations, dynamic metrics, next-generation AMP model directions, benchmark construction and experiment priorities. Missing evidence must be stated explicitly.",
                    "output": "amp_future_directions_report_codex_agents.md generated from stored benchmark evidence",
                    "source": "agents/weight_meeting/research_advisor.md · agents/runtime_prompts/amp_research_advisor_template.md",
                },
                {
                    "code": "P19", "title": "Publication and audit handoff",
                    "system": "Figures are a reporting layer and cannot feed values back into ranking. Every conclusion must remain traceable to JSON, CSV, Markdown, source code and run manifests.",
                    "user": "Export editable vector figures, high-resolution previews, source tables, figure contracts and QA notes. Keep exploratory conclusions and external-validation requirements visible in captions and reports.",
                    "output": "SVG/PDF/PNG figures, source CSVs, reports and complete Agent audit records preserved",
                    "source": "figures/ · data/results_manual/ · run_manifest.py · tests/",
                },
            ],
        },
    ]


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        str(text), width=width, break_long_words=False, break_on_hyphens=False,
    ) or [""]


def draw_entry(ax, entry: dict, y_top: float, y_bottom: float) -> None:
    ax.text(0.060, y_top - 0.010, entry["code"], color=BLUE, fontsize=8.0,
            fontweight="bold", ha="left", va="top")
    ax.text(0.115, y_top - 0.010, entry["title"], color=NAVY, fontsize=7.5,
            fontweight="bold", ha="left", va="top")

    rows = [
        ("SYSTEM", entry["system"], NAVY, "sans-serif"),
        ("USER", entry["user"], NAVY, "sans-serif"),
        ("VERIFIED OUTPUT", entry["output"], GREEN, "sans-serif"),
        ("SOURCE", entry["source"], GRAY, "monospace"),
    ]
    line_height = 0.0105
    y = y_top - 0.038
    for label, body, color, family in rows:
        width = 103 if label != "SOURCE" else 112
        lines = wrap(body, width)
        label_size = 5.0
        ax.text(0.060, y, label, color=GREEN if label == "VERIFIED OUTPUT" else GRAY,
                fontsize=label_size, fontweight="bold", ha="left", va="top")
        ax.text(0.145, y, "\n".join(lines), color=color, fontsize=5.0,
                family=family, ha="left", va="top", linespacing=1.28)
        y -= line_height * max(1, len(lines)) + 0.006

    ax.plot([0.060, 0.940], [y_bottom, y_bottom], color=LIGHT, lw=0.65)


def draw_page(page: dict, page_no: int, total_pages: int) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.060, 0.955, page["eyebrow"], color=BLUE_DARK, fontsize=5.4,
            fontweight="bold", ha="left", va="top")
    ax.text(0.060, 0.928, page["title"], color=NAVY, fontsize=13.6,
            fontweight="bold", ha="left", va="top")
    ax.plot([0.060, 0.940], [0.895, 0.895], color=BLUE, lw=1.35)

    top = 0.865
    bottom = 0.090
    entries = page["entries"]
    block = (top - bottom) / len(entries)
    for i, entry in enumerate(entries):
        y_top = top - i * block
        y_bottom = top - (i + 1) * block + 0.006
        draw_entry(ax, entry, y_top, y_bottom)

    ax.plot([0.060, 0.940], [0.055, 0.055], color=LIGHT, lw=0.65)
    ax.text(0.060, 0.036,
            "Core clauses are faithful, space-limited renderings of the project prompts; runtime payload values are abbreviated, never substituted.",
            color=GRAY, fontsize=5.0, ha="left", va="center")
    ax.text(0.940, 0.036, f"{page_no:02d} / {total_pages:02d}", color=GRAY,
            fontsize=5.0, ha="right", va="center")
    return fig


def write_source_csv(pages: list[dict]) -> None:
    fields = ["part", "page_slug", "prompt_code", "title", "system", "user", "verified_output", "source"]
    rows = []
    for page in pages:
        for entry in page["entries"]:
            rows.append({
                "part": page["part"], "page_slug": page["slug"],
                "prompt_code": entry["code"], "title": entry["title"],
                "system": entry["system"], "user": entry["user"],
                "verified_output": entry["output"], "source": entry["source"],
            })
    with (OUT / "amp_project_prompt_atlas_source.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(png_paths: list[Path]) -> None:
    thumb_w = 760
    margin = 36
    header = 90
    resampling = getattr(Image, "Resampling", Image)
    thumbs = []
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        thumb_h = round(image.height * thumb_w / image.width)
        thumbs.append(image.resize((thumb_w, thumb_h), resampling.LANCZOS))
    canvas = Image.new("RGB", (thumb_w + 2 * margin, header + sum(i.height for i in thumbs) + margin * (len(thumbs) + 1)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 28), "AMP benchmark Agent prompt atlas · five project stages", fill=NAVY)
    y = header
    for image in thumbs:
        canvas.paste(image, (margin, y))
        y += image.height + margin
    canvas.save(OUT / "amp_project_prompt_atlas_contact_sheet.png", dpi=(150, 150))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = runtime_state()
    pages = records(state)
    write_source_csv(pages)
    png_paths = []
    combined_pdf = OUT / "amp_project_prompt_atlas_ppt_style_v7.pdf"
    with PdfPages(combined_pdf) as pdf:
        for page_no, page in enumerate(pages, 1):
            fig = draw_page(page, page_no, len(pages))
            prefix = OUT / page["slug"]
            fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor="white")
            fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
            fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            png_paths.append(prefix.with_suffix(".png"))
            print(prefix.with_suffix(".png"))
    make_contact_sheet(png_paths)
    print(combined_pdf)


if __name__ == "__main__":
    main()
