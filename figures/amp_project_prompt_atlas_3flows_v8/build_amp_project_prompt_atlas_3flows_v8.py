from __future__ import annotations

import csv
import importlib.util
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw


OUT = Path(__file__).resolve().parent
V7_PATH = (
    ROOT / "figures" / "amp_project_prompt_atlas_ppt_style_v7"
    / "build_amp_project_prompt_atlas_ppt_style_v7.py"
)
SPEC = importlib.util.spec_from_file_location("prompt_atlas_v7", V7_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import source prompt atlas: {V7_PATH}")
V7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V7)

WIDTH_MM, HEIGHT_MM = 210.0, 297.0
WIDTH_IN, HEIGHT_IN = WIDTH_MM / 25.4, HEIGHT_MM / 25.4

NAVY = "#111C2E"
BLUE = "#17698D"
BLUE_DARK = "#315B77"
GREEN = "#16844A"
GRAY = "#617184"
LIGHT = "#D7E1E8"
WHITE = "#FFFFFF"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def merged_entry(code: str, title: str, entries: list[dict], output: str | None = None) -> dict:
    system = " ".join(entry["system"] for entry in entries)
    user = " ".join(entry["user"] for entry in entries)
    source_parts = []
    for entry in entries:
        for part in entry["source"].split(" · "):
            if part not in source_parts:
                source_parts.append(part)
    sources = " · ".join(source_parts)
    return {
        "code": code,
        "title": title,
        "system": system,
        "user": user,
        "output": output or " · ".join(entry["output"] for entry in entries),
        "source": sources,
    }


def build_pages(state: dict) -> list[dict]:
    old = V7.records(state)
    literature = old[0]["entries"]
    onboarding = old[1]["entries"]
    benchmark = old[2]["entries"]
    ranking = old[3]["entries"]
    ensemble = old[4]["entries"]

    critic = {
        "code": "P03", "title": "Independent evidence and continuity audit",
        "system": "Audit scope, provenance, reproducibility, leakage and unsupported certainty. Check out-of-scope systems, absent weights, ambiguous aliases, dataset independence, threshold leakage and continuity with prior memory.",
        "user": "Adjudicate every proposed model and dataset as accept, reject or defer. A previously retained model may be downgraded only with explicit new evidence; a coverage target cannot be copied into the recommendation list.",
        "output": "Model, dataset and metric decisions preserved with warnings, dissent and follow-up tasks",
        "source": "agents/deepseek_meeting/critic_agent.md · deep_research_literature_agent.py:3374-3402",
    }
    chief = {
        "code": "P04", "title": "Chief reconciliation and persistent memory",
        "system": "Reconcile Scout, Metrics and Critic outputs into long-term meeting memory. Preserve proposals, criticisms, rebuttals, dissent, provenance, blocking issues and final execution decisions.",
        "user": "Write strict JSON for all candidates, benchmark-ready and deployment models, dataset decision traces, architecture representatives, metrics and open questions. Historical memory is an evidence anchor, not a fixed winner list.",
        "output": "literature_deep_research_memory.json/.md and recommendation CSVs refreshed",
        "source": "agents/deepseek_meeting/chief_agent.md · deep_research_literature_agent.py:3407-3474",
    }

    flow1_entries = []
    for i, entry in enumerate(literature[:3]):
        item = dict(entry)
        item["code"] = f"P{i:02d}"
        flow1_entries.append(item)
    flow1_entries.extend([critic, chief])

    registry = merged_entry(
        "P05", "Repository inspection and validated registration",
        onboarding[:2],
        output=f"{state['registry']} registry records persisted; Agent proposals remain separate from readiness",
    )
    self_heal = merged_entry(
        "P06", "Bounded self-heal and smoke-test admission",
        onboarding[2:4],
        output=f"{state['repairs']} repair artifacts; {state['registry_ready']} of {state['registry']} records currently ready",
    )
    flow2_entries = [registry, self_heal]
    for i, entry in enumerate(benchmark, 7):
        item = dict(entry)
        item["code"] = f"P{i:02d}"
        flow2_entries.append(item)

    flow3_entries = []
    for i, entry in enumerate(ranking, 11):
        item = dict(entry)
        item["code"] = f"P{i:02d}"
        flow3_entries.append(item)
    trio = merged_entry(
        "P15", "Exhaustive Top-3 complementarity selection",
        ensemble[:2],
        output=(
            f"{state['ensemble_combinations']} combinations; recommended trio: "
            + " / ".join(state["ensemble_models"])
        ),
    )
    report = merged_entry(
        "P16", "Research Advisor and publication handoff",
        ensemble[2:4],
        output="Evidence-grounded future-directions report plus editable figures, source CSVs and QA records",
    )
    flow3_entries.extend([trio, report])

    return [
        {
            "slug": "flow1_literature_memory",
            "eyebrow": "AUDITABLE AGENT PROMPTS · FLOW I",
            "title": "Evidence search, round-table review and persistent memory",
            "entries": flow1_entries,
        },
        {
            "slug": "flow2_deployment_evaluation",
            "eyebrow": "AUDITABLE AGENT PROMPTS · FLOW II",
            "title": "Model deployment, self-healing and unified evaluation",
            "entries": flow2_entries,
        },
        {
            "slug": "flow3_ranking_ensemble",
            "eyebrow": "AUDITABLE AGENT PROMPTS · FLOW III",
            "title": "Fifty-round ranking, Top-3 ensemble and reporting",
            "entries": flow3_entries,
        },
    ]


def wrap(value: str, width: int) -> list[str]:
    return textwrap.wrap(
        str(value), width=width, break_long_words=False, break_on_hyphens=False,
    ) or [""]


def draw_entry(ax, entry: dict, y_top: float, y_bottom: float) -> None:
    ax.text(0.055, y_top - 0.006, entry["code"], color=BLUE, fontsize=8.4,
            fontweight="bold", ha="left", va="top")
    ax.text(0.112, y_top - 0.006, entry["title"], color=NAVY, fontsize=8.0,
            fontweight="bold", ha="left", va="top")

    rows = [
        ("SYSTEM", entry["system"], NAVY, "sans-serif", 94),
        ("USER", entry["user"], NAVY, "sans-serif", 94),
        ("VERIFIED OUTPUT", entry["output"], GREEN, "sans-serif", 98),
        ("SOURCE", entry["source"], GRAY, "monospace", 104),
    ]
    y = y_top - 0.028
    line_height = 0.0091
    for label, body, color, family, width in rows:
        lines = wrap(body, width)
        ax.text(0.055, y, label, color=GREEN if label == "VERIFIED OUTPUT" else GRAY,
                fontsize=5.4, fontweight="bold", ha="left", va="top")
        ax.text(0.145, y, "\n".join(lines), color=color, fontsize=5.65,
                family=family, ha="left", va="top", linespacing=1.18)
        y -= line_height * len(lines) + 0.0035
    ax.plot([0.055, 0.945], [y_bottom, y_bottom], color=LIGHT, lw=0.65)


def entry_height(entry: dict) -> float:
    widths = {"system": 94, "user": 94, "output": 98, "source": 104}
    line_count = sum(len(wrap(entry[key], width)) for key, width in widths.items())
    return 0.050 + 0.0091 * line_count


def draw_page(page: dict, page_no: int, total: int) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.055, 0.967, page["eyebrow"], color=BLUE_DARK, fontsize=6.0,
            fontweight="bold", ha="left", va="top")
    ax.text(0.055, 0.938, page["title"], color=NAVY, fontsize=14.2,
            fontweight="bold", ha="left", va="top")
    ax.plot([0.055, 0.945], [0.901, 0.901], color=BLUE, lw=1.45)

    top = 0.878
    bottom = 0.058
    available = top - bottom
    heights = [entry_height(entry) for entry in page["entries"]]
    extra = max(0.0, available - sum(heights)) / len(heights)
    heights = [height + extra for height in heights]
    if sum(heights) > available:
        scale = available / sum(heights)
        heights = [height * scale for height in heights]
    cursor = top
    for entry, height in zip(page["entries"], heights):
        y_top = cursor
        cursor -= height
        draw_entry(ax, entry, y_top, cursor + 0.003)

    ax.plot([0.055, 0.945], [0.033, 0.033], color=LIGHT, lw=0.65)
    ax.text(0.055, 0.018, "Core prompt clauses shown; large runtime JSON payloads remain in the cited audit files.",
            color=GRAY, fontsize=5.2, ha="left", va="center")
    ax.text(0.945, 0.018, f"{page_no:02d} / {total:02d}", color=GRAY,
            fontsize=5.2, ha="right", va="center")
    return fig


def write_source_csv(pages: list[dict]) -> None:
    fields = ["flow", "page_slug", "prompt_code", "title", "system", "user", "verified_output", "source"]
    rows = []
    for flow_no, page in enumerate(pages, 1):
        for entry in page["entries"]:
            rows.append({
                "flow": flow_no,
                "page_slug": page["slug"],
                "prompt_code": entry["code"],
                "title": entry["title"],
                "system": entry["system"],
                "user": entry["user"],
                "verified_output": entry["output"],
                "source": entry["source"],
            })
    with (OUT / "amp_project_prompt_atlas_3flows_source.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(paths: list[Path]) -> None:
    thumb_w = 900
    margin = 42
    header = 100
    resampling = getattr(Image, "Resampling", Image)
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = round(image.height * thumb_w / image.width)
        images.append(image.resize((thumb_w, height), resampling.LANCZOS))
    canvas = Image.new("RGB", (thumb_w + 2 * margin, header + sum(i.height for i in images) + margin * (len(images) + 1)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 30), "AMP benchmark Agent prompt atlas · three consolidated flows", fill=NAVY)
    y = header
    for image in images:
        canvas.paste(image, (margin, y))
        y += image.height + margin
    canvas.save(OUT / "amp_project_prompt_atlas_3flows_contact_sheet.png", dpi=(150, 150))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = V7.runtime_state()
    pages = build_pages(state)
    write_source_csv(pages)
    pngs = []
    combined = OUT / "amp_project_prompt_atlas_3flows_v8.pdf"
    with PdfPages(combined) as pdf:
        for page_no, page in enumerate(pages, 1):
            fig = draw_page(page, page_no, len(pages))
            prefix = OUT / page["slug"]
            fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor="white")
            fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
            fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            pngs.append(prefix.with_suffix(".png"))
            print(prefix.with_suffix(".png"))
    contact_sheet(pngs)
    print(combined)


if __name__ == "__main__":
    main()
