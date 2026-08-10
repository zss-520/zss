# -*- coding: utf-8 -*-
"""Build the six-panel AMP Agent framework using real project outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE = ROOT / "figures" / "amp_framework_real_outputs_hd_v41" / "amp_agent_framework_real_outputs_hd_v41.png"

STEP2_TABLES = ROOT / "figures" / "publication" / "step2_agent_evaluation_revised.png"
STEP2_LANDSCAPE = (
    ROOT
    / "data"
    / "results_manual"
    / "publication_figures_filtered"
    / "filtered_cross_dataset_performance.png"
)
RANKING = ROOT / "figures" / "publication" / "posthoc_filtered_ranking" / "posthoc_filtered_boxplot_bubble.png"
REPORT = ROOT / "figures" / "amp_main_layered_editable_v40" / "assets" / "stage3_report_md.png"

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


SCALE = 4
GREEN = "#2D9B65"
PURPLE = "#7C5BB3"
TEXT = "#1F2937"
SUBPANEL_COLORS = ["#2F80ED", "#2D9B65", "#F2994A", "#7C5BB3"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def scaled_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(value * SCALE for value in box)


def crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((int(width * left), int(height * top), int(width * right), int(height * bottom)))


def paste_contain(canvas: Image.Image, asset: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    target = ImageOps.contain(asset.convert("RGB"), (x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - target.width) // 2
    y = y0 + (y1 - y0 - target.height) // 2
    canvas.paste(target, (x, y))


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: str,
    title: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rectangle(
        (x0 - 10 * SCALE, y0 - 10 * SCALE, x1 + 12 * SCALE, y1 + 10 * SCALE),
        fill="white",
    )
    draw.rounded_rectangle(
        box,
        radius=10 * SCALE,
        fill="white",
        outline=color,
        width=2 * SCALE,
    )
    draw.text(
        (x0 + 14 * SCALE, y0 + 9 * SCALE),
        title,
        fill=color,
        font=font(12 * SCALE, bold=True),
    )
    return x0 + 12 * SCALE, y0 + 38 * SCALE, x1 - 12 * SCALE, y1 - 12 * SCALE


def mini_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    image: Image.Image,
    label: str,
    color: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6 * SCALE, fill="white", outline=color, width=2 * SCALE)
    draw.text(
        (x0 + 9 * SCALE, y0 + 7 * SCALE),
        label,
        fill=color,
        font=font(8 * SCALE, bold=True),
    )
    paste_contain(
        canvas,
        image,
        (x0 + 5 * SCALE, y0 + 27 * SCALE, x1 - 5 * SCALE, y1 - 5 * SCALE),
    )


def render_step2_outputs(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    inner = card(
        draw,
        box,
        GREEN,
        "Real Stage 2 outputs: exact tables and cross-dataset evidence",
    )
    x0, y0, x1, y1 = inner
    gap = 8 * SCALE
    half_width = (x1 - x0 - gap) // 2
    half_height = (y1 - y0 - gap) // 2

    table_page = Image.open(STEP2_TABLES).convert("RGB")
    landscape = Image.open(STEP2_LANDSCAPE).convert("RGB")

    raw_tables = crop_fraction(table_page, (0.03, 0.64, 0.99, 0.985))
    heatmap = crop_fraction(landscape, (0.025, 0.00, 0.985, 0.625))
    rank_shift = crop_fraction(landscape, (0.02, 0.61, 0.55, 0.995))
    operating_point = crop_fraction(landscape, (0.54, 0.61, 0.995, 0.995))

    panels = [
        (raw_tables, "All-model exact metric tables", SUBPANEL_COLORS[0]),
        (heatmap, "Filtered-cohort performance landscape", SUBPANEL_COLORS[1]),
        (rank_shift, "Cross-dataset rank shifts", SUBPANEL_COLORS[2]),
        (operating_point, "Precision / recall trade-off", SUBPANEL_COLORS[3]),
    ]
    boxes = [
        (x0, y0, x0 + half_width, y0 + half_height),
        (x0 + half_width + gap, y0, x1, y0 + half_height),
        (x0, y0 + half_height + gap, x0 + half_width, y1),
        (x0 + half_width + gap, y0 + half_height + gap, x1, y1),
    ]
    for panel_box, (asset, label, color) in zip(boxes, panels):
        mini_panel(canvas, draw, panel_box, asset, label, color)


def render_ranking_outputs(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    inner = card(draw, box, PURPLE, "Real 50-round ranking and generated Agent report")
    x0, y0, x1, y1 = inner
    gap = 10 * SCALE
    ranking = Image.open(RANKING).convert("RGB")
    bubble = crop_fraction(ranking, (0.485, 0.0, 1.0, 1.0))
    report = Image.open(REPORT).convert("RGB")
    split = x0 + int((x1 - x0 - gap) * 0.64)

    draw.rounded_rectangle((x0, y0, split, y1), radius=6 * SCALE, fill="white", outline=PURPLE, width=2 * SCALE)
    paste_contain(canvas, bubble, (x0 + 5 * SCALE, y0 + 5 * SCALE, split - 5 * SCALE, y1 - 5 * SCALE))

    draw.rounded_rectangle(
        (split + gap, y0, x1, y1),
        radius=6 * SCALE,
        fill="white",
        outline=GREEN,
        width=2 * SCALE,
    )
    paste_contain(canvas, report, (split + gap + 5 * SCALE, y0 + 5 * SCALE, x1 - 5 * SCALE, y1 - 5 * SCALE))


def export(canvas: Image.Image) -> dict[str, str]:
    prefix = OUT_DIR / "amp_framework_user_reference_v54"
    paths = {
        "png": str(prefix.with_suffix(".png")),
        "tiff": str(prefix.with_suffix(".tiff")),
        "pdf": str(prefix.with_suffix(".pdf")),
        "svg": str(prefix.with_suffix(".svg")),
    }
    canvas.save(paths["png"], format="PNG", dpi=(600, 600), optimize=True)
    canvas.save(paths["tiff"], format="TIFF", dpi=(600, 600), compression="tiff_lzw")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 10.1855), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(paths["svg"], dpi=600, facecolor="white", pad_inches=0)
    fig.savefig(paths["pdf"], dpi=600, facecolor="white", pad_inches=0)
    plt.close(fig)
    return paths


def main() -> int:
    required = [BASE, STEP2_TABLES, STEP2_LANDSCAPE, RANKING, REPORT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required figure assets:\n" + "\n".join(missing))

    canvas = Image.open(BASE).convert("RGB")
    canvas = ImageEnhance.Contrast(canvas).enhance(1.01)
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=4))
    draw = ImageDraw.Draw(canvas)

    render_step2_outputs(canvas, draw, scaled_box((900, 720, 1570, 1075)))
    render_ranking_outputs(canvas, draw, scaled_box((855, 1105, 1570, 1490)))
    paths = export(canvas)

    audit = {
        "core_conclusion": "The AMP workflow links evidence mining, bounded deployment, measured benchmarking, 50-round ranking, ensemble design and configuration evaluation through auditable Agent handoffs.",
        "archetype": "schematic-led composite",
        "base_layout": str(BASE.relative_to(ROOT)),
        "real_stage2_sources": [
            str(STEP2_TABLES.relative_to(ROOT)),
            str(STEP2_LANDSCAPE.relative_to(ROOT)),
        ],
        "real_stage3_sources": [
            str(RANKING.relative_to(ROOT)),
            str(REPORT.relative_to(ROOT)),
        ],
        "data_values_modified": False,
        "cohort_note": "The raw metric-table screenshots retain all 18 evaluated models. The cross-dataset landscape and ranking outputs use the audited 15-model posthoc display cohort.",
        "excluded_models_from_display_cohort": [
            "pepnet_standard",
            "amplify_imb",
            "amplify_bal",
        ],
        "exclusion_rule": "Retain the three requested focal models and every model originally ranked below the lowest-ranked focal model.",
        "posthoc_result_conditioned_filter": True,
        "valid_for_unbiased_global_top3_claim": False,
        "output_pixels": list(canvas.size),
        "output_dpi": 600,
        "outputs": paths,
        "note": "The supplied six-panel structure was retained. Only the c and d output regions were replaced with higher-resolution real project artifacts.",
    }
    (OUT_DIR / "source_and_integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    print(f"pixels={canvas.width}x{canvas.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
