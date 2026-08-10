from __future__ import annotations

from pathlib import Path
import base64
import csv
import io
import re
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


SOURCE_SVG = Path(r"C:\Users\203-2\Desktop\fig5-bcd.svg")
ROOT = Path(__file__).resolve().parent
ASSET_SOURCE = ROOT.parent / "amp_bcd_fig5_agents_editable_v45" / "assets"
ASSET_DELIVERY = ROOT / "input_output_assets_hd"
OUTPUT_SVG = ROOT / "fig5-bcd-input-output-hd.svg"
PREVIEW_PNG = ROOT / "fig5-bcd-input-output-hd-preview.png"
OUTPUT_PDF = ROOT / "fig5-bcd-input-output-hd.pdf"
OUTPUT_TIFF = ROOT / "fig5-bcd-input-output-hd.tiff"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    }
)


# Exact image-to-project-asset mapping verified by pixelwise comparison after
# reproducing the SVG's vertical image transform. Only the embedded screenshots
# are replaced; all vector architecture elements remain byte-for-byte intact.
IMAGE_MAP = {
    # Stage 3 inputs and outputs
    "img0": "stage3_prompt_contract.png",
    "img2": "stage3_blinded_evidence.png",
    "img4": "consensus_rank_top3_bubble_only.png",
    "img7": "restricted_candidate_report.png",
    # Stage 2 inputs and outputs
    "img9": "stage2_prompt_contract.png",
    "img11": "stage2_registry_input.png",
    "img13": "C_AMPs-predict_test_pr_crop.png",
    "img15": "Veltri_test_pr_crop.png",
    "img17": "ProteoGPT_all_predictions_pr_crop.png",
    # Stage 1 inputs and outputs
    "img19": "recommended_datasets_table.png",
    "img21": "stage1_prompt_contract.png",
    "img23": "stage1_compact_evidence.png",
    "img25": "recommended_metrics_table.png",
    "img27": "recommended_models_table.png",
}


def png_data_uri_for_svg(asset: Path) -> tuple[str, tuple[int, int]]:
    """Encode a lossless high-resolution PNG in the orientation expected by SVG."""
    image = Image.open(asset).convert("RGB")
    size = image.size
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, dpi=(600, 600))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", size


def replace_href(svg: str, image_id: str, data_uri: str) -> tuple[str, int, int]:
    pattern = re.compile(
        rf'<image\b(?=[^>]*\bid="{re.escape(image_id)}")[^>]*></image>',
        re.DOTALL,
    )
    match = pattern.search(svg)
    if not match:
        raise RuntimeError(f"Missing embedded image definition: {image_id}")
    tag = match.group(0)
    width_match = re.search(r'\bwidth="([0-9.]+)"', tag)
    height_match = re.search(r'\bheight="([0-9.]+)"', tag)
    if not width_match or not height_match:
        raise RuntimeError(f"Missing display dimensions: {image_id}")
    new_tag, count = re.subn(
        r'xlink:href="[^"]*"',
        f'xlink:href="{data_uri}"',
        tag,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not replace image payload: {image_id}")
    svg = svg[: match.start()] + new_tag + svg[match.end() :]
    return svg, int(float(width_match.group(1))), int(float(height_match.group(1)))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_DELIVERY.mkdir(parents=True, exist_ok=True)
    svg = SOURCE_SVG.read_text(encoding="utf-8")
    audit_rows = []

    for image_id, filename in IMAGE_MAP.items():
        asset = ASSET_SOURCE / filename
        if not asset.exists():
            raise FileNotFoundError(asset)
        data_uri, source_size = png_data_uri_for_svg(asset)
        svg, display_width, display_height = replace_href(svg, image_id, data_uri)
        shutil.copy2(asset, ASSET_DELIVERY / filename)
        audit_rows.append(
            {
                "svg_image_id": image_id,
                "asset": filename,
                "display_width_px": display_width,
                "display_height_px": display_height,
                "embedded_width_px": source_size[0],
                "embedded_height_px": source_size[1],
                "width_resolution_multiplier": round(source_size[0] / display_width, 2),
                "height_resolution_multiplier": round(source_size[1] / display_height, 2),
            }
        )

    OUTPUT_SVG.write_text(svg, encoding="utf-8")
    with (ROOT / "input_output_resolution_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    # A raster preview is rendered from the SVG during visual QA. When present,
    # package it as conventional journal-review PDF/TIFF companions. The SVG
    # remains the primary editable publication artifact.
    if PREVIEW_PNG.exists():
        preview = Image.open(PREVIEW_PNG).convert("RGB")
        fig, ax = plt.subplots(figsize=(7.2, 12.8))  # 182.9 mm wide
        ax.imshow(preview)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(OUTPUT_PDF, bbox_inches=None)
        fig.savefig(
            OUTPUT_TIFF,
            dpi=600,
            bbox_inches=None,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
