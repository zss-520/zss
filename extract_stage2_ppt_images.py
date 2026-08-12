# -*- coding: utf-8 -*-
"""Extract slide images (slides 4-14) from the Stage2 supplementary PPTX.

Uses only the standard library (zipfile + xml) so it runs without python-pptx.
For each target slide it resolves the embedded picture relationship and copies
the media file to figures/stage2_ppt_extract/slideNN.<ext>.
"""
from __future__ import annotations

import posixpath
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PPTX = ROOT / "Stage2_Supplementary_Appendix_Two_Stage_style_harmonized.pptx"
OUT = ROOT / "figures" / "stage2_ppt_extract"
TARGET_SLIDES = list(range(4, 15))  # slides 4..14

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
BLIP_TAG = f"{{{A_NS}}}blip"
EMBED_ATTR = f"{{{R_NS}}}embed"


def parse_rels(zipf: zipfile.ZipFile, slide_no: int) -> dict[str, str]:
    rels_path = f"ppt/slides/_rels/slide{slide_no}.xml.rels"
    if rels_path not in zipf.namelist():
        return {}
    root = ET.fromstring(zipf.read(rels_path))
    mapping: dict[str, str] = {}
    # Relationships XML uses the default namespace (no prefix).
    for child in root.iter():
        if not child.tag.endswith("}Relationship") and child.tag != "Relationship":
            continue
        rtype = child.get("Type", "")
        if "image" not in rtype.lower():
            continue
        rid = child.get("Id", "")
        target = child.get("Target", "")
        if rid and target:
            mapping[rid] = target
    return mapping


def slide_image_targets(zipf: zipfile.ZipFile, slide_no: int) -> list[str]:
    slide_path = f"ppt/slides/slide{slide_no}.xml"
    if slide_path not in zipf.namelist():
        return []
    root = ET.fromstring(zipf.read(slide_path))
    rids: list[str] = []
    for blip in root.iter(BLIP_TAG):
        rid = blip.get(EMBED_ATTR)
        if rid:
            rids.append(rid)
    rels = parse_rels(zipf, slide_no)
    targets = [rels[r] for r in rids if r in rels]
    # resolve relative to ppt/slides/ (e.g. ../media/image4.png -> ppt/media/image4.png)
    resolved = [posixpath.normpath(posixpath.join("ppt/slides", t)) for t in targets]
    return resolved


def main() -> None:
    if not PPTX.exists():
        raise FileNotFoundError(PPTX)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(PPTX) as zipf:
        for n in TARGET_SLIDES:
            targets = slide_image_targets(zipf, n)
            for idx, rel_target in enumerate(targets):
                member = rel_target.lstrip("/")
                if member not in zipf.namelist():
                    continue
                ext = Path(member).suffix.lower() or ".png"
                out_name = f"slide{n:02d}_img{idx+1}{ext}"
                out_path = OUT / out_name
                with zipf.open(member) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                manifest.append({"slide": n, "image_idx": idx + 1,
                                 "source_member": member, "output": str(out_path)})
                print(f"[slide {n}] {member} -> {out_path.name}")
    print(f"[done] extracted {len(manifest)} images to {OUT}")
    import json
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
