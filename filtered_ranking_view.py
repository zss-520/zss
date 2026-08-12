# -*- coding: utf-8 -*-
"""Create an explicitly post-hoc display view from a complete model ranking."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


def build_filtered_view(
    source_csv: Path,
    output_dir: Path,
    target_models: Sequence[str],
) -> Dict[str, Any]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"rank", "model", "median_score"}.issubset(rows[0]):
        raise ValueError("Source ranking CSV is empty or missing rank/model/median_score")

    targets = list(dict.fromkeys(str(model).strip() for model in target_models if str(model).strip()))
    by_model = {str(row["model"]): row for row in rows}
    missing = [model for model in targets if model not in by_model]
    if missing:
        raise ValueError(f"Target models missing from source ranking: {missing}")

    target_global_ranks = {model: int(by_model[model]["rank"]) for model in targets}
    cutoff = max(target_global_ranks.values())
    excluded = [
        row for row in rows
        if int(row["rank"]) < cutoff and str(row["model"]) not in set(targets)
    ]
    retained = [
        row for row in rows
        if str(row["model"]) in set(targets) or int(row["rank"]) > cutoff
    ]
    retained.sort(key=lambda row: int(row["rank"]))

    filtered_rows = []
    for filtered_rank, row in enumerate(retained, 1):
        filtered_rows.append({
            "filtered_rank": filtered_rank,
            "global_rank": int(row["rank"]),
            **{key: value for key, value in row.items() if key != "rank"},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "posthoc_filtered_model_ranking.csv"
    fields = list(filtered_rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(filtered_rows)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ranking_csv": str(source_csv),
        "target_models": targets,
        "target_global_ranks": target_global_ranks,
        "cutoff_global_rank": cutoff,
        "rule": "retain target models and every model originally ranked below the lowest-ranked target",
        "excluded_models": [
            {"model": row["model"], "global_rank": int(row["rank"])} for row in excluded
        ],
        "retained_model_count": len(filtered_rows),
        "scores_recomputed": False,
        "metric_weights_changed": False,
        "posthoc_result_conditioned_filter": True,
        "valid_for_unbiased_global_top3_claim": False,
        "interpretation": "display-only ranking; complete benchmark ranking remains authoritative",
        "output_csv": str(csv_path),
    }
    audit_path = output_dir / "posthoc_filtered_ranking_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Post-hoc filtered model ranking",
        "",
        "> Display-only view. This is not an unbiased global Top3 ranking.",
        "",
        f"- Target models: {', '.join(targets)}",
        f"- Hidden higher-ranked non-target models: {', '.join(row['model'] for row in excluded)}",
        "- Scores recomputed: no",
        "- Metric weights changed: no",
        "",
        "| Filtered rank | Global rank | Model | Median score | Top3 frequency |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in filtered_rows:
        lines.append(
            f"| {row['filtered_rank']} | {row['global_rank']} | {row['model']} | "
            f"{float(row['median_score']):.6f} | {float(row['top3_frequency']):.1%} |"
        )
    report_path = output_dir / "posthoc_filtered_model_ranking.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": filtered_rows, "audit": audit, "report_file": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-models", nargs="+", required=True)
    args = parser.parse_args()
    result = build_filtered_view(
        args.source_csv.resolve(), args.output_dir.resolve(), args.target_models
    )
    print(json.dumps({
        "ranking": result["rows"],
        "excluded_models": result["audit"]["excluded_models"],
        "report_file": result["report_file"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
