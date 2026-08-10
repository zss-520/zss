# -*- coding: utf-8 -*-
"""Rank a prespecified model cohort without altering the complete benchmark."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from amp_research_advisor import collect_eval_results, load_meeting_metric_weights
from iterative_weight_meeting import run_iterative_weight_meeting


def run_restricted_ranking(
    results_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    rounds: int = 50,
    seed: int = 20260716,
) -> Dict[str, Any]:
    requested = list(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
    if len(requested) < 3:
        raise ValueError("At least three prespecified candidate models are required")

    all_rows = collect_eval_results(results_dir)
    all_models = sorted({str(row["model"]) for row in all_rows}, key=str.lower)
    missing = [model for model in requested if model not in all_models]
    if missing:
        raise ValueError(f"Candidate models are missing from evaluation results: {missing}")

    requested_set = set(requested)
    cohort_rows = [row for row in all_rows if str(row["model"]) in requested_set]
    coverage = {
        model: sorted({str(row["dataset"]) for row in cohort_rows if row["model"] == model})
        for model in requested
    }
    expected_datasets = sorted({str(row["dataset"]) for row in all_rows})
    incomplete = {
        model: datasets for model, datasets in coverage.items() if datasets != expected_datasets
    }
    if incomplete:
        raise ValueError(f"Candidates do not have complete cross-dataset evidence: {incomplete}")

    output_dir.mkdir(parents=True, exist_ok=True)
    meeting = run_iterative_weight_meeting(
        rows=cohort_rows,
        output_dir=output_dir,
        rounds=rounds,
        seed=seed,
        initial_weights=load_meeting_metric_weights(),
    )
    ranking = meeting["final_ranking"]
    scope = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "prespecified_restricted_candidate_cohort",
        "candidate_models": requested,
        "candidate_model_count": len(requested),
        "datasets": expected_datasets,
        "weight_method": "unchanged 50-round evidence-driven dynamic metric-weight meeting",
        "model_specific_score_bonus": False,
        "posthoc_removal_based_on_rank": False,
        "not_a_global_top3_claim": True,
        "models_outside_current_comparison": [model for model in all_models if model not in requested_set],
        "outside_model_reason": "not included in the prespecified candidate cohort; complete benchmark evidence remains unchanged",
        "ranking": ranking,
    }
    scope_path = output_dir / "restricted_candidate_scope.json"
    scope_path.write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Prespecified candidate-cohort single-model ranking",
        "",
        "> This is a restricted-cohort comparison, not a global Top3 claim across all evaluated models.",
        "",
        f"- Candidate models: {', '.join(requested)}",
        f"- Datasets: {', '.join(expected_datasets)}",
        f"- Dynamic weight rounds: {rounds}",
        "- Model-specific score bonus: disabled",
        "- Post-hoc deletion based on observed rank: disabled",
        "",
        "| Cohort rank | Model | Median score | Mean score | Score IQR | Top3 frequency |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['median_score']:.6f} | "
            f"{row['mean_score']:.6f} | {row['score_iqr']:.6f} | {row['top3_frequency']:.1%} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The three rows above are the complete ranking within the prespecified cohort. "
        "Models outside this cohort were not reclassified as failures and remain in the complete benchmark.",
    ])
    report_path = output_dir / "restricted_candidate_ranking.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"scope": scope, "scope_file": str(scope_path), "report_file": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/results_manual"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    result = run_restricted_ranking(
        args.results_dir.resolve(), args.output_dir.resolve(), args.models, args.rounds, args.seed
    )
    print(json.dumps({
        "ranking": result["scope"]["ranking"],
        "report_file": result["report_file"],
        "scope_file": result["scope_file"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
