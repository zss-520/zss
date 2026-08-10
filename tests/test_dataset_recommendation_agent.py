from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_recommendation_agent import recommend


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _sequence(index: int, length: int = 10) -> str:
    chars = ["A"] * length
    value = index + 1
    for offset in range(length):
        chars[offset] = ALPHABET[value % len(ALPHABET)]
        value //= len(ALPHABET)
    return "".join(chars)


def _write_dataset(root: Path, name: str, positive: int, negative: int, start: int) -> None:
    directory = root / "data" / "datasets" / name
    directory.mkdir(parents=True)
    with (directory / "ground_truth.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label"])
        for offset in range(positive + negative):
            writer.writerow([f"s{start + offset}", _sequence(start + offset), 1 if offset < positive else 0])


def _write_policy(root: Path, **overrides: object) -> None:
    policy = {
        "enabled": True,
        "required_dataset_count": 3,
        "required_profiles": {"balanced": 1, "imbalanced": 2},
        "balanced_minority_majority_ratio_min": 0.7,
        "min_imbalanced_minority_fraction_gap": 0.1,
        "min_total_samples": 2,
        "min_samples_per_class": 1,
        "min_primary_length_fraction": 0.8,
        "absolute_min_length_aa": 5,
        "absolute_max_length_aa": 100,
        "max_class_median_length_gap_aa": 15,
    }
    policy.update(overrides)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "dataset_selection_policy.json").write_text(
        json.dumps({"dataset_selection_policy": policy}), encoding="utf-8"
    )
    (data / "dataset_metadata.json").write_text('{"datasets": []}', encoding="utf-8")
    (data / "evidence_pool.json").write_text('{"external_datasets": []}', encoding="utf-8")


def _write_manual_result(root: Path, name: str, positive: int, negative: int, start: int) -> None:
    directory = root / "data" / "results_manual" / name
    directory.mkdir(parents=True)
    with (directory / "final_results_with_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Standard_ID", "True_Label", "Example_Prob"])
        for offset in range(positive + negative):
            writer.writerow([_sequence(start + offset), 1 if offset < positive else 0, 0.5])


class DatasetRecommendationAgentTests(unittest.TestCase):
    def test_manual_results_dynamically_form_balanced_plus_two_imbalanced_trio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            (root / "data" / "literature_deep_research_memory.json").write_text(
                '{"final_recommended_datasets": [], "datasets": []}', encoding="utf-8"
            )
            names = ["C_AMPs-predict_test", "ProteoGPT_all_predictions", "Veltri_test"]
            seeds = {
                "datasets": [
                    {
                        "dataset_name": name,
                        "aliases": [name],
                        "source_url": f"https://example.org/{index}",
                        "dataset_role": "AMP external benchmark test set",
                        "evidence_level": "official_repository_plus_primary_paper",
                    }
                    for index, name in enumerate(names, 1)
                ]
            }
            (root / "data" / "required_benchmark_dataset_seeds.json").write_text(
                json.dumps(seeds), encoding="utf-8"
            )
            _write_manual_result(root, "Veltri_test", 5, 5, 0)
            _write_manual_result(root, "ProteoGPT_all_predictions", 7, 3, 100)
            _write_manual_result(root, "C_AMPs-predict_test", 9, 1, 200)

            result = recommend(root)

            selected = result["empirically_evaluated_top3"]
            self.assertEqual(len(selected), 3)
            self.assertEqual(
                {row["dataset_name"] for row in selected},
                set(names),
            )
            self.assertEqual(
                sorted((row.get("audit") or {}).get("observed_profile") for row in selected),
                ["balanced", "imbalanced", "imbalanced"],
            )
            self.assertEqual(
                result["empirically_evaluated_top3_status"],
                "selected_pending_independence_and_homology_gates",
            )

    def test_curated_dataset_seeds_become_auditable_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            (root / "data" / "literature_deep_research_memory.json").write_text(
                '{"final_recommended_datasets": [], "datasets": []}', encoding="utf-8"
            )
            seeds = {
                "datasets": [
                    {
                        "dataset_name": "Veltri / DAMP test set",
                        "aliases": ["APIN benchmarking set"],
                        "linked_models": ["AMP Scanner v2", "APIN"],
                        "source_url": "https://github.com/example/veltri/tree/main/data",
                        "source_doi": "10.0000/example",
                        "dataset_role": "balanced benchmark test set",
                        "evidence_level": "official_repository_plus_primary_paper",
                    }
                ]
            }
            (root / "data" / "required_benchmark_dataset_seeds.json").write_text(
                json.dumps(seeds), encoding="utf-8"
            )

            result = recommend(root)

            self.assertEqual(result["literature_candidate_count"], 1)
            candidate = result["acquisition_queue"][0]
            self.assertIn("APIN benchmarking set", candidate["aliases"])
            self.assertEqual(candidate["linked_models"], ["AMP Scanner v2", "APIN"])
            self.assertIn("required_verified_seed", candidate["origin"])
            self.assertEqual(result["formal_selection"], [])

    def test_literature_only_candidates_never_enter_formal_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            memory = {
                "final_recommended_datasets": [
                    {
                        "dataset_rank": index,
                        "dataset_name": f"AMP candidate {index}",
                        "dataset_source_or_link": f"https://zenodo.org/records/{100 + index}",
                        "source_doi": f"10.0000/test.{index}",
                        "recommendation_origin": "literature_global_meeting_consensus",
                    }
                    for index in range(1, 4)
                ],
                "datasets": [],
            }
            (root / "data" / "literature_deep_research_memory.json").write_text(
                json.dumps(memory), encoding="utf-8"
            )

            result = recommend(root)

            self.assertEqual(result["formal_selection_status"], "blocked_no_three_formally_eligible_meeting_datasets")
            self.assertEqual(result["formal_selection"], [])
            self.assertEqual(len(result["provisional_acquisition_top3"]), 3)
            self.assertFalse((root / "data" / "benchmark_strategy.agent.json").exists())

    def test_meeting_shortlist_without_direct_url_is_kept_for_acquisition_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            memory = {
                "final_recommended_datasets": [
                    {
                        "dataset_rank": index,
                        "dataset_name": f"Meeting paper dataset {index}",
                        "dataset_source_or_link": f"Primary paper DOI 10.0000/no-url.{index}",
                        "source_doi": f"10.0000/no-url.{index}",
                        "recommendation_origin": "literature_global_meeting_consensus",
                    }
                    for index in range(1, 4)
                ],
                "datasets": [],
            }
            (root / "data" / "literature_deep_research_memory.json").write_text(
                json.dumps(memory), encoding="utf-8"
            )

            result = recommend(root)

            self.assertEqual(result["meeting_shortlist_count"], 3)
            self.assertEqual(result["meeting_shortlist_status"], "ready_for_acquisition")
            self.assertTrue(all(not row["source_url"] for row in result["meeting_shortlist"]))
            self.assertEqual(result["formal_selection"], [])

    def test_verified_seeds_cannot_replace_meeting_top_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            memory = {
                "final_recommended_datasets": [
                    {
                        "dataset_rank": index,
                        "dataset_name": f"Meeting choice {index}",
                        "dataset_source_or_link": f"https://example.org/meeting-{index}",
                        "recommendation_origin": "literature_global_meeting_consensus",
                    }
                    for index in range(1, 4)
                ],
                "datasets": [],
            }
            (root / "data" / "literature_deep_research_memory.json").write_text(
                json.dumps(memory), encoding="utf-8"
            )
            (root / "data" / "required_benchmark_dataset_seeds.json").write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "dataset_name": "High scoring seed",
                                "source_url": "https://zenodo.org/records/999",
                                "source_doi": "10.0000/seed",
                                "dataset_role": "AMP external benchmark test set",
                                "evidence_level": "official_repository_plus_primary_paper",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = recommend(root)

            self.assertEqual(
                [row["dataset_name"] for row in result["provisional_acquisition_top3"]],
                ["Meeting choice 1", "Meeting choice 2", "Meeting choice 3"],
            )
            self.assertEqual(result["meeting_shortlist_status"], "ready_for_acquisition")

    def test_real_sequence_audit_selects_one_balanced_and_two_imbalanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root)
            (root / "data" / "literature_deep_research_memory.json").write_text(
                json.dumps(
                    {
                        "final_recommended_datasets": [
                            {
                                "dataset_rank": index,
                                "dataset_name": name,
                                "dataset_source_or_link": f"https://example.org/{name}",
                                "recommendation_origin": "literature_global_meeting_consensus",
                            }
                            for index, name in enumerate(["balanced", "mild", "strong"], 1)
                        ],
                        "datasets": [],
                    }
                ),
                encoding="utf-8",
            )
            _write_dataset(root, "balanced", 7, 5, 0)
            _write_dataset(root, "mild", 6, 4, 100)
            _write_dataset(root, "strong", 8, 2, 200)

            result = recommend(root)

            self.assertEqual(result["formal_selection_status"], "selected")
            profiles = sorted(row["selection_profile"] for row in result["formal_selection"])
            self.assertEqual(profiles, ["balanced", "imbalanced", "imbalanced"])
            self.assertTrue(result["strategy_written"])
            strategy = json.loads((root / "data" / "benchmark_strategy.agent.json").read_text(encoding="utf-8"))
            self.assertEqual(
                strategy["selection_origin"],
                "literature_global_meeting_consensus_then_dataset_recommendation_agent_audit",
            )
            self.assertEqual(len(strategy["recommended_datasets"]), 3)


if __name__ == "__main__":
    unittest.main()
