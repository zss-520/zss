import unittest

from deep_research_literature_agent import build_final_recommended_datasets


class LiteratureMeetingDatasetSelectionTests(unittest.TestCase):
    def test_static_or_seed_candidates_cannot_create_winners(self):
        result = build_final_recommended_datasets(
            {
                "final_recommended_datasets": [
                    {"dataset_name": "Legacy hard-coded winner"}
                ],
                "datasets": [
                    {
                        "dataset_name": "Evidence-only candidate",
                        "dataset_url": "https://example.org/evidence-only",
                    }
                ],
            }
        )
        self.assertEqual(result, [])

    def test_only_meeting_consensus_rows_are_normalized(self):
        result = build_final_recommended_datasets(
            {
                "meeting_recommended_datasets": [
                    {
                        "dataset_rank": 1,
                        "dataset_name": "Meeting balanced",
                        "target_profile": "balanced",
                        "dataset_source_or_link": "https://example.org/balanced",
                        "why_selected": "Scout proposed; Metrics accepted; Critic deferred only sequence audit.",
                    },
                    {
                        "dataset_rank": 2,
                        "dataset_name": "Meeting imbalanced",
                        "target_profile": "imbalanced",
                        "dataset_source_or_link": "https://example.org/imbalanced",
                    },
                ],
                "datasets": [
                    {
                        "dataset_name": "Unselected evidence candidate",
                        "dataset_url": "https://example.org/not-selected",
                    }
                ],
            }
        )
        self.assertEqual(
            [row["dataset_name"] for row in result],
            ["Meeting balanced", "Meeting imbalanced"],
        )
        self.assertTrue(all(row["recommendation_origin"] == "literature_global_meeting_consensus" for row in result))
        self.assertTrue(all(row["needs_sequence_audit"] for row in result))

    def test_complete_empirical_top3_overrides_meeting_without_fixed_names(self):
        empirical = [
            {
                "dataset_name": name,
                "selection_profile": profile,
                "source_url": f"https://example.org/{index}",
                "linked_models": [f"model-{index}"],
                "manual_evaluation_available": True,
                "formal_eligible": False,
                "formal_blockers": ["low_homology_report_missing"],
                "independent_external_test": False,
                "audit": {
                    "row_count": 100 * index,
                    "positive_count": 10 * index,
                    "negative_count": 90 * index,
                    "positive_fraction": 0.1,
                    "observed_profile": profile,
                    "within_dataset_duplicate_count": 0,
                    "length": {"min_aa": 10, "max_aa": 50},
                },
            }
            for index, (name, profile) in enumerate(
                [("dynamic-a", "balanced"), ("dynamic-b", "imbalanced"), ("dynamic-c", "imbalanced")],
                1,
            )
        ]
        payload = {
            "dataset_agent_recommendation": {
                "empirically_evaluated_top3": empirical,
                "empirically_evaluated_top3_status": "selected_pending_independence_and_homology_gates",
                "formal_selection_status": "blocked_no_three_formally_eligible_meeting_datasets",
            },
            "meeting_recommended_datasets": [{"dataset_name": "old-meeting-winner"}],
        }

        result = build_final_recommended_datasets(payload)

        self.assertEqual([row["dataset_name"] for row in result], ["dynamic-a", "dynamic-b", "dynamic-c"])
        self.assertTrue(all(row["recommendation_origin"] == "dataset_agent_empirical_top3_dynamic_merge" for row in result))
        self.assertTrue(all(row["status"] == "selected_pending_independence_and_homology_gates" for row in result))
        self.assertEqual(payload["final_dataset_selection_context"]["selection_is_name_template"], False)

    def test_incomplete_empirical_selection_falls_back_to_meeting(self):
        result = build_final_recommended_datasets(
            {
                "dataset_agent_recommendation": {
                    "empirically_evaluated_top3": [{"dataset_name": "only-one"}],
                    "empirically_evaluated_top3_status": "insufficient_complementary_manual_datasets",
                },
                "meeting_recommended_datasets": [
                    {"dataset_name": "meeting-fallback", "target_profile": "balanced"}
                ],
            }
        )

        self.assertEqual([row["dataset_name"] for row in result], ["meeting-fallback"])


if __name__ == "__main__":
    unittest.main()
