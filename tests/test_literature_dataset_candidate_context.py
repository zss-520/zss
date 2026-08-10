from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deep_research_literature_agent as literature
from dataset_recommendation_agent import _candidate_name_keys


class _Loader:
    def load(self, name: str) -> str:
        return f"system:{name}"


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    def chat_json(self, agent: str, system: str, user: str):
        self.prompts[agent] = user
        if agent == "model_dataset_agent":
            return {"models": [], "datasets": [], "dataset_shortlist_top3": []}
        if agent == "metric_agent":
            return {"metrics": [], "benchmark_implications": []}
        if agent == "critic_agent":
            return {"warnings": [], "dataset_decisions": []}
        return {
            "all_candidate_models": [],
            "benchmark_ready_models": [],
            "models": [],
            "repositories": [],
            "datasets": [],
            "dataset_links": [],
            "model_dataset_links": [],
            "dataset_followup_tasks": [],
            "meeting_recommended_datasets": [],
            "meeting_dataset_decision_trace": [],
            "metrics": [],
            "papers": [],
            "benchmark_implications": [],
            "open_questions": [],
            "model_classification": [],
            "representative_models_by_category": [],
            "agent_discussion": [],
        }


def _seed_payload() -> dict:
    return {
        "datasets": [
            {
                "dataset_name": "Veltri original test set",
                "aliases": ["Veltri_test", "Veltri_test_out0309_corrected"],
                "linked_models": ["AMP Scanner v2"],
                "source_doi": "10.1093/bioinformatics/bty179",
            },
            {
                "dataset_name": "AMPSorter ProteoGPT benchmarking set",
                "aliases": ["ProteoGPT_all_predictions"],
                "linked_models": ["ProteoGPT", "AMPSorter"],
                "source_doi": "10.5281/zenodo.16633186",
            },
            {
                "dataset_name": "C_AMPs-predict test data",
                "aliases": ["C_AMPs-predict_test"],
                "linked_models": ["C_AMPs-predict"],
                "source_doi": "10.1038/s41587-022-01226-0",
            },
        ]
    }


class LiteratureDatasetCandidateContextTests(unittest.TestCase):
    def test_seed_queries_use_scientific_names_models_and_dois(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_path = Path(tmp) / "seeds.json"
            seed_path.write_text(json.dumps(_seed_payload()), encoding="utf-8")
            with patch.object(literature, "REQUIRED_DATASET_SEEDS_JSON", seed_path):
                plan = literature.augment_query_plan_with_dataset_seeds({"pubmed": [], "github": []})

        pubmed_query = plan["pubmed"][-1]["query"]
        github_query = plan["github"][-1]["query"]
        self.assertIn("C_AMPs-predict", pubmed_query)
        self.assertIn("ProteoGPT", pubmed_query)
        self.assertIn("AMP Scanner v2", github_query)
        self.assertIn("10.1038/s41587-022-01226-0", pubmed_query)
        self.assertNotIn("out0309_corrected", pubmed_query)

    def test_local_result_name_is_mapped_to_seed_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            seed_path = data_dir / "seeds.json"
            result_dir = data_dir / "results_manual" / "Veltri_test"
            result_dir.mkdir(parents=True)
            seed_path.write_text(json.dumps(_seed_payload()), encoding="utf-8")
            with (result_dir / "final_results_with_predictions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "label", "score"])
                writer.writerows([["p1", 1, 0.9], ["p2", 1, 0.8], ["n1", 0, 0.2], ["n2", 0, 0.1]])

            with (
                patch.object(literature, "DATA_DIR", data_dir),
                patch.object(literature, "REQUIRED_DATASET_SEEDS_JSON", seed_path),
            ):
                profiles = literature.load_local_evaluated_dataset_profiles()

        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["matched_seed_dataset_name"], "Veltri original test set")
        self.assertEqual(profiles[0]["observed_profile"], "balanced")
        self.assertIn("not_primary_literature_evidence", profiles[0]["evidence_scope"])

    def test_every_meeting_agent_receives_candidates_as_nonfixed_evidence(self):
        candidates = [{"dataset_name": "C_AMPs-predict test data", "source_doi": "10.1038/test"}]
        profiles = [{"local_dataset_name": "C_AMPs-predict_test", "observed_profile": "imbalanced"}]
        llm = _RecordingLLM()
        compact_pool = {
            "created_at": "test",
            "chunk_summaries": [],
            "paper_overview": [],
            "source_counts": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "meeting.jsonl"
            with (
                patch.object(literature, "load_verified_dataset_acquisition_candidates", return_value=candidates),
                patch.object(literature, "load_local_evaluated_dataset_profiles", return_value=profiles),
                patch.object(literature, "GLOBAL_MEETING_RAW_JSONL", raw_path),
            ):
                _, raw = literature.global_meeting(llm, _Loader(), compact_pool, {})

        for agent in ("model_dataset_agent", "metric_agent", "critic_agent", "chief_agent"):
            self.assertIn("C_AMPs-predict test data", llm.prompts[agent])
        self.assertIn("不是固定推荐名单", llm.prompts["model_dataset_agent"])
        self.assertIn("不得自动入选", llm.prompts["chief_agent"])
        self.assertEqual(raw["dataset_candidate_context"]["verified_acquisition_candidates"], candidates)

    def test_alias_keys_merge_local_and_scientific_dataset_names(self):
        keys = _candidate_name_keys(
            {
                "dataset_name": "Veltri / DAMP original test set",
                "aliases": ["Veltri_test", "AMP Scanner v2 test set"],
            }
        )
        self.assertIn("veltri damp original", keys)
        self.assertIn("veltri test", keys)
        self.assertIn("amp scanner v2", keys)


if __name__ == "__main__":
    unittest.main()
