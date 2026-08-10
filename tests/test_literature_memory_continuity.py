from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deep_research_literature_agent as literature


def _model(index: int) -> dict:
    return {
        "model_name": f"AMP model {index:03d}",
        "canonical_name": f"AMP model {index:03d}",
        "task_type": "antimicrobial peptide classification",
        "method_family": "machine learning",
        "code_repository_url": f"https://github.com/example/amp-model-{index:03d}",
        "source_doi": f"10.0000/amp.{index:03d}",
        "evidence_level": "primary_paper_plus_official_repository",
        "confidence": 0.8,
        "benchmark_candidate": True,
    }


class LiteratureMemoryContinuityTests(unittest.TestCase):
    def test_context_keeps_full_programmatic_pool_beyond_prompt_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_path = root / "memory.json"
            index_path = root / "index.json"
            models = [_model(index) for index in range(150)]
            memory_path.write_text(
                json.dumps(
                    {
                        "all_candidate_models": models,
                        "benchmark_ready_models": [],
                        "models": [],
                        "final_deployment_models": models[-20:],
                    }
                ),
                encoding="utf-8",
            )
            index_path.write_text("{}", encoding="utf-8")
            with (
                patch.object(literature, "MEMORY_JSON", memory_path),
                patch.object(literature, "INDEX_JSON", index_path),
            ):
                context = literature.MemoryManager().context()

        self.assertEqual(len(context["historical_model_pool"]), 150)
        self.assertEqual(len(context["all_candidate_models"]), 120)
        self.assertEqual(len(context["previous_final_deployment_models"]), 20)
        self.assertEqual(context["memory_counts"]["all_candidate_models"], 150)

    def test_historical_model_is_merged_before_current_ranking(self):
        remembered = _model(999)
        current = _model(1)
        merged = literature.merge_historical_model_memory(
            {"all_candidate_models": [current], "models": []},
            {
                "historical_model_pool": [remembered],
                "previous_final_deployment_models": [remembered],
            },
        )

        names = {
            row.get("model_name")
            for row in merged["all_candidate_models"]
            if isinstance(row, dict)
        }
        self.assertEqual(names, {"AMP model 001", "AMP model 999"})
        self.assertEqual(merged["memory_continuity"]["historical_model_count"], 1)
        self.assertFalse(merged["memory_continuity"]["historical_models_are_fixed_winners"])

    def test_historical_pool_order_is_deterministic(self):
        memory = {"all_candidate_models": [_model(2), _model(1), _model(3)]}
        first = [row["model_name"] for row in literature.build_historical_model_pool(memory)]
        second = [row["model_name"] for row in literature.build_historical_model_pool(memory)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
