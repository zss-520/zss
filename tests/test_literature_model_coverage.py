from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import deep_research_literature_agent as literature


class LiteratureModelCoverageTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        path = root / "targets.json"
        path.write_text(
            json.dumps(
                {
                    "minimum_coverage_fraction": 0.66,
                    "models": [
                        {
                            "model_name": "C_AMPs-predict",
                            "aliases": ["c_AMPs-prediction"],
                            "required_core": True,
                            "count_toward_coverage": True,
                        },
                        {
                            "model_name": "AMPSorter",
                            "aliases": ["AMPsorter", "ProteoGPT AMPSorter"],
                            "required_core": True,
                            "count_toward_coverage": True,
                        },
                        {
                            "model_name": "PepNet",
                            "aliases": ["pepnet_fast", "pepnet_standard"],
                            "search_terms": ["PepNet", "Pep-Net"],
                            "required_core": False,
                            "count_toward_coverage": True,
                        },
                        {
                            "model_name": "LSTM evaluation baseline",
                            "aliases": ["lstm"],
                            "identity_status": "generic_internal_baseline_not_unique_literature_model",
                            "count_toward_coverage": False,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_verification(self, root: Path) -> Path:
        path = root / "verification.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "model_name": name,
                            "canonical_name": name,
                            "verification_status": "scientifically_verified",
                            "eligible_for_evidence_pool": True,
                            "task_type": "AMP binary prediction/classification",
                            "code_repository_url": f"https://github.com/example/{name}",
                            "source_doi": doi,
                            "nomination": {"model_name": name},
                        }
                        for name, doi in [
                            ("C_AMPs-predict", "10.0000/camps"),
                            ("AMPSorter", "10.0000/sorter"),
                        ]
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_queries_cover_canonical_models_but_not_generic_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._write_config(Path(tmp))
            with patch.object(literature, "BENCHMARK_MODEL_COVERAGE_TARGETS_JSON", config):
                plan = literature.augment_query_plan_with_model_coverage_targets({"pubmed": []})

        query = " ".join(row["query"] for row in plan["pubmed"])
        self.assertIn("C_AMPs-predict", query)
        self.assertIn("AMPSorter", query)
        self.assertIn("PepNet", query)
        self.assertIn("Pep-Net", query)
        self.assertNotIn("LSTM evaluation baseline", query)

    def test_coverage_is_a_gate_not_a_fixed_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_config(root)
            verification = self._write_verification(root)
            with (
                patch.object(literature, "BENCHMARK_MODEL_COVERAGE_TARGETS_JSON", config),
                patch.object(literature, "REQUIRED_BENCHMARK_MODEL_VERIFICATION_JSON", verification),
            ):
                context = literature.build_benchmark_model_coverage_context(
                    {"all_candidate_models": []}, {"historical_model_pool": []}
                )

        self.assertEqual(context["coverage_denominator"], 3)
        self.assertEqual(context["covered_model_count"], 2)
        self.assertTrue(context["coverage_gate_passed"])
        self.assertEqual(context["missing_coverage_models"], ["PepNet"])
        self.assertIn("not a fixed recommendation", context["selection_semantics"])

    def test_required_verified_models_are_real_candidate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            verification = self._write_verification(Path(tmp))
            with patch.object(
                literature, "REQUIRED_BENCHMARK_MODEL_VERIFICATION_JSON", verification
            ):
                models = literature.load_scientifically_verified_required_models()

        self.assertEqual({row["model_name"] for row in models}, {"C_AMPs-predict", "AMPSorter"})
        self.assertTrue(all(row["required_candidate"] for row in models))
        self.assertTrue(all(row["code_repository_url"].startswith("https://github.com/") for row in models))

    def test_explicit_classifier_is_not_excluded_by_generative_parent_paper_title(self):
        row = {
            "model_name": "AMPSorter",
            "task_type": "AMP binary prediction/classification",
            "method_family": "ProteoGPT transformer protein language model classifier",
            "paper_title": "A generative artificial intelligence approach for AMP discovery",
            "code_repository_url": "https://github.com/example/AMP_Project",
            "source_doi": "10.0000/sorter",
            "evidence_level": "primary_publisher_crossref_openalex_github_verified",
            "benchmark_candidate": True,
            "blocking_issues": [],
        }
        self.assertTrue(literature._is_main_amp_binary_candidate(row))
        self.assertTrue(literature._strict_main_deployment_candidate(row))


if __name__ == "__main__":
    unittest.main()
