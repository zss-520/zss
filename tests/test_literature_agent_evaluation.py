import json
import tempfile
import unittest
from pathlib import Path

from literature_agent_evaluation import evaluate_literature_agent


class LiteratureAgentEvaluationTests(unittest.TestCase):
    def test_wrong_model_detection_and_final_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_path = root / "memory.json"
            gold_path = root / "gold.json"
            valid = {
                "model_name": "ValidClassifier",
                "task_type": "AMP binary prediction/classification",
                "paper_title": "An antimicrobial peptide prediction model",
                "source_doi": "10.1000/valid",
                "code_repository_url": "https://github.com/example/valid",
                "evidence_level": "fulltext",
                "confidence": 1.0,
                "benchmark_candidate": True,
            }
            invalid = {
                "model_name": "WrongTask",
                "task_type": "minor histocompatibility antigen prediction",
                "source_doi": "10.1000/wrong",
                "code_repository_url": "https://github.com/example/wrong",
                "evidence_level": "fulltext",
                "confidence": 1.0,
                "benchmark_candidate": False,
                "deployment_eligible": False,
                "blocking_issues": ["out_of_scope_for_AMP_benchmark"],
            }
            memory_path.write_text(json.dumps({
                "all_candidate_models": [valid, invalid],
                "models": [],
                "benchmark_ready_models": [valid, invalid],
                "final_deployment_models": [valid],
            }), encoding="utf-8")
            gold_path.write_text(json.dumps({"labels": [
                {
                    "model_name": "ValidClassifier",
                    "gold_label": "eligible_main_amp_binary",
                    "expected_primary_metadata": {"source_doi": "10.1000/valid"},
                },
                {
                    "model_name": "WrongTask",
                    "gold_label": "ineligible_main_amp_binary",
                    "expected_primary_metadata": {"source_doi": "10.1000/wrong"},
                },
            ]}), encoding="utf-8")

            report = evaluate_literature_agent(memory_path, gold_path)
            metrics = report["metrics"]
            self.assertEqual(metrics["valid_model_retention_rate"], 1.0)
            self.assertEqual(metrics["wrong_model_detection_recall"], 1.0)
            self.assertEqual(metrics["wrong_model_leakage_rate"], 0.0)
            self.assertEqual(metrics["primary_metadata_field_accuracy"], 1.0)
            self.assertEqual(metrics["final_deployment_contamination_rate"], 0.0)
            self.assertEqual(report["confusion_matrix"]["tn_invalid_rejected"], 1)
            census = report["meeting_screening_census"]
            self.assertEqual(census["total_unique_models_retrieved"], 2)
            self.assertEqual(census["meeting_valid_models"], 1)
            self.assertEqual(census["meeting_misretrieval_or_out_of_scope_models"], 1)
            self.assertEqual(census["meeting_valid_ratio"], 0.5)
            self.assertEqual(census["meeting_valid_percent"], 50.0)
            self.assertEqual(len(report["screening_decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
