import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scientific_evaluation import evaluate_prediction_table


class ScientificEvaluationTests(unittest.TestCase):
    def test_validation_threshold_bootstrap_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            validation = root / "validation.csv"
            pd.DataFrame(
                {
                    "Standard_ID": ["a", "b", "c", "d"],
                    "True_Label": [0, 0, 1, 1],
                    "Model-A_Prob": [0.1, 0.4, 0.45, 0.9],
                }
            ).to_csv(predictions, index=False)
            pd.DataFrame(
                {
                    "Standard_ID": ["v1", "v2", "v3", "v4"],
                    "True_Label": [0, 0, 1, 1],
                    "Model-A_Prob": [0.1, 0.4, 0.45, 0.9],
                }
            ).to_csv(validation, index=False)

            result = evaluate_prediction_table(
                predictions,
                root / "out",
                validation_csv=validation,
                expected_models=["Model A"],
                iterations=30,
                seed=7,
            )
            model = result["report"]["models"]["Model-A"]
            self.assertEqual(model["threshold_source"], "validation_max_mcc")
            self.assertAlmostEqual(model["threshold"], 0.45)
            self.assertEqual(result["eval_result"]["Model-A"]["MCC"], 1.0)
            for name in ["scientific_evaluation.json", "scientific_evaluation.md", "eval_result.json"]:
                self.assertTrue((root / "out" / name).exists())
            json.loads((root / "out" / "scientific_evaluation.json").read_text(encoding="utf-8"))

    def test_missing_expected_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.csv"
            pd.DataFrame({"True_Label": [0, 1], "Other_Prob": [0.1, 0.9]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Target"):
                evaluate_prediction_table(
                    path,
                    root / "out",
                    expected_models=["Target"],
                    iterations=1,
                )

    def test_no_validation_uses_preregistered_point_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.csv"
            pd.DataFrame(
                {"True_Label": [0, 1], "Target_Prob": [0.49, 0.51]}
            ).to_csv(path, index=False)
            result = evaluate_prediction_table(
                path,
                root / "out",
                expected_models=["Target"],
                iterations=2,
            )
            item = result["report"]["models"]["Target"]
            self.assertEqual(item["threshold"], 0.5)
            self.assertEqual(
                item["threshold_source"], "diagnostic_fixed_0.5_no_validation_data"
            )

    def test_calibration_utility_pairing_holm_and_cluster_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.csv"
            pd.DataFrame(
                {
                    "Standard_ID": [f"s{i}" for i in range(8)],
                    "Homology_Cluster": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
                    "Sequence": ["ACDEFGHIKL"] * 8,
                    "True_Label": [0, 0, 0, 0, 1, 1, 1, 1],
                    "Good_Prob": [0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99],
                    "Weak_Prob": [0.20, 0.40, 0.60, 0.80, 0.20, 0.40, 0.60, 0.80],
                }
            ).to_csv(path, index=False)

            result = evaluate_prediction_table(path, root / "out", iterations=30, seed=3)

            good = result["report"]["models"]["Good"]["selected_threshold_metrics"]
            self.assertIn("brier_score", good)
            self.assertIn("expected_calibration_error", good)
            self.assertIn("auprc_lift", good["ranking_utility"])
            self.assertEqual(
                result["report"]["models"]["Good"]["bootstrap_resampling_unit"],
                "Homology_Cluster",
            )
            self.assertTrue(result["report"]["pairwise_bootstrap_differences"])
            comparison = result["report"]["pairwise_mcnemar"][0]
            self.assertIn("p_value_holm", comparison)
            self.assertIn("reject_holm_0_05", comparison)
            self.assertIn("10_20_aa", result["report"]["models"]["Good"]["length_subgroup_metrics"])

    def test_formal_run_requires_validation_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "predictions.csv"
            pd.DataFrame(
                {"True_Label": [0, 1], "Target_Prob": [0.1, 0.9]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "requires an independent validation-derived threshold"):
                evaluate_prediction_table(
                    path,
                    root / "out",
                    expected_models=["Target"],
                    iterations=1,
                    require_validation_threshold=True,
                )


if __name__ == "__main__":
    unittest.main()
