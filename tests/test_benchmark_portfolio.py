import unittest

from benchmark_portfolio import build_benchmark_portfolio, publication_year


def model(name, year, architecture, score, *, code=True, reason=""):
    return {
        "model_name": name,
        "publication_year": year,
        "architecture_category": architecture,
        "deployment_selection_score": score,
        "task_type": "AMP prediction",
        "code_repository_url": f"https://github.com/example/{name}" if code else "",
        "candidate_reason": reason,
    }


class BenchmarkPortfolioTests(unittest.TestCase):
    def test_hard_covers_classics_and_recent_sota_candidates(self):
        rows = [
            model("Macrel", 2020, "machine_learning_models", 3),
            model("amPEPpy", 2020, "machine_learning_models", 2),
            model("AMPlify", 2021, "rnn_lstm_dominant_models", 4),
            model("CG-AMP", 2025, "cnn_dominant_models", 8),
            model("deepAMPNet", 2024, "gnn_models", 7),
            model("UniproLcad", 2024, "transformer_llm_dominant_models", 6),
            model("Hybrid", 2023, "cnn_rnn_hybrid_models", 5),
            model("Ensemble", 2023, "pipeline_or_ensemble_frameworks", 5),
        ]
        portfolio = build_benchmark_portfolio(rows, current_year=2026, max_models=10)
        self.assertEqual(portfolio["role_counts"]["classic_baseline"], 3)
        self.assertEqual(portfolio["role_counts"]["recent_sota_candidate"], 3)
        names = [row["model_name"] for row in portfolio["selected_models"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("CG-AMP", names)
        self.assertIn("Macrel", names)

    def test_self_claim_without_code_is_not_recent_sota_candidate(self):
        rows = [
            model(
                "PaperOnly",
                2025,
                "transformer_llm_dominant_models",
                100,
                code=False,
                reason="Outperforms SOTA on an independent external test",
            )
        ]
        portfolio = build_benchmark_portfolio(rows, current_year=2026, max_models=5)
        self.assertEqual(portfolio["role_counts"]["recent_sota_candidate"], 0)
        self.assertTrue(any(gap["type"] == "recent_sota_candidate_shortfall" for gap in portfolio["gaps"]))

    def test_recent_non_watchlist_needs_sota_and_external_evidence(self):
        rows = [
            model(
                "NewExternalModel",
                2025,
                "cnn_dominant_models",
                1,
                reason="Outperforms state-of-the-art methods on two independent external datasets",
            )
        ]
        portfolio = build_benchmark_portfolio(
            rows,
            current_year=2026,
            max_models=5,
            recent_sota_min=1,
            classic_min=0,
        )
        self.assertEqual(portfolio["role_counts"]["recent_sota_candidate"], 1)

    def test_blocking_no_code_is_not_treated_as_runnable_code(self):
        row = model("PepNet", 2024, "transformer_llm_dominant_models", 10)
        row["code_repository_url"] = "https://zenodo.org/records/123"
        row["blocking_issues"] = ["no code, no details"]
        portfolio = build_benchmark_portfolio(
            [row],
            current_year=2026,
            max_models=5,
            recent_sota_min=1,
            classic_min=0,
        )
        self.assertEqual(portfolio["role_counts"]["recent_sota_candidate"], 0)

    def test_publication_year_parses_date_fields(self):
        self.assertEqual(publication_year({"publication_date": "2024-09-28"}), 2024)
        self.assertIsNone(publication_year({"publication_date": "unknown"}))

    def test_old_non_anchor_does_not_become_classic_baseline(self):
        row = model("ArbitraryOldModel", 2019, "machine_learning_models", 99)
        portfolio = build_benchmark_portfolio(
            [row], current_year=2026, max_models=5, classic_min=1, recent_sota_min=0
        )
        self.assertEqual(portfolio["role_counts"]["classic_baseline"], 0)
        self.assertTrue(any(gap["type"] == "classic_baseline_shortfall" for gap in portfolio["gaps"]))

    def test_known_aliases_deduplicate_and_web_shell_does_not_replace_model(self):
        actual = model("AMP Scanner", 2018, "cnn_rnn_hybrid_models", 1)
        actual["code_repository_url"] = "https://github.com/dan-veltri/amp-scanner-v2"
        web_shell = model("AMPScanner vr.2 web server", 2018, "", 100)
        web_shell["task_type"] = ""
        web_shell["code_repository_url"] = "https://github.com/dan-veltri/amp-scanner-v2"
        portfolio = build_benchmark_portfolio(
            [web_shell, actual],
            current_year=2026,
            max_models=5,
            classic_min=1,
            recent_sota_min=0,
        )
        classics = [
            row for row in portfolio["selected_models"]
            if row.get("benchmark_role") == "classic_baseline"
        ]
        self.assertEqual([row["model_name"] for row in classics], ["AMP Scanner"])

    def test_classic_anchor_year_overrides_later_usage_paper_year(self):
        row = model("AMP Scanner v2", 2025, "cnn_rnn_hybrid_models", 10)
        row["code_repository_url"] = "https://github.com/dan-veltri/amp-scanner-v2"
        portfolio = build_benchmark_portfolio(
            [row], current_year=2026, max_models=5, classic_min=1, recent_sota_min=0
        )
        selected = portfolio["selected_models"][0]
        self.assertEqual(selected["model_name"], "AMP Scanner v2")
        self.assertEqual(selected["publication_year"], 2018)

    def test_verified_required_core_is_included_without_fixing_its_rank(self):
        rows = [
            model("AMPSorter", 2025, "transformer_llm_dominant_models", 0.1),
            model("HigherScoreA", 2023, "cnn_dominant_models", 100),
            model("HigherScoreB", 2023, "gnn_models", 90),
        ]
        portfolio = build_benchmark_portfolio(
            rows,
            current_year=2026,
            max_models=3,
            classic_min=0,
            recent_sota_min=0,
            required_core_names=["AMPSorter"],
        )
        selected = portfolio["selected_models"]
        self.assertIn("AMPSorter", [row["model_name"] for row in selected])
        sorter = next(row for row in selected if row["model_name"] == "AMPSorter")
        self.assertIn("verified_required_core", sorter["benchmark_roles"])
        self.assertEqual(portfolio["role_counts"]["verified_required_core"], 1)
        self.assertEqual(selected[0]["model_name"], "HigherScoreA")


if __name__ == "__main__":
    unittest.main()
