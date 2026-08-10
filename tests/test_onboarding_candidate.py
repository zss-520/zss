import unittest

from new_model_onboarding import candidate_from_row


class OnboardingCandidateTests(unittest.TestCase):
    def test_portfolio_fields_survive_candidate_normalization(self):
        row = {
            "model_name": "CG-AMP",
            "code_repository_url": "https://github.com/ghli16/CG-AMP",
            "publication_year": 2025,
            "benchmark_role": "recent_sota_candidate",
            "benchmark_roles": ["recent_sota_candidate", "architecture_representative"],
            "benchmark_role_label": "近期 SOTA 候选",
            "benchmark_role_reason": "external tests",
            "architecture_category": "cnn_dominant_models",
            "citation_count": 5,
        }
        candidate = candidate_from_row(row, "test")
        self.assertEqual(candidate["benchmark_role"], "recent_sota_candidate")
        self.assertEqual(candidate["publication_year"], 2025)
        self.assertEqual(candidate["architecture_category"], "cnn_dominant_models")
        self.assertEqual(candidate["citation_count"], 5)


if __name__ == "__main__":
    unittest.main()
