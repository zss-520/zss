from __future__ import annotations

import unittest
from unittest.mock import patch

import deep_research_literature_agent as literature


def _row(name: str, score: float) -> dict:
    return {
        "model_name": name,
        "canonical_name": name,
        "score": score,
    }


class LiteratureTop20ContinuityTests(unittest.TestCase):
    def _stabilize(self, ranked: list[dict], previous: list[dict], max_models: int = 2):
        with patch.object(
            literature,
            "final_deployment_selection_score",
            side_effect=lambda row: float(row["score"]),
        ):
            return literature.stabilize_top20_membership(
                ranked,
                ranked,
                previous,
                max_models=max_models,
            )

    def test_lower_or_equal_challenger_cannot_replace_incumbent(self):
        previous = [_row("incumbent-a", 10), _row("incumbent-b", 9)]
        selected, audit = self._stabilize(
            previous + [_row("lower", 8), _row("equal", 9)],
            previous,
        )

        self.assertEqual({row["model_name"] for row in selected}, {"incumbent-a", "incumbent-b"})
        self.assertEqual(audit["replacement_decisions"], [])
        self.assertFalse(audit["equal_score_replaces_incumbent"])

    def test_strictly_higher_challenger_replaces_weakest_incumbent(self):
        previous = [_row("incumbent-a", 10), _row("incumbent-b", 9)]
        selected, audit = self._stabilize(
            previous + [_row("challenger", 9.1)],
            previous,
        )

        self.assertEqual({row["model_name"] for row in selected}, {"incumbent-a", "challenger"})
        self.assertEqual(audit["replacement_decisions"][0]["replaced_model"], "incumbent-b")
        self.assertGreater(audit["replacement_decisions"][0]["score_improvement"], 0)

    def test_ineligible_incumbent_creates_a_fillable_vacancy(self):
        previous = [_row("incumbent-a", 10), _row("missing-incumbent", 9)]
        selected, audit = self._stabilize(
            [_row("incumbent-a", 10), _row("replacement", 7)],
            previous,
        )

        self.assertEqual({row["model_name"] for row in selected}, {"incumbent-a", "replacement"})
        self.assertEqual(audit["ineligible_previous_names"], ["missing-incumbent"])
        self.assertEqual(
            audit["replacement_decisions"][0]["reason"],
            "filled_vacancy_after_incumbent_became_ineligible_or_list_was_incomplete",
        )


if __name__ == "__main__":
    unittest.main()
