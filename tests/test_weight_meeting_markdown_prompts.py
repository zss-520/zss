from __future__ import annotations

import unittest

import llm_agent_weight_meeting as meeting
from agent_md_loader import AgentMDLoader


class WeightMeetingMarkdownPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = AgentMDLoader(meeting.WEIGHT_PROMPT_DIR)

    def test_runtime_constants_match_markdown_sources(self) -> None:
        shared = self.loader.load_composed("shared_system").strip()
        reviewer = self.loader.load_composed("reviewer_agent").strip()
        chief = self.loader.load_composed("chief_agent").strip()

        self.assertEqual(meeting.SHARED_SYSTEM, shared)
        self.assertEqual(
            meeting.ROLE_INSTRUCTIONS["literature_agent"],
            self.loader.load_composed("literature_agent").strip(),
        )
        self.assertEqual(
            meeting.ROLE_INSTRUCTIONS["statistics_agent"],
            self.loader.load_composed("statistics_agent").strip(),
        )
        self.assertEqual(
            meeting.ROLE_INSTRUCTIONS["screening_agent"],
            self.loader.load_composed("screening_agent").strip(),
        )
        self.assertEqual(meeting.REVIEWER_SYSTEM, shared + "\n\n" + reviewer)
        self.assertEqual(meeting.CHIEF_SYSTEM, shared + "\n\n" + chief)

    def test_prompt_builders_still_append_dynamic_round_context(self) -> None:
        payload = {
            "exact_metric_keys": ["AUPRC", "MCC"],
            "previous_accepted_weights": {"AUPRC": 0.6, "MCC": 0.4},
            "blinding": "No model identities supplied.",
        }
        prompt = meeting.expert_prompt("literature_agent", payload)
        self.assertIn("Literature Evidence Agent", prompt)
        self.assertIn('"AUPRC": "number"', prompt)
        self.assertIn("No model identities supplied", prompt)


if __name__ == "__main__":
    unittest.main()
