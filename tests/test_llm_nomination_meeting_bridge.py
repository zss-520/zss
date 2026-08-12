from __future__ import annotations

import unittest

from deep_research_literature_agent import (
    build_chunk_derived_final,
    build_llm_nomination_meeting_context,
)


class LlmNominationMeetingBridgeTests(unittest.TestCase):
    def test_only_verified_eligible_nomination_becomes_meeting_evidence(self):
        nominations = {"models": [{"model_name": "GoodAMP"}, {"model_name": "HallucinatedAMP"}]}
        verification = {
            "results": [
                {
                    "model_name": "GoodAMP",
                    "verification_status": "verified",
                    "eligible_for_evidence_pool": True,
                    "source_doi": "10.0000/good",
                    "code_repository_url": "https://github.com/example/goodamp",
                    "best_match_score": 0.95,
                },
                {
                    "model_name": "HallucinatedAMP",
                    "verification_status": "rejected_no_matching_paper",
                    "eligible_for_evidence_pool": False,
                },
            ]
        }

        bridge = build_llm_nomination_meeting_context(nominations, verification)
        derived = build_chunk_derived_final({"chunk_summaries": [], "llm_nomination_verification": bridge})

        self.assertEqual(bridge["nominated_count"], 2)
        self.assertEqual(bridge["verified_count"], 1)
        self.assertEqual([row["model_name"] for row in bridge["verified_models"]], ["GoodAMP"])
        names = [row.get("model_name") for row in derived["all_candidate_models"]]
        self.assertIn("GoodAMP", names)
        self.assertNotIn("HallucinatedAMP", names)
        self.assertEqual(derived["benchmark_ready_models"][0]["model_name"], "GoodAMP")


if __name__ == "__main__":
    unittest.main()
