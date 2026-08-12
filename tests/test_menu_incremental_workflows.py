import tempfile
import unittest
from pathlib import Path
from unittest import mock

import amp_benchmark_menu as menu


class MenuIncrementalWorkflowTests(unittest.TestCase):
    def test_meeting_only_uses_existing_evidence_without_search_or_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "compact_evidence_pool.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(menu, "DATA_DIR", data_dir), mock.patch.object(
                menu, "_run", return_value=0
            ) as run:
                self.assertEqual(menu._meeting_only_no_search(), 0)
            command = run.call_args[0][0]
            self.assertIn("--resume-global-only", command)
            self.assertIn("--no-github-enrichment", command)
            self.assertNotIn("--use-existing-meeting", command)

    def test_dataset_recommendation_refresh_never_calls_gate(self):
        with mock.patch.object(menu, "_run", return_value=0) as run:
            self.assertEqual(menu._refresh_dataset_recommendation(), 0)
        command = run.call_args[0][0]
        self.assertEqual(command[-1], "recommend")
        self.assertIn("dataset_recommendation_agent.py", command[-2])
        self.assertNotIn("dataset_gate.py", " ".join(command))

    def test_latest_search_uses_current_and_previous_year(self):
        with mock.patch.object(menu, "_current_year", return_value=2026), mock.patch.object(
            menu, "_recent_search", return_value=0
        ) as search:
            self.assertEqual(menu._latest_model_incremental_search(), 0)
        self.assertEqual(search.call_args[0][:2], (2025, 2026))


if __name__ == "__main__":
    unittest.main()
