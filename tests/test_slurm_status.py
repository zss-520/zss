import sys
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from workflow_utils import parse_sacct_output, slurm_job_succeeded, wait_for_slurm_job


class SlurmStatusTests(unittest.TestCase):
    def test_parse_exact_allocation_row(self):
        text = "123|COMPLETED|0:0|00:03:12|1200M\n123.batch|COMPLETED|0:0|00:03:11|900M\n"
        record = parse_sacct_output(text, "123")
        self.assertEqual(record["job_id"], "123")
        self.assertEqual(record["state"], "COMPLETED")
        self.assertTrue(slurm_job_succeeded(record))

    def test_nonzero_exit_is_not_success(self):
        record = parse_sacct_output("42|COMPLETED|1:0|00:00:02|10M\n", "42")
        self.assertFalse(slurm_job_succeeded(record))

    def test_wait_requires_sacct_success(self):
        def remote(_ssh, command, stream=False):
            if command.startswith("squeue"):
                return "", ""
            if command.startswith("sacct"):
                return "88|COMPLETED|0:0|00:01:00|1G\n", ""
            raise AssertionError(command)

        with patch("workflow_utils.read_remote_text", side_effect=remote):
            record = wait_for_slurm_job(
                object(),
                "88",
                poll_seconds=0,
                timeout_seconds=10,
                accounting_retries=1,
            )
        self.assertEqual(record["exit_code"], "0:0")

    def test_failed_accounting_record_raises(self):
        def remote(_ssh, command, stream=False):
            if command.startswith("squeue"):
                return "", ""
            return "99|OUT_OF_MEMORY|0:125|00:02:00|40G\n", ""

        with patch("workflow_utils.read_remote_text", side_effect=remote):
            with self.assertRaisesRegex(RuntimeError, "OUT_OF_MEMORY"):
                wait_for_slurm_job(
                    object(),
                    "99",
                    poll_seconds=0,
                    timeout_seconds=10,
                    accounting_retries=1,
                )

    def test_timeout_cancels_job(self):
        commands = []

        def remote(_ssh, command, stream=False):
            commands.append(command)
            return "", ""

        clock = iter([0.0, 11.0])
        with patch("workflow_utils.read_remote_text", side_effect=remote):
            with self.assertRaises(TimeoutError):
                wait_for_slurm_job(
                    object(),
                    "101",
                    poll_seconds=0,
                    timeout_seconds=10,
                    monotonic_fn=lambda: next(clock),
                )
        self.assertIn("scancel 101", commands)


if __name__ == "__main__":
    unittest.main()
