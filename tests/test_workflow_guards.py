import unittest

from workflow_guards import model_readiness_issues, require_models_ready


class WorkflowGuardTests(unittest.TestCase):
    def test_ready_and_smoke_passed_is_allowed(self):
        models = [{"model_name": "A", "hpc_env_status": "ready", "hpc_smoke_test": "passed"}]
        self.assertEqual(model_readiness_issues(models), [])
        require_models_ready(models)

    def test_unverified_model_is_blocked_by_default(self):
        models = [{"model_name": "A", "hpc_env_status": "ready", "hpc_smoke_test": "unknown"}]
        with self.assertRaisesRegex(RuntimeError, "A"):
            require_models_ready(models)

    def test_explicit_diagnostic_override_is_allowed(self):
        models = [{"model_name": "A", "hpc_env_status": "failed", "hpc_smoke_test": "failed"}]
        require_models_ready(models, allow_unverified=True)


if __name__ == "__main__":
    unittest.main()
