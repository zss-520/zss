from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import hpc_model_ops as hpc
import new_model_onboarding as onboarding


class PromptContractEnforcementTests(unittest.TestCase):
    def test_registry_validator_keeps_authoritative_identity_and_filters_agent_payload(self) -> None:
        onboarding.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        repo = Path(tempfile.mkdtemp(prefix="contract_", dir=onboarding.MODELS_DIR))
        try:
            (repo / "predict.py").write_text("print('ok')\n", encoding="utf-8")
            evidence = {"model_name": "EvidenceModel", "code_repository_url": "https://github.com/example/model"}
            fallback = onboarding.heuristic_registry(evidence, repo)
            candidate = {
                "model_name": "InventedName",
                "local_model_dir": "C:/outside",
                "dependencies": ["json", "sklearn==1.4", "https://bad.example/pkg.whl"],
                "env_setup_commands": ["sudo pip install bad", "python -m pip install numpy"],
                "inference_cmd_template": "rm -rf data",
                "skip_env_setup": True,
                "agent_registry_confidence": 7,
            }
            row = onboarding.validate_registry_record(candidate, fallback, evidence, repo)
            self.assertEqual(row["model_name"], "EvidenceModel")
            self.assertEqual(row["repo_url"], evidence["code_repository_url"])
            self.assertEqual(row["dependencies"], ["scikit-learn==1.4"])
            self.assertEqual(row["env_setup_commands"], ["python -m pip install numpy"])
            self.assertIn("{fasta_path}", row["inference_cmd_template"])
            self.assertFalse(row["skip_env_setup"])
            self.assertEqual(row["agent_registry_confidence"], 1.0)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_self_heal_validator_limits_commands_dependencies_and_registry_updates(self) -> None:
        model = {"model_name": "Demo", "dependencies": ["numpy"], "inference_cmd_template": "python predict.py -i {fasta_path} -o {output_dir}"}
        plan = hpc.validate_self_heal_plan(
            model,
            {
                "diagnosis": "missing dependency",
                "pip_install": ["json", "sklearn", "numpy"],
                "conda_install": ["cudatoolkit=11.3", "pkg; rm -rf data"],
                "env_setup_commands": ["sudo pip install bad", "python -m pip install torch"],
                "registry_updates": {
                    "python_version": "3.9",
                    "inference_cmd_template": "python fixed.py --input {fasta_path}",
                    "repo_url": "https://attacker.invalid/repo",
                },
                "retry_smoke": True,
            },
        )
        self.assertNotIn("json", plan["pip_install"])
        self.assertIn("scikit-learn", plan["pip_install"])
        self.assertEqual(plan["conda_install"], ["cudatoolkit=11.3"])
        self.assertEqual(plan["env_setup_commands"], ["python -m pip install torch"])
        self.assertEqual(plan["registry_updates"], {"python_version": "3.9"})


if __name__ == "__main__":
    unittest.main()
