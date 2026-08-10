import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_manifest import RunManifest, sha256_file


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_dataset_hash_and_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data" / "datasets" / "ds1"
            dataset.mkdir(parents=True)
            ground_truth = dataset / "ground_truth.csv"
            ground_truth.write_text("id,label\na,1\n", encoding="utf-8")
            model = {
                "model_name": "M",
                "hpc_env_status": "ready",
                "hpc_smoke_test": "passed",
                "local_model_dir": "",
            }
            with patch.dict(os.environ, {"AMP_RUN_ID": "unit-test-run"}):
                manifest = RunManifest.start(
                    root=root,
                    models=[model],
                    datasets=[dataset],
                    metric_protocol={"version": "test"},
                    llm_model="test-llm",
                    allow_unverified_models=False,
                )
            manifest.record_dataset("ds1", status="success")
            manifest.finalize("success", successful_datasets=1)

            payload = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["datasets"][0]["files"][0]["sha256"], sha256_file(ground_truth))
            latest = json.loads((root / "data" / "runs" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["run_id"], "unit-test-run")
            self.assertEqual(latest["status"], "success")

    def test_duplicate_explicit_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            kwargs = {
                "root": root,
                "models": [],
                "datasets": [dataset],
                "metric_protocol": {},
                "llm_model": "test",
                "allow_unverified_models": False,
            }
            with patch.dict(os.environ, {"AMP_RUN_ID": "same-run"}):
                first = RunManifest.start(**kwargs)
                first.finalize("success")
                with self.assertRaises(FileExistsError):
                    RunManifest.start(**kwargs)


if __name__ == "__main__":
    unittest.main()
