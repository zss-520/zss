from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from dataset_gate import (
    dataset_gate_issues,
    generate_dataset_plan,
    run_dataset_gate,
    safe_extract_archive,
)


def _write_standard_dataset(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "ground_truth.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label"])
        writer.writerows(rows)
    with (path / "combined_test.fasta").open("w", encoding="utf-8") as handle:
        for seq_id, sequence, _ in rows:
            handle.write(f">{seq_id}\n{sequence}\n")


def _unique_sequence(index: int, length: int = 10) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    chars = ["A"] * length
    value = index + 1
    for offset in range(length):
        chars[offset] = alphabet[value % len(alphabet)]
        value //= len(alphabet)
    return "".join(chars)


def _profile_rows(start: int, positive: int, negative: int, *, length: int = 10) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for offset in range(positive + negative):
        label = 1 if offset < positive else 0
        rows.append((f"seq_{start + offset}", _unique_sequence(start + offset, length), label))
    return rows


class DatasetGateTests(unittest.TestCase):
    def test_legacy_sequence_label_csv_gets_stable_fasta_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data" / "datasets" / "legacy"
            dataset.mkdir(parents=True)
            (dataset / "ground_truth.csv").write_text(
                "sequence,label\nACDEFG,1\nHIKLMN,0\n",
                encoding="utf-8",
            )
            (dataset / "combined_test.fasta").write_text(
                ">positive_id\nACDEFG\n>negative_id\nHIKLMN\n",
                encoding="utf-8",
            )
            manifest, _ = run_dataset_gate(root)
            self.assertEqual(manifest["status"], "passed")
            header = (dataset / "ground_truth.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, "id,sequence,label")
            self.assertTrue((dataset / "_legacy" / "ground_truth_without_id.csv").is_file())

    def test_plan_bootstraps_existing_local_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_standard_dataset(
                root / "data" / "datasets" / "local_ds",
                [("p", "ACDEFG", 1), ("n", "HIKLMN", 0)],
            )
            plan = generate_dataset_plan(root)
            self.assertEqual(plan["generation_mode"], "local_bootstrap")
            self.assertEqual([row["name"] for row in plan["datasets"]], ["local_ds"])
            self.assertTrue((root / "data" / "dataset_plan.json").is_file())

    def test_complete_gate_writes_manifest_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data" / "datasets" / "ds1"
            _write_standard_dataset(
                dataset,
                [("p", "ACDEFG", 1), ("n", "HIKLMN", 0)],
            )
            manifest, path = run_dataset_gate(root)
            self.assertEqual(manifest["status"], "passed")
            self.assertTrue(path.is_file())
            self.assertEqual(dataset_gate_issues(root, [dataset]), [])

            with (dataset / "combined_test.fasta").open("a", encoding="utf-8") as handle:
                handle.write(">changed\nQRSTVW\n")
            issues = dataset_gate_issues(root, [dataset])
            self.assertTrue(any("发生变化" in issue for issue in issues))

    def test_tofu_source_hash_is_enforced_on_subsequent_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data" / "datasets" / "ds1"
            _write_standard_dataset(
                dataset,
                [("p", "ACDEFG", 1), ("n", "HIKLMN", 0)],
            )
            raw = dataset / "raw_source.txt"
            raw.write_text("original", encoding="utf-8")
            first, _ = run_dataset_gate(root)
            self.assertEqual(first["status"], "passed")
            self.assertTrue((root / "data" / "dataset_source_lock.json").is_file())

            raw.write_text("changed", encoding="utf-8")
            second, _ = run_dataset_gate(root)
            self.assertEqual(second["status"], "failed")
            self.assertIn("SHA256 不匹配", second["datasets"][0]["error"])

    def test_cross_dataset_overlap_blocks_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_standard_dataset(
                root / "data" / "datasets" / "a",
                [("a1", "ACDEFG", 1), ("a0", "HIKLMN", 0)],
            )
            _write_standard_dataset(
                root / "data" / "datasets" / "b",
                [("b1", "ACDEFG", 1), ("b0", "QRSTVW", 0)],
            )
            manifest, _ = run_dataset_gate(root)
            self.assertEqual(manifest["status"], "failed")
            issue_types = {row["type"] for row in manifest["leakage_check"]["issues"]}
            self.assertIn("cross_dataset_overlap", issue_types)

    def test_expected_sha256_mismatch_fails_before_standardization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv"
            source.write_text("sequence,label\nACDEFG,1\nHIKLMN,0\n", encoding="utf-8")
            strategy = {
                "recommended_datasets": [
                    {
                        "dataset_name": "remote_ds",
                        "download_url": str(source),
                        "sha256": "0" * 64,
                    }
                ]
            }
            strategy_path = root / "data" / "benchmark_strategy.json"
            strategy_path.parent.mkdir(parents=True)
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
            manifest, _ = run_dataset_gate(root)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("SHA256 不匹配", manifest["datasets"][0]["error"])

    def test_training_reference_overlap_blocks_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data" / "datasets" / "ds"
            _write_standard_dataset(
                dataset,
                [("p", "ACDEFG", 1), ("n", "HIKLMN", 0)],
            )
            reference = root / "train.fasta"
            reference.write_text(">train\nACDEFG\n", encoding="utf-8")
            strategy = {
                "recommended_datasets": [
                    {
                        "dataset_name": "ds",
                        "download_url": "",
                        "training_reference_paths": ["train.fasta"],
                    }
                ]
            }
            strategy_path = root / "data" / "benchmark_strategy.json"
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8")
            manifest, _ = run_dataset_gate(root)
            self.assertEqual(manifest["status"], "failed")
            issue_types = {row["type"] for row in manifest["leakage_check"]["issues"]}
            self.assertIn("training_reference_overlap", issue_types)

    def test_scientific_selection_accepts_one_balanced_and_two_imbalanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specifications = [
                ("balanced", 7, 5, "balanced", 0),
                ("mild_imbalance", 6, 4, "imbalanced", 100),
                ("strong_imbalance", 8, 2, "imbalanced", 200),
            ]
            for name, positive, negative, _, start in specifications:
                _write_standard_dataset(
                    root / "data" / "datasets" / name,
                    _profile_rows(start, positive, negative),
                )
            strategy = {
                "dataset_selection_policy": {
                    "enabled": True,
                    "required_dataset_count": 3,
                    "required_profiles": {"balanced": 1, "imbalanced": 2},
                    "balanced_minority_majority_ratio_min": 0.7,
                    "min_imbalanced_minority_fraction_gap": 0.1,
                    "min_total_samples": 2,
                    "min_samples_per_class": 1,
                    "min_primary_length_fraction": 0.8,
                    "min_primary_length_fraction_per_class": 0.8,
                },
                "recommended_datasets": [
                    {"dataset_name": name, "selection_profile": profile}
                    for name, _, _, profile, _ in specifications
                ],
            }
            strategy_path = root / "data" / "benchmark_strategy.json"
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

            manifest, _ = run_dataset_gate(root)

            self.assertEqual(manifest["status"], "passed")
            selection = manifest["dataset_selection_check"]
            self.assertEqual(selection["status"], "passed")
            self.assertEqual(selection["observed_profile_counts"], {"balanced": 1, "imbalanced": 2})
            for dataset in manifest["datasets"]:
                length_stats = dataset["standardized"]["length_distribution"]["overall"]
                self.assertEqual(length_stats["fraction_10_50_aa"], 1.0)

    def test_scientific_selection_rejects_low_primary_length_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specifications = [
                ("balanced", 7, 5, "balanced", 0, 6),
                ("mild_imbalance", 6, 4, "imbalanced", 100, 10),
                ("strong_imbalance", 8, 2, "imbalanced", 200, 10),
            ]
            for name, positive, negative, _, start, length in specifications:
                _write_standard_dataset(
                    root / "data" / "datasets" / name,
                    _profile_rows(start, positive, negative, length=length),
                )
            strategy = {
                "dataset_selection_policy": {
                    "enabled": True,
                    "required_dataset_count": 3,
                    "required_profiles": {"balanced": 1, "imbalanced": 2},
                    "balanced_minority_majority_ratio_min": 0.7,
                    "min_imbalanced_minority_fraction_gap": 0.1,
                    "min_total_samples": 2,
                    "min_samples_per_class": 1,
                    "min_primary_length_fraction": 0.8,
                    "min_primary_length_fraction_per_class": 0.8,
                },
                "recommended_datasets": [
                    {"dataset_name": name, "selection_profile": profile}
                    for name, _, _, profile, _, _ in specifications
                ],
            }
            strategy_path = root / "data" / "benchmark_strategy.json"
            strategy_path.write_text(json.dumps(strategy), encoding="utf-8")

            manifest, _ = run_dataset_gate(root)

            self.assertEqual(manifest["status"], "failed")
            issue_types = {row["type"] for row in manifest["dataset_selection_check"]["issues"]}
            self.assertIn("primary_length_coverage_below_threshold", issue_types)

    def test_safe_extract_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "no")
            with self.assertRaisesRegex(RuntimeError, "路径穿越"):
                safe_extract_archive(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
