"""Reproducible dataset acquisition and readiness gate for AMP benchmarks.

The gate is intentionally independent from the benchmark runner.  It creates a
dataset plan, acquires sources, verifies or records SHA256 digests, safely
extracts archives, standardizes datasets, checks exact-sequence leakage, and
writes an immutable manifest consumed by ``main.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
STANDARD_FILES = ("ground_truth.csv", "combined_test.fasta")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _safe_name(value: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value.strip())
    return name.replace(" ", "_") or "Unknown_Dataset"


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _remove_sha256_prefix(value: Any) -> str:
    text = str(value).strip()
    return text[7:] if text.lower().startswith("sha256:") else text


def _file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"JSON 读取失败: {path}: {exc}") from exc


def generate_dataset_plan(
    root: Path,
    *,
    strategy_path: Path | None = None,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    """Generate a normalized plan from benchmark_strategy or local datasets."""
    root = root.resolve()
    strategy_path = strategy_path or root / "data" / "benchmark_strategy.json"
    plan_path = plan_path or root / "data" / "dataset_plan.json"
    strategy = _load_json(strategy_path, {})
    recommended = strategy.get("recommended_datasets", []) if isinstance(strategy, dict) else []
    selection_policy = strategy.get("dataset_selection_policy", {}) if isinstance(strategy, dict) else {}
    datasets_dir = root / "data" / "datasets"
    local_dirs = (
        {path.name: path for path in datasets_dir.iterdir() if path.is_dir()}
        if datasets_dir.exists()
        else {}
    )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(recommended, list):
        for raw in recommended:
            if not isinstance(raw, dict):
                continue
            name = _safe_name(str(raw.get("dataset_name") or raw.get("name") or ""))
            if name in seen:
                continue
            seen.add(name)
            checksum = raw.get("sha256") or raw.get("expected_sha256") or raw.get("checksum")
            references = (
                raw.get("leakage_reference_paths")
                or raw.get("training_reference_paths")
                or []
            )
            rows.append(
                {
                    "name": name,
                    "source_url": str(raw.get("download_url") or raw.get("source_url") or "").strip(),
                    "expected_sha256": checksum,
                    "description": str(raw.get("description") or "").strip(),
                    "role": str(raw.get("role") or "benchmark_test").strip(),
                    "leakage_reference_paths": references if isinstance(references, list) else [references],
                    "selection_profile": str(raw.get("selection_profile") or "").strip(),
                    "citation": raw.get("citation"),
                    "dataset_version": str(raw.get("dataset_version") or "").strip(),
                    "retrieved_at_utc": str(raw.get("retrieved_at_utc") or "").strip(),
                    "license": str(raw.get("license") or "").strip(),
                    "independent_external_test": raw.get("independent_external_test"),
                    "label_definition": str(raw.get("label_definition") or "").strip(),
                    "negative_sampling_strategy": str(raw.get("negative_sampling_strategy") or "").strip(),
                    "homology_report_path": str(raw.get("homology_report_path") or "").strip(),
                }
            )

    # A formal strategy is an allow-list by default.  Local directories are
    # auto-added only during bootstrap, or when explicitly requested.
    include_unlisted = bool(strategy.get("include_unlisted_local_datasets", not recommended)) if isinstance(strategy, dict) else True
    for name in sorted(local_dirs, key=str.casefold) if include_unlisted else []:
        if name not in seen:
            rows.append(
                {
                    "name": name,
                    "source_url": "",
                    "expected_sha256": None,
                    "description": "Existing local dataset discovered during plan generation.",
                    "role": "benchmark_test",
                    "leakage_reference_paths": [],
                    "selection_profile": "",
                    "citation": None,
                    "dataset_version": "",
                    "retrieved_at_utc": "",
                    "license": "",
                    "independent_external_test": None,
                    "label_definition": "",
                    "negative_sampling_strategy": "",
                    "homology_report_path": "",
                }
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "strategy_path": _relative_or_absolute(strategy_path, root),
        "generation_mode": "benchmark_strategy" if recommended else "local_bootstrap",
        "dataset_selection_policy": selection_policy if isinstance(selection_policy, dict) else {},
        "datasets": rows,
    }
    _atomic_json(plan_path, payload)
    return payload


def _strip_git_subpath(url: str) -> tuple[str, str | None]:
    match = re.search(r"/(?:tree|blob)/([^/]+)(?:/.*)?$", url)
    branch = match.group(1) if match else None
    clean = re.sub(r"/(?:tree|blob)/.*$", "", url)
    return clean, branch


def _download_http(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AMP-benchmark-dataset-gate/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _download_zenodo(url: str, download_dir: Path) -> list[Path]:
    match = re.search(r"zenodo\.org/(?:records?|record)/(\d+)", url)
    if not match:
        raise RuntimeError(f"无法从 Zenodo URL 提取 record id: {url}")
    api_url = f"https://zenodo.org/api/records/{match.group(1)}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "AMP-benchmark-dataset-gate/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.loads(response.read().decode("utf-8"))
    outputs: list[Path] = []
    for entry in record.get("files", []):
        link = (entry.get("links") or {}).get("self")
        key = Path(str(entry.get("key") or "download.bin")).name
        if not link:
            continue
        destination = download_dir / key
        _download_http(str(link), destination)
        outputs.append(destination)
    if not outputs:
        raise RuntimeError(f"Zenodo record 没有可下载文件: {url}")
    return outputs


def _source_files(dataset_dir: Path) -> list[Path]:
    ignored_roots = {"_extracted", "_legacy"}
    files: list[Path] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.name in STANDARD_FILES or path.name == "ambiguous.csv":
            continue
        rel = path.relative_to(dataset_dir)
        if rel.parts and rel.parts[0] in ignored_roots:
            continue
        if ".git" in rel.parts or path.name.startswith("."):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.as_posix().casefold())


def acquire_dataset(root: Path, row: dict[str, Any]) -> tuple[Path, list[Path], str]:
    dataset_dir = root / "data" / "datasets" / str(row["name"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing = _source_files(dataset_dir)
    if existing or all((dataset_dir / name).exists() for name in STANDARD_FILES):
        return dataset_dir, existing, "existing_local"

    source = str(row.get("source_url") or "").strip()
    if not source:
        raise RuntimeError(f"{row['name']}: 没有 source_url，且本地目录为空")
    download_dir = dataset_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Check native paths before URL parsing because ``C:\\...`` is otherwise
    # interpreted as URL scheme ``c`` on Windows.
    local_candidate = Path(source)
    if local_candidate.exists():
        destination = download_dir / local_candidate.name
        shutil.copy2(local_candidate, destination)
        return dataset_dir, [destination], "local_file"

    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "file":
        local = Path(urllib.request.url2pathname(parsed.path))
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", str(local)):
            local = Path(str(local)[1:])
        destination = download_dir / local.name
        shutil.copy2(local, destination)
        return dataset_dir, [destination], "local_file"
    if "github.com" in source.lower() or "gitee.com" in source.lower():
        clean, branch = _strip_git_subpath(source)
        clone_dir = dataset_dir / "_source_repo"
        command = ["git", "clone", "--depth", "1"]
        if branch:
            command.extend(["--branch", branch])
        command.extend([clean, str(clone_dir)])
        proc = subprocess.run(command, text=True, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone 失败: {(proc.stderr or proc.stdout).strip()[:500]}")
        return dataset_dir, _source_files(dataset_dir), "git_clone"
    if "zenodo.org" in source.lower():
        return dataset_dir, _download_zenodo(source, download_dir), "zenodo_api"
    if parsed.scheme in {"http", "https"}:
        filename = Path(urllib.parse.unquote(parsed.path)).name or "download.bin"
        destination = download_dir / filename
        _download_http(source, destination)
        return dataset_dir, [destination], "http"
    raise RuntimeError(f"不支持的数据集来源: {source}")


def _expected_map(expected: Any, files: list[Path], dataset_dir: Path) -> dict[str, str]:
    if not expected:
        return {}
    if isinstance(expected, str):
        value = _remove_sha256_prefix(expected).strip().lower()
        if len(files) != 1:
            raise RuntimeError("单个 expected_sha256 只能用于恰好一个原始文件；多文件请使用映射")
        return {files[0].relative_to(dataset_dir).as_posix(): value}
    if isinstance(expected, dict):
        return {str(k).replace("\\", "/"): _remove_sha256_prefix(v).strip().lower() for k, v in expected.items()}
    raise RuntimeError("expected_sha256 必须是字符串或 {文件名: SHA256} 映射")


def verify_source_hashes(
    dataset_dir: Path,
    files: list[Path],
    expected: Any,
    *,
    require_expected: bool,
) -> tuple[list[dict[str, Any]], str]:
    records = [_file_record(path, dataset_dir) for path in files]
    expected_map = _expected_map(expected, files, dataset_dir)
    if require_expected and files and not expected_map:
        raise RuntimeError("严格 SHA256 模式要求清单提供 expected_sha256")
    for record in records:
        rel = record["path"]
        expected_hash = expected_map.get(rel) or expected_map.get(Path(rel).name)
        if expected_hash and record["sha256"].lower() != expected_hash:
            raise RuntimeError(
                f"SHA256 不匹配: {rel}; expected={expected_hash}; observed={record['sha256']}"
            )
        record["expected_sha256"] = expected_hash
        record["verified_against_expected"] = bool(expected_hash)
    if require_expected:
        unverified = [record["path"] for record in records if not record["verified_against_expected"]]
        if unverified:
            raise RuntimeError("严格 SHA256 模式缺少以下文件的 expected_sha256: " + ", ".join(unverified))
    missing = sorted(set(expected_map) - {r["path"] for r in records} - {Path(r["path"]).name for r in records})
    if missing:
        raise RuntimeError("expected_sha256 中的文件不存在: " + ", ".join(missing))
    status = "verified" if expected_map else ("recorded_tofu" if records else "standardized_only")
    return records, status


def _source_lock_path(root: Path) -> Path:
    return root / "data" / "dataset_source_lock.json"


def _load_source_lock(root: Path) -> dict[str, Any]:
    payload = _load_json(_source_lock_path(root), {})
    return payload if isinstance(payload, dict) else {}


def _save_source_lock(root: Path, datasets: dict[str, Any]) -> None:
    _atomic_json(
        _source_lock_path(root),
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": _utc_now(),
            "datasets": datasets,
        },
    )


def _inside(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_archive(archive: Path, destination: Path) -> list[Path]:
    """Extract zip/tar while rejecting traversal, links, and special files."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    lower = archive.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = destination / member.filename
                if not _inside(destination, target):
                    raise RuntimeError(f"压缩包路径穿越被拒绝: {member.filename}")
                mode = member.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise RuntimeError(f"压缩包符号链接被拒绝: {member.filename}")
            handle.extractall(destination)
        extracted = [p for p in destination.rglob("*") if p.is_file()]
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:*") as handle:
            members = handle.getmembers()
            for member in members:
                target = destination / member.name
                if not _inside(destination, target):
                    raise RuntimeError(f"压缩包路径穿越被拒绝: {member.name}")
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"压缩包链接或特殊文件被拒绝: {member.name}")
            handle.extractall(destination, members=members)
        extracted = [p for p in destination.rglob("*") if p.is_file()]
    else:
        raise RuntimeError(f"不支持的压缩格式: {archive.name}")
    return sorted(extracted, key=lambda p: p.as_posix().casefold())


def extract_archives(dataset_dir: Path, source_files: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for archive in source_files:
        if not archive.name.lower().endswith(ARCHIVE_SUFFIXES):
            continue
        stem = re.sub(r"\.(?:tar\.gz|tgz|tar|zip)$", "", archive.name, flags=re.IGNORECASE)
        destination = dataset_dir / "_extracted" / _safe_name(stem)
        if destination.exists():
            shutil.rmtree(destination)
        files = safe_extract_archive(archive, destination)
        records.append(
            {
                "archive": archive.relative_to(dataset_dir).as_posix(),
                "destination": destination.relative_to(dataset_dir).as_posix(),
                "file_count": len(files),
            }
        )
    return records


def _read_ground_truth(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "sequence", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{path}: ground_truth.csv 必须包含 id,sequence,label")
        rows: list[dict[str, Any]] = []
        for number, row in enumerate(reader, 2):
            sequence = re.sub(r"\s+", "", str(row.get("sequence") or "")).upper()
            label_raw = str(row.get("label") or "").strip()
            if not sequence or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence):
                raise RuntimeError(f"{path}:{number}: 非法肽序列")
            if label_raw not in {"0", "1", "0.0", "1.0"}:
                raise RuntimeError(f"{path}:{number}: label 必须为 0/1")
            rows.append({"id": str(row.get("id") or ""), "sequence": sequence, "label": int(float(label_raw))})
    if not rows:
        raise RuntimeError(f"{path}: 数据集为空")
    return rows


def _read_fasta_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return sequences


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id = ""
    current: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records.append((current_id, "".join(current).upper()))
                    current = []
                current_id = line[1:].split()[0] if line[1:].strip() else f"seq_{len(records) + 1}"
            else:
                current.append(line)
    if current:
        records.append((current_id, "".join(current).upper()))
    return records


def _upgrade_legacy_ground_truth(dataset_dir: Path) -> dict[str, Any] | None:
    """Add stable FASTA identifiers to legacy ``sequence,label`` CSV files."""
    gt_path = dataset_dir / "ground_truth.csv"
    fasta_path = dataset_dir / "combined_test.fasta"
    if not gt_path.is_file() or not fasta_path.is_file():
        return None
    with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "id" in fieldnames or not {"sequence", "label"}.issubset(fieldnames):
            return None
        rows = list(reader)
    fasta_records = _read_fasta_records(fasta_path)
    if len(rows) != len(fasta_records):
        raise RuntimeError("旧版 ground_truth.csv 缺少 id，且无法与 FASTA 按记录数对齐")
    for number, (row, (_, sequence)) in enumerate(zip(rows, fasta_records), 2):
        csv_sequence = re.sub(r"\s+", "", str(row.get("sequence") or "")).upper()
        if csv_sequence != sequence:
            raise RuntimeError(f"旧版 ground_truth.csv 缺少 id，且第 {number} 行无法与 FASTA 对齐")

    legacy_dir = dataset_dir / "_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    backup = legacy_dir / "ground_truth_without_id.csv"
    if not backup.exists():
        shutil.copy2(gt_path, backup)
    tmp = gt_path.with_suffix(f".csv.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sequence", "label"])
        writer.writeheader()
        for row, (seq_id, sequence) in zip(rows, fasta_records):
            writer.writerow({"id": seq_id, "sequence": sequence, "label": row.get("label")})
    os.replace(tmp, gt_path)
    return _file_record(backup, dataset_dir)


def validate_standardized_dataset(dataset_dir: Path) -> dict[str, Any]:
    gt_path = dataset_dir / "ground_truth.csv"
    fasta_path = dataset_dir / "combined_test.fasta"
    if not gt_path.is_file() or not fasta_path.is_file():
        raise RuntimeError("缺少 ground_truth.csv 或 combined_test.fasta")
    rows = _read_ground_truth(gt_path)
    fasta_sequences = _read_fasta_sequences(fasta_path)
    csv_sequences = [row["sequence"] for row in rows]
    if len(fasta_sequences) != len(csv_sequences):
        raise RuntimeError(
            f"FASTA/CSV 记录数不一致: fasta={len(fasta_sequences)}, csv={len(csv_sequences)}"
        )
    if fasta_sequences != csv_sequences:
        raise RuntimeError("FASTA 与 ground_truth.csv 的序列或顺序不一致")
    labels = [row["label"] for row in rows]
    if set(labels) != {0, 1}:
        raise RuntimeError("二分类 benchmark 数据集必须同时包含正样本和负样本")
    lengths = [len(row["sequence"]) for row in rows]
    lengths_by_label = {
        str(label): [len(row["sequence"]) for row in rows if row["label"] == label]
        for label in (0, 1)
    }
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    majority_count = max(positive_count, negative_count)
    minority_count = min(positive_count, negative_count)

    def length_summary(values: list[int]) -> dict[str, Any]:
        in_primary_range = sum(10 <= value <= 50 for value in values)
        return {
            "count": len(values),
            "min_aa": min(values),
            "max_aa": max(values),
            "median_aa": statistics.median(values),
            "count_10_50_aa": in_primary_range,
            "fraction_10_50_aa": in_primary_range / len(values),
        }

    return {
        "row_count": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": positive_count / len(labels),
        "minority_fraction": minority_count / len(labels),
        "minority_majority_ratio": minority_count / majority_count,
        "length_distribution": {
            "overall": length_summary(lengths),
            "negative": length_summary(lengths_by_label["0"]),
            "positive": length_summary(lengths_by_label["1"]),
        },
        "files": [_file_record(gt_path, dataset_dir), _file_record(fasta_path, dataset_dir)],
    }


def standardize_dataset(dataset_dir: Path, name: str) -> tuple[dict[str, Any], str]:
    if all((dataset_dir / filename).is_file() for filename in STANDARD_FILES):
        legacy_backup = _upgrade_legacy_ground_truth(dataset_dir)
        result = validate_standardized_dataset(dataset_dir)
        if legacy_backup:
            result["legacy_source_backup"] = legacy_backup
            return result, "upgraded_legacy_sequence_label_csv"
        return result, "validated_existing"
    # Import lazily so plan/status/tests do not require the LLM SDK or pandas.
    from data_prep import process_single_folder

    result = process_single_folder(str(dataset_dir), name)
    if result is None or getattr(result, "empty", True):
        raise RuntimeError("标准化未生成有效记录")
    return validate_standardized_dataset(dataset_dir), "generated"


def _sequences_from_reference(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return set()
            seq_col = next((c for c in reader.fieldnames if c.lower() in {"sequence", "seq", "peptide"}), None)
            if not seq_col:
                raise RuntimeError(f"泄漏参考 CSV 找不到 sequence 列: {path}")
            return {re.sub(r"\s+", "", str(row.get(seq_col) or "")).upper() for row in reader if row.get(seq_col)}
    return set(_read_fasta_sequences(path))


def check_leakage(
    root: Path,
    dataset_rows: list[dict[str, Any]],
    *,
    allow_cross_dataset_overlap: bool,
) -> dict[str, Any]:
    sets: dict[str, set[str]] = {}
    labels_by_sequence: dict[str, set[int]] = {}
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for item in dataset_rows:
        name = str(item["name"])
        gt = root / "data" / "datasets" / name / "ground_truth.csv"
        rows = _read_ground_truth(gt)
        seen: set[str] = set()
        within_duplicates: set[str] = set()
        for row in rows:
            sequence = row["sequence"]
            if sequence in seen:
                within_duplicates.add(sequence)
            seen.add(sequence)
            labels_by_sequence.setdefault(sequence, set()).add(row["label"])
        sets[name] = seen
        if within_duplicates:
            issues.append({"type": "within_dataset_duplicate", "dataset": name, "count": len(within_duplicates), "examples": sorted(within_duplicates)[:5]})

        reference_paths = item.get("leakage_reference_paths") or []
        reference_union: set[str] = set()
        for raw_path in reference_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                issues.append({"type": "missing_leakage_reference", "dataset": name, "path": str(path)})
                continue
            reference_union.update(_sequences_from_reference(path))
        overlap = seen & reference_union
        if overlap:
            issues.append({"type": "training_reference_overlap", "dataset": name, "count": len(overlap), "examples": sorted(overlap)[:5]})
        if not reference_paths:
            warnings.append({"type": "training_reference_not_configured", "dataset": name})

    names = sorted(sets, key=str.casefold)
    cross_overlaps: list[dict[str, Any]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                record = {"type": "cross_dataset_overlap", "datasets": [left, right], "count": len(overlap), "examples": sorted(overlap)[:5]}
                cross_overlaps.append(record)
                (warnings if allow_cross_dataset_overlap else issues).append(record)

    conflicts = sorted(sequence for sequence, values in labels_by_sequence.items() if len(values) > 1)
    if conflicts:
        issues.append({"type": "conflicting_labels", "count": len(conflicts), "examples": conflicts[:5]})
    return {
        "status": "passed" if not issues else "failed",
        "allow_cross_dataset_overlap": allow_cross_dataset_overlap,
        "issues": issues,
        "warnings": warnings,
        "cross_dataset_overlaps": cross_overlaps,
        "training_reference_coverage": {
            str(item["name"]): bool(item.get("leakage_reference_paths")) for item in dataset_rows
        },
    }


def check_dataset_selection(
    root: Path,
    plan: dict[str, Any],
    dataset_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the preregistered scientific dataset-selection policy."""
    policy = plan.get("dataset_selection_policy") or {}
    if not isinstance(policy, dict) or not policy.get("enabled", False):
        return {
            "status": "not_configured",
            "issues": [],
            "warnings": [{"type": "dataset_selection_policy_not_configured"}],
            "observed_profiles": {},
        }

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    plan_by_name = {str(row.get("name")): row for row in plan.get("datasets", [])}
    balanced_ratio = float(policy.get("balanced_minority_majority_ratio_min", 0.70))
    required_count = int(policy.get("required_dataset_count", 3))
    required_profiles = policy.get("required_profiles") or {"balanced": 1, "imbalanced": 2}
    min_total = int(policy.get("min_total_samples", 500))
    min_per_class = int(policy.get("min_samples_per_class", 100))
    primary_min = int(policy.get("primary_length_min_aa", 10))
    primary_max = int(policy.get("primary_length_max_aa", 50))
    min_length_fraction = float(policy.get("min_primary_length_fraction", 0.80))
    min_class_length_fraction = float(policy.get("min_primary_length_fraction_per_class", min_length_fraction))
    absolute_min = int(policy.get("absolute_min_length_aa", 5))
    absolute_max = int(policy.get("absolute_max_length_aa", 100))
    observed_profiles: dict[str, dict[str, Any]] = {}
    profile_counts = {"balanced": 0, "imbalanced": 0}
    imbalanced_minority_fractions: list[float] = []

    if len(dataset_records) != required_count:
        issues.append({
            "type": "dataset_count_mismatch",
            "required": required_count,
            "observed": len(dataset_records),
        })

    for record in dataset_records:
        name = str(record.get("name"))
        row = plan_by_name.get(name, {})
        standardized = record.get("standardized") or {}
        positive = int(standardized.get("positive_count", 0))
        negative = int(standardized.get("negative_count", 0))
        total = int(standardized.get("row_count", 0))
        ratio = float(standardized.get("minority_majority_ratio", 0.0))
        minority_fraction = float(standardized.get("minority_fraction", 0.0))
        observed_profile = "balanced" if ratio >= balanced_ratio else "imbalanced"
        profile_counts[observed_profile] += 1
        if observed_profile == "imbalanced":
            imbalanced_minority_fractions.append(minority_fraction)
        declared_profile = str(row.get("selection_profile") or "").strip().lower()
        observed_profiles[name] = {
            "declared": declared_profile or None,
            "observed": observed_profile,
            "positive_count": positive,
            "negative_count": negative,
            "positive_fraction": standardized.get("positive_fraction"),
            "minority_fraction": minority_fraction,
            "minority_majority_ratio": ratio,
        }

        if declared_profile not in {"balanced", "imbalanced"}:
            issues.append({"type": "selection_profile_missing", "dataset": name})
        elif declared_profile != observed_profile:
            issues.append({
                "type": "selection_profile_mismatch",
                "dataset": name,
                "declared": declared_profile,
                "observed": observed_profile,
                "minority_majority_ratio": ratio,
                "balanced_threshold": balanced_ratio,
            })
        if total < min_total:
            issues.append({"type": "insufficient_total_samples", "dataset": name, "required": min_total, "observed": total})
        if min(positive, negative) < min_per_class:
            issues.append({
                "type": "insufficient_class_samples",
                "dataset": name,
                "required_per_class": min_per_class,
                "positive_count": positive,
                "negative_count": negative,
            })

        length_distribution = standardized.get("length_distribution") or {}
        # Statistics are fixed to 10--50 in the standardized manifest.  A
        # non-default range is reported as unsupported rather than silently
        # applying the wrong interval.
        if (primary_min, primary_max) != (10, 50):
            issues.append({"type": "unsupported_primary_length_range", "dataset": name, "required_range": [primary_min, primary_max]})
        else:
            for group, required_fraction in (("overall", min_length_fraction), ("positive", min_class_length_fraction), ("negative", min_class_length_fraction)):
                summary = length_distribution.get(group) or {}
                fraction = summary.get("fraction_10_50_aa")
                if fraction is None or float(fraction) < required_fraction:
                    issues.append({
                        "type": "primary_length_coverage_below_threshold",
                        "dataset": name,
                        "group": group,
                        "range_aa": [primary_min, primary_max],
                        "required_fraction": required_fraction,
                        "observed_fraction": fraction,
                    })
        overall_lengths = length_distribution.get("overall") or {}
        if overall_lengths and (
            int(overall_lengths.get("min_aa", absolute_min)) < absolute_min
            or int(overall_lengths.get("max_aa", absolute_max)) > absolute_max
        ):
            issues.append({
                "type": "absolute_length_out_of_range",
                "dataset": name,
                "allowed_range_aa": [absolute_min, absolute_max],
                "observed_range_aa": [overall_lengths.get("min_aa"), overall_lengths.get("max_aa")],
            })
        max_median_gap = policy.get("max_class_median_length_gap_aa")
        positive_lengths = length_distribution.get("positive") or {}
        negative_lengths = length_distribution.get("negative") or {}
        if max_median_gap is not None and positive_lengths and negative_lengths:
            observed_gap = abs(float(positive_lengths.get("median_aa", 0)) - float(negative_lengths.get("median_aa", 0)))
            if observed_gap > float(max_median_gap):
                issues.append({
                    "type": "class_length_distribution_mismatch",
                    "dataset": name,
                    "max_median_gap_aa": float(max_median_gap),
                    "observed_median_gap_aa": observed_gap,
                    "positive_median_aa": positive_lengths.get("median_aa"),
                    "negative_median_aa": negative_lengths.get("median_aa"),
                })

        metadata_requirements = (
            ("require_source_url", "source_url", "dataset_source_missing"),
            ("require_citation", "citation", "dataset_citation_missing"),
            ("require_version", "dataset_version", "dataset_version_missing"),
            ("require_retrieval_date", "retrieved_at_utc", "dataset_retrieval_date_missing"),
            ("require_license", "license", "dataset_license_missing"),
            ("require_expected_sha256", "expected_sha256", "expected_sha256_missing"),
            ("require_training_references", "leakage_reference_paths", "training_reference_not_configured"),
            ("require_label_definition", "label_definition", "label_definition_missing"),
            ("require_negative_sampling_strategy", "negative_sampling_strategy", "negative_sampling_strategy_missing"),
        )
        for policy_key, row_key, issue_type in metadata_requirements:
            if policy.get(policy_key, False) and not row.get(row_key):
                issues.append({"type": issue_type, "dataset": name})
        if policy.get("require_independent_external_test", False) and row.get("independent_external_test") is not True:
            issues.append({"type": "independent_external_test_not_confirmed", "dataset": name})
        if policy.get("require_low_homology_report", False):
            raw_report_path = str(row.get("homology_report_path") or "").strip()
            if not raw_report_path:
                issues.append({"type": "low_homology_report_missing", "dataset": name})
            else:
                report_path = Path(raw_report_path)
                if not report_path.is_absolute():
                    report_path = root / report_path
                if not report_path.is_file():
                    issues.append({"type": "low_homology_report_not_found", "dataset": name, "path": str(report_path)})
                else:
                    try:
                        report = _load_json(report_path, {})
                        required_identity = float(policy.get("max_training_sequence_identity", 0.40))
                        report_identity = float(report.get("identity_threshold"))
                        if (
                            report.get("status") != "passed"
                            or report_identity > required_identity
                            or not report.get("tool")
                            or not report.get("tool_version")
                            or not report.get("training_reference_sha256")
                        ):
                            issues.append({
                                "type": "low_homology_report_invalid",
                                "dataset": name,
                                "path": str(report_path),
                                "required_identity_threshold_max": required_identity,
                            })
                    except Exception as exc:
                        issues.append({"type": "low_homology_report_unreadable", "dataset": name, "path": str(report_path), "error": str(exc)})

    for profile, expected in required_profiles.items():
        observed = profile_counts.get(str(profile), 0)
        if observed != int(expected):
            issues.append({
                "type": "dataset_profile_count_mismatch",
                "profile": str(profile),
                "required": int(expected),
                "observed": observed,
            })

    min_gap = float(policy.get("min_imbalanced_minority_fraction_gap", 0.10))
    if len(imbalanced_minority_fractions) == 2:
        observed_gap = abs(imbalanced_minority_fractions[0] - imbalanced_minority_fractions[1])
        if observed_gap < min_gap:
            issues.append({
                "type": "imbalanced_datasets_too_similar",
                "required_minority_fraction_gap": min_gap,
                "observed_gap": observed_gap,
            })

    return {
        "status": "passed" if not issues else "failed",
        "policy": policy,
        "issues": issues,
        "warnings": warnings,
        "observed_profile_counts": profile_counts,
        "observed_profiles": observed_profiles,
    }


def _write_manifest(root: Path, payload: dict[str, Any]) -> Path:
    manifests = root / "data" / "dataset_manifests"
    path = manifests / f"{payload['gate_id']}.json"
    _atomic_json(path, payload)
    _atomic_json(
        manifests / "latest.json",
        {
            "gate_id": payload["gate_id"],
            "status": payload["status"],
            "manifest": path.relative_to(root).as_posix(),
            "updated_at_utc": _utc_now(),
        },
    )
    return path


def run_dataset_gate(
    root: Path,
    *,
    strategy_path: Path | None = None,
    plan_path: Path | None = None,
    allow_cross_dataset_overlap: bool = False,
    require_expected_sha256: bool = False,
) -> tuple[dict[str, Any], Path]:
    root = root.resolve()
    plan_path = plan_path or root / "data" / "dataset_plan.json"
    plan = generate_dataset_plan(root, strategy_path=strategy_path, plan_path=plan_path)
    gate_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": gate_id,
        "status": "running",
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "policy": {
            "allow_cross_dataset_overlap": allow_cross_dataset_overlap,
            "require_expected_sha256": require_expected_sha256,
            "checksum_policy": "official checksum when supplied; otherwise TOFU baseline recorded for subsequent verification",
        },
        "plan": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": sha256_file(plan_path),
            "generation_mode": plan.get("generation_mode"),
        },
        "datasets": [],
        "dataset_selection_check": {},
        "leakage_check": {},
        "errors": [],
    }

    if not plan.get("datasets"):
        manifest["status"] = "failed"
        manifest["errors"].append("数据集清单为空")
        manifest["completed_at_utc"] = _utc_now()
        return manifest, _write_manifest(root, manifest)

    successful_rows: list[dict[str, Any]] = []
    source_lock = _load_source_lock(root)
    locked_datasets = source_lock.get("datasets", {})
    if not isinstance(locked_datasets, dict):
        locked_datasets = {}
    lock_changed = False
    for row in plan["datasets"]:
        record: dict[str, Any] = {
            "name": row["name"],
            "source_url": row.get("source_url") or None,
            "role": row.get("role"),
            "status": "running",
            "stages": {},
        }
        manifest["datasets"].append(record)
        try:
            dataset_dir, files, method = acquire_dataset(root, row)
            record["path"] = dataset_dir.relative_to(root).as_posix()
            record["stages"]["download"] = {"status": "passed", "method": method, "file_count": len(files)}
            official_expected = row.get("expected_sha256")
            tofu_expected = locked_datasets.get(str(row["name"]))
            effective_expected = official_expected or tofu_expected
            hashes, integrity_status = verify_source_hashes(
                dataset_dir,
                files,
                effective_expected,
                require_expected=require_expected_sha256,
            )
            if tofu_expected and not official_expected:
                integrity_status = "verified_tofu"
            elif not effective_expected and hashes:
                locked_datasets[str(row["name"])] = {
                    item["path"]: item["sha256"] for item in hashes
                }
                lock_changed = True
            record["source_files"] = hashes
            record["stages"]["sha256"] = {"status": "passed", "integrity_status": integrity_status}
            extraction = extract_archives(dataset_dir, files)
            record["extracted_archives"] = extraction
            record["stages"]["extract"] = {"status": "passed", "archive_count": len(extraction)}
            standardized, method = standardize_dataset(dataset_dir, str(row["name"]))
            record["standardized"] = standardized
            record["stages"]["standardize"] = {"status": "passed", "method": method}
            record["status"] = "prepared"
            successful_rows.append(row)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            manifest["errors"].append(f"{row.get('name')}: {exc}")

    if len(successful_rows) == len(plan["datasets"]):
        selection = check_dataset_selection(root, plan, manifest["datasets"])
        manifest["dataset_selection_check"] = selection
        if selection["status"] == "failed":
            manifest["errors"].append("科学数据集选取规则未通过")
        elif selection["status"] == "passed":
            for record in manifest["datasets"]:
                record["stages"]["dataset_selection"] = {"status": "passed"}
        leakage = check_leakage(
            root,
            successful_rows,
            allow_cross_dataset_overlap=allow_cross_dataset_overlap,
        )
        manifest["leakage_check"] = leakage
        if leakage["status"] != "passed":
            manifest["errors"].append("数据泄漏检查未通过")
        else:
            for record in manifest["datasets"]:
                record["stages"]["leakage"] = {"status": "passed"}
                record["status"] = "passed"
    else:
        manifest["dataset_selection_check"] = {"status": "not_run", "reason": "前置数据集阶段失败"}
        manifest["leakage_check"] = {"status": "not_run", "reason": "前置数据集阶段失败"}

    manifest["status"] = "passed" if not manifest["errors"] else "failed"
    manifest["completed_at_utc"] = _utc_now()
    if lock_changed:
        _save_source_lock(root, locked_datasets)
    return manifest, _write_manifest(root, manifest)


def dataset_gate_issues(root: Path, dataset_dirs: Iterable[Path] | None = None) -> list[str]:
    root = root.resolve()
    latest_path = root / "data" / "dataset_manifests" / "latest.json"
    if not latest_path.exists():
        return ["没有数据集门禁 manifest；请先从菜单运行‘准备数据集并执行完整门禁’"]
    try:
        latest = _load_json(latest_path, {})
        manifest_path = root / str(latest["manifest"])
        manifest = _load_json(manifest_path, {})
    except Exception as exc:
        return [f"数据集门禁 manifest 无法读取: {exc}"]
    issues: list[str] = []
    if manifest.get("status") != "passed":
        issues.append(f"最近一次数据集门禁状态为 {manifest.get('status', 'unknown')}")
        return issues
    by_name = {str(row.get("name")): row for row in manifest.get("datasets", [])}
    selected = list(dataset_dirs) if dataset_dirs is not None else [root / str(row.get("path")) for row in by_name.values()]
    for dataset_dir in selected:
        record = by_name.get(dataset_dir.name)
        if not record:
            issues.append(f"数据集未包含在门禁 manifest 中: {dataset_dir.name}")
            continue
        standardized = record.get("standardized") or {}
        expected_files = {str(item.get("path")): item for item in standardized.get("files", [])}
        for filename in STANDARD_FILES:
            path = dataset_dir / filename
            expected = expected_files.get(filename)
            if not path.is_file():
                issues.append(f"{dataset_dir.name}: 缺少 {filename}")
            elif not expected:
                issues.append(f"{dataset_dir.name}: manifest 缺少 {filename} 记录")
            elif sha256_file(path) != expected.get("sha256"):
                issues.append(f"{dataset_dir.name}: {filename} 在门禁后发生变化")
        for source in record.get("source_files", []):
            rel = str(source.get("path") or "")
            path = dataset_dir / rel
            if not path.is_file():
                issues.append(f"{dataset_dir.name}: 原始文件在门禁后缺失: {rel}")
            elif sha256_file(path) != source.get("sha256"):
                issues.append(f"{dataset_dir.name}: 原始文件在门禁后发生变化: {rel}")
    return issues


def require_dataset_gate(root: Path, dataset_dirs: Iterable[Path] | None = None) -> None:
    issues = dataset_gate_issues(root, dataset_dirs)
    if issues:
        raise RuntimeError("数据集准备门禁未通过: " + "; ".join(issues))


def _print_summary(manifest: dict[str, Any], path: Path) -> None:
    print("\n========== 数据集准备门禁结果 ==========")
    print("状态:", manifest["status"])
    print("manifest:", path)
    for row in manifest.get("datasets", []):
        count = (row.get("standardized") or {}).get("row_count", "-")
        print(f" - {row.get('name')}: {row.get('status')} ({count} records)")
        if row.get("error"):
            print("   错误:", row["error"])
    leakage = manifest.get("leakage_check") or {}
    selection = manifest.get("dataset_selection_check") or {}
    print("科学选集检查:", selection.get("status", "not_run"))
    for issue in selection.get("issues", []):
        print(" -", json.dumps(issue, ensure_ascii=False))
    for warning in selection.get("warnings", []):
        print(" [warning]", json.dumps(warning, ensure_ascii=False))
    print("泄漏检查:", leakage.get("status", "not_run"))
    for issue in leakage.get("issues", []):
        print(" -", json.dumps(issue, ensure_ascii=False))
    for warning in leakage.get("warnings", []):
        print(" [warning]", json.dumps(warning, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMP benchmark dataset preparation gate")
    parser.add_argument("command", choices=["prepare", "status"], nargs="?", default="prepare")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--strategy", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--allow-cross-dataset-overlap", action="store_true")
    parser.add_argument("--require-expected-sha256", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "status":
        issues = dataset_gate_issues(root)
        if issues:
            print("数据集门禁未通过:")
            for issue in issues:
                print(" -", issue)
            return 2
        print("数据集门禁已通过，标准化文件 SHA256 与 manifest 一致。")
        return 0
    manifest, path = run_dataset_gate(
        root,
        strategy_path=args.strategy,
        plan_path=args.plan,
        allow_cross_dataset_overlap=args.allow_cross_dataset_overlap,
        require_expected_sha256=args.require_expected_sha256,
    )
    _print_summary(manifest, path)
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
