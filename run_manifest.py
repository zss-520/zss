"""Run-scoped provenance manifest for reproducible AMP benchmark executions."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python 3.7
    try:
        import importlib_metadata  # type: ignore
    except ImportError:
        importlib_metadata = None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=8,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _repo_commit(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD") if (path / ".git").exists() else ""


def _dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "paramiko", "openai"]:
        try:
            if importlib_metadata is None:
                raise LookupError(name)
            out[name] = importlib_metadata.version(name)
        except Exception:
            out[name] = "not-installed"
    return out


def _dataset_record(path: Path) -> dict[str, Any]:
    files = []
    for name in ["ground_truth.csv", "combined_test.fasta", "validation_results_with_predictions.csv"]:
        fp = path / name
        if fp.exists() and fp.is_file():
            files.append(
                {
                    "name": name,
                    "size_bytes": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                }
            )
    return {"name": path.name, "path": str(path), "files": files}


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_compute_config() -> dict[str, Any]:
    keys = [
        "HPC_TARGET_DIR",
        "CONDA_SH_PATH",
        "VLAB_ENV",
        "SLURM_PARTITION",
        "SLURM_JOB_NAME",
        "SLURM_GPUS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_POLL_SECONDS",
        "SLURM_JOB_TIMEOUT_SECONDS",
        "SLURM_SACCT_RETRIES",
        "SLURM_SACCT_RETRY_SECONDS",
    ]
    return {key: os.getenv(key) for key in keys}


def _model_record(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    local_raw = str(row.get("local_model_dir") or "").strip()
    local = Path(local_raw) if local_raw else None
    if local is not None and not local.is_absolute():
        local = root / local
    return {
        "model_name": row.get("model_name"),
        "env_name": row.get("env_name"),
        "python_version": row.get("python_version"),
        "repo_url": row.get("repo_url") or row.get("code_repository_url"),
        "local_model_dir": row.get("local_model_dir"),
        "local_repo_commit": (
            _repo_commit(local)
            if local is not None and local.exists() and local.is_dir()
            else ""
        ),
        "remote_repo_dir": row.get("remote_repo_dir"),
        "hpc_env_status": row.get("hpc_env_status"),
        "hpc_smoke_test": row.get("hpc_smoke_test"),
        "benchmark_role": row.get("benchmark_role"),
        "benchmark_roles": row.get("benchmark_roles"),
        "benchmark_role_reason": row.get("benchmark_role_reason"),
        "publication_year": row.get("publication_year") or row.get("source_year") or row.get("year"),
        "inference_cmd_template": row.get("inference_cmd_template"),
    }


class RunManifest:
    def __init__(self, root: Path, payload: dict[str, Any]):
        self.root = root
        self.data = payload
        self.run_id = str(payload["run_id"])
        self.run_dir = root / "data" / "runs" / self.run_id
        self.results_dir = self.run_dir / "results"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.path = self.run_dir / "manifest.json"
        if self.path.exists():
            raise FileExistsError(
                f"run_id {self.run_id!r} already exists; choose a new AMP_RUN_ID"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._finalized = False
        self.save()
        atexit.register(self._finalize_aborted)

    @classmethod
    def start(
        cls,
        *,
        root: Path,
        models: Iterable[dict[str, Any]],
        datasets: Iterable[Path],
        metric_protocol: dict[str, Any],
        llm_model: str,
        allow_unverified_models: bool,
        benchmark_portfolio: dict[str, Any] | None = None,
    ) -> "RunManifest":
        requested = os.getenv("AMP_RUN_ID", "").strip()
        if requested:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", requested):
                raise ValueError("AMP_RUN_ID 只允许字母、数字、点、下划线和连字符")
            run_id = requested
        else:
            run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
        git_head = _git(root, "rev-parse", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain"))
        source_names = [
            "main.py",
            "workflow_utils.py",
            "workflow_guards.py",
            "scientific_evaluation.py",
            "run_manifest.py",
            "config.py",
        ]
        source_files = [
            _file_record(root / name) for name in source_names if (root / name).is_file()
        ]
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "created_at_utc": _utc_now(),
            "completed_at_utc": None,
            "project": {
                "root": str(root),
                "git_head": git_head,
                "git_dirty": dirty,
                "workflow_sources": source_files,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
                "llm_model": llm_model,
                "argv": list(sys.argv),
                "compute_config": _safe_compute_config(),
            },
            "policy": {
                "allow_unverified_models": allow_unverified_models,
                "metric_protocol": metric_protocol,
                "benchmark_portfolio": benchmark_portfolio or {},
            },
            "models": [_model_record(root, row) for row in models],
            "datasets": [_dataset_record(path) for path in datasets],
            "dataset_runs": {},
            "artifacts": {},
            "events": [],
        }
        return cls(root, payload)

    def add_event(self, stage: str, status: str, **details: Any) -> None:
        self.data["events"].append(
            {"time_utc": _utc_now(), "stage": stage, "status": status, **details}
        )
        self.save()

    def record_dataset(self, name: str, *, status: str, **details: Any) -> None:
        self.data["dataset_runs"][name] = {
            "status": status,
            "updated_at_utc": _utc_now(),
            **details,
        }
        self.save()

    def record_artifact(self, name: str, path: Path) -> None:
        self.data["artifacts"][name] = _file_record(path)
        self.save()

    def finalize(self, status: str, **details: Any) -> None:
        self.data["status"] = status
        self.data["completed_at_utc"] = _utc_now()
        self.data.update(details)
        self._finalized = True
        self.save()

    def _finalize_aborted(self) -> None:
        if not self._finalized and self.data.get("status") == "running":
            self.finalize("aborted", termination_reason="process exited before explicit finalization")

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".json.tmp-{os.getpid()}")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
        latest = self.root / "data" / "runs" / "latest.json"
        latest_tmp = latest.with_suffix(f".json.tmp-{os.getpid()}")
        latest_tmp.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "manifest": str(self.path),
                    "status": self.data.get("status"),
                    "updated_at_utc": _utc_now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(latest_tmp, latest)
