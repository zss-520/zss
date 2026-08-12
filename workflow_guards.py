"""Pure preflight guards shared by the menu and the benchmark runner."""
from __future__ import annotations

import os
from typing import Any, Iterable


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def model_readiness_issues(models: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in models:
        name = str(row.get("model_name") or "unknown_model")
        env_status = str(row.get("hpc_env_status") or "unknown")
        smoke_status = str(row.get("hpc_smoke_test") or "unknown")
        if env_status != "ready" or smoke_status != "passed":
            issues.append(
                {
                    "model_name": name,
                    "hpc_env_status": env_status,
                    "hpc_smoke_test": smoke_status,
                    "reason": "benchmark requires hpc_env_status=ready and hpc_smoke_test=passed",
                }
            )
    return issues


def require_models_ready(
    models: Iterable[dict[str, Any]],
    *,
    allow_unverified: bool = False,
) -> None:
    issues = model_readiness_issues(models)
    if not issues or allow_unverified:
        return
    details = "; ".join(
        f"{x['model_name']} (HPC={x['hpc_env_status']}, smoke={x['hpc_smoke_test']})"
        for x in issues
    )
    raise RuntimeError(
        "模型运行门禁未通过: "
        + details
        + ". 请先完成 HPC 部署与 smoke test；如确需诊断性强制运行，显式设置 "
        + "ALLOW_UNVERIFIED_MODELS=1。"
    )

