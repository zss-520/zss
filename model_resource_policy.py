# -*- coding: utf-8 -*-
"""Auditable, model-agnostic resource eligibility gate for model ranking."""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = ROOT / "data" / "model_resource_policy.json"


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def load_model_resource_policy(path: Path | None = None) -> Dict[str, Any]:
    configured = os.getenv("MODEL_RESOURCE_POLICY_PATH", "").strip()
    policy_path = path or (Path(configured) if configured else DEFAULT_POLICY_PATH)
    if not policy_path.exists():
        return {
            "schema_version": 1,
            "enabled": False,
            "policy_mode": "measured_budget_gate",
            "missing_measurement_policy": "keep_and_flag",
            "models": {},
            "policy_path": str(policy_path),
        }
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError(f"Resource policy must be a JSON object: {policy_path}")
    policy["policy_path"] = str(policy_path)
    return policy


def _model_names(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    return sorted(
        {str(row.get("model") or "").strip() for row in rows if str(row.get("model") or "").strip()},
        key=str.lower,
    )


def apply_resource_gate(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Apply the same predeclared resource limits to every measured model.

    Missing measurements are retained by default. This prevents absence of
    profiling data from becoming a hidden way to remove leaderboard leaders.
    """
    policy = dict(policy or load_model_resource_policy())
    models = _model_names(rows)
    profiles = policy.get("models") if isinstance(policy.get("models"), dict) else {}
    enabled = bool(policy.get("enabled", True))
    missing_policy = str(policy.get("missing_measurement_policy", "keep_and_flag")).strip().lower()
    if missing_policy != "keep_and_flag":
        raise ValueError(
            "missing_measurement_policy must be 'keep_and_flag'; models cannot be excluded without measurements"
        )

    limits = {
        "runtime_seconds": _finite(policy.get("max_runtime_seconds")),
        "peak_memory_mb": _finite(policy.get("max_peak_memory_mb")),
        "gpu_memory_mb": _finite(policy.get("max_gpu_memory_mb")),
    }
    minimum_throughput = _finite(policy.get("minimum_throughput_sequences_per_second"))
    checks: List[Dict[str, Any]] = []
    excluded: List[str] = []
    flagged: List[str] = []

    for model in models:
        profile = profiles.get(model, {}) if isinstance(profiles.get(model, {}), dict) else {}
        measurements = {field: _finite(profile.get(field)) for field in limits}
        measurements["throughput_sequences_per_second"] = _finite(
            profile.get("throughput_sequences_per_second")
        )
        measurement_source = str(profile.get("measurement_source") or "").strip()
        has_values = any(value is not None for value in measurements.values())
        available = has_values and bool(measurement_source)
        reasons: List[str] = []
        if enabled and available:
            for field, maximum in limits.items():
                value = measurements[field]
                if maximum is not None and value is not None and value > maximum:
                    reasons.append(f"{field}={value:g} exceeds limit {maximum:g}")
            throughput = measurements["throughput_sequences_per_second"]
            if minimum_throughput is not None and throughput is not None and throughput < minimum_throughput:
                reasons.append(
                    f"throughput_sequences_per_second={throughput:g} below minimum {minimum_throughput:g}"
                )

        if not enabled:
            status = "eligible_gate_disabled"
        elif reasons:
            status = "excluded_measured_budget_failure"
            excluded.append(model)
        elif not available:
            status = "eligible_missing_resource_measurement"
            if has_values and not measurement_source:
                reasons.append("resource values lack measurement_source and were not used")
            flagged.append(model)
        else:
            status = "eligible_measured_budget_pass"

        checks.append({
            "model": model,
            "status": status,
            **measurements,
            "measurement_source": measurement_source,
            "reasons": reasons,
        })

    excluded_set = set(excluded)
    filtered = [row for row in rows if str(row.get("model") or "").strip() not in excluded_set]
    eligible = [model for model in models if model not in excluded_set]
    if len(eligible) < 3:
        raise ValueError(
            f"Resource gate left only {len(eligible)} eligible models; at least 3 are required for Top3 ranking"
        )
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_mode": "measured_budget_gate",
        "policy_path": str(policy.get("policy_path") or DEFAULT_POLICY_PATH),
        "enabled": enabled,
        "missing_measurement_policy": missing_policy,
        "limits": {**limits, "minimum_throughput_sequences_per_second": minimum_throughput},
        "models_before": len(models),
        "models_after": len(eligible),
        "rows_before": len(rows),
        "rows_after": len(filtered),
        "eligible_models": eligible,
        "excluded_models": excluded,
        "flagged_models": flagged,
        "checks": checks,
        "principle": "Eligibility uses measured resource evidence only; ranking values and model names are not gate inputs.",
    }
    return filtered, audit


def write_resource_gate_audit(output_dir: Path, audit: Mapping[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_resource_gate_audit.json"
    csv_path = output_dir / "model_resource_gate_audit.csv"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "model", "status", "runtime_seconds", "peak_memory_mb", "gpu_memory_mb",
        "throughput_sequences_per_second", "measurement_source", "reasons",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for check in audit.get("checks", []):
            row = {field: check.get(field) for field in fields}
            row["reasons"] = "; ".join(check.get("reasons") or [])
            writer.writerow(row)
    return {"json": str(json_path), "csv": str(csv_path)}
