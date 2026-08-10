# -*- coding: utf-8 -*-
"""Real LLM multi-agent metric-weight deliberation for the AMP benchmark.

The existing ``iterative_weight_meeting.py`` remains the deterministic control.
This module runs three independent expert Agents, one Reviewer Agent, and one
Chief Agent for an initial literature meeting and every subsequent bootstrap
round.  Model identities and leaderboard positions are withheld from the
weight-setting Agents.  Complete prompts, replies, usage metadata, checkpoints,
weights and rankings are persisted for audit and resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from agent_md_loader import AgentMDLoader

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from iterative_weight_meeting import (
    _normalize,
    _prepare,
    _review_metric_evidence,
    _score_models,
)
from model_resource_policy import (
    apply_resource_gate,
    load_model_resource_policy,
    write_resource_gate_audit,
)


load_dotenv(ROOT / ".env")

DEFAULT_MODEL = os.getenv("MODEL_NAME", "qwen3-coder-plus")
ROLE_NAMES = ("literature_agent", "statistics_agent", "screening_agent")
RAW_CALL_LOCK = threading.Lock()
WEIGHT_PROMPT_DIR = ROOT / "agents" / "weight_meeting"
_WEIGHT_PROMPT_LOADER = AgentMDLoader(WEIGHT_PROMPT_DIR)


def _load_weight_prompt(name: str) -> str:
    """Load one UTF-8 Step-3 Agent prompt from the shared prompt directory."""
    return _WEIGHT_PROMPT_LOADER.load_composed(name).strip()

METRIC_GUIDE = {
    "ACC": "Overall accuracy; prevalence-sensitive in imbalanced tests.",
    "AUPRC": "Threshold-free precision-recall ranking quality; primary imbalance-sensitive endpoint.",
    "AUROC": "Threshold-free discrimination across FPR/TPR; can look optimistic under severe imbalance.",
    "BalancedAccuracy": "Mean sensitivity and specificity; threshold-dependent class-balanced summary.",
    "BrierScore": "Probability calibration and accuracy; lower is better.",
    "ECE": "Expected calibration error; lower is better and binning-dependent.",
    "F1-Score": "Harmonic mean of precision and recall at a frozen threshold.",
    "MCC": "Correlation-like summary using all confusion-matrix cells; robust to imbalance.",
    "NPV": "Reliability of negative calls; prevalence-dependent and relevant to exclusion decisions.",
    "Precision": "Positive predictive value; controls experimental false-positive burden.",
    "Recall": "Sensitivity; controls missed AMP candidates.",
    "Specificity": "True-negative rate; controls non-AMP rejection at a frozen threshold.",
}

LITERATURE_CONSENSUS = {
    "source_id": "literature_deep_research_memory.metric_consensus",
    "evidence_pool": "2365 papers, 304 evidence batches and 241 compact chunk summaries",
    "historical_primary_proposal": {
        "AUPRC": 0.35,
        "MCC": 0.30,
        "Recall": 0.20,
        "Precision": 0.15,
    },
    "historical_revised_proposal": {
        "AUPRC": 0.35,
        "MCC": 0.25,
        "Recall": 0.20,
        "Precision": 0.15,
        "AUROC": 0.05,
    },
    "claims": [
        "AUPRC is the primary endpoint for highly imbalanced AMP binary classification.",
        "MCC uses TP/TN/FP/FN and is more robust than accuracy under class imbalance.",
        "Recall limits missed AMP candidates; precision limits false candidates entering wet-lab screening.",
        "ACC, specificity, AUROC, F1 and calibration metrics remain mandatory report dimensions even when not dominant ranking endpoints.",
        "Thresholds should be selected on independent validation evidence and frozen before formal testing.",
        "Homology leakage and dataset provenance remain unresolved gates for the three currently stored datasets.",
    ],
    "verified_dataset_sources": [
        "Veltri_test: DOI 10.1093/bioinformatics/bty179; model-associated data, independence not confirmed.",
        "C_AMPs-predict_test: DOI 10.1038/s41587-022-01226-0; model-associated data, independence not confirmed.",
        "ProteoGPT_all_predictions: DOI 10.5281/zenodo.16633186; model-associated data, overlap audit pending.",
    ],
}


SHARED_SYSTEM = _load_weight_prompt("shared_system")

ROLE_INSTRUCTIONS = {
    "literature_agent": _load_weight_prompt("literature_agent"),
    "statistics_agent": _load_weight_prompt("statistics_agent"),
    "screening_agent": _load_weight_prompt("screening_agent"),
}

REVIEWER_SYSTEM = (
    SHARED_SYSTEM + "\n\n" + _load_weight_prompt("reviewer_agent")
).strip()

CHIEF_SYSTEM = (
    SHARED_SYSTEM + "\n\n" + _load_weight_prompt("chief_agent")
).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_client() -> OpenAI:
    model_name = str(DEFAULT_MODEL or "").lower()
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MEETING_LLM_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not base_url and model_name.startswith(("qwen", "deepseek")):
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("DASHSCOPE_API_KEY") or api_key
    elif base_url and "dashscope" in base_url.lower():
        api_key = os.getenv("DASHSCOPE_API_KEY") or api_key
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY or DASHSCOPE_API_KEY is configured")
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("LLM_WEIGHT_TIMEOUT_SECONDS", "180")),
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def retryable(exc: Exception) -> bool:
    text = (exc.__class__.__name__ + " " + str(exc)).lower()
    return any(token in text for token in (
        "timeout", "connection", "rate limit", "429", "500", "502", "503", "504", "temporarily"
    ))


def usage_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CALL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(
    *, output_dir: Path, meeting: str, round_no: int, role: str,
    system_prompt: str, user_prompt: str, max_tokens: int,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    max_attempts = int(os.getenv("LLM_WEIGHT_MAX_ATTEMPTS", "5"))
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        started = utc_now()
        try:
            client = make_client()
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            record = {
                "meeting": meeting,
                "round": round_no,
                "role": role,
                "attempt": attempt,
                "model": DEFAULT_MODEL,
                "started_at": started,
                "completed_at": utc_now(),
                "request_id": getattr(response, "id", None),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "prompt_sha256": sha256_text(system_prompt + "\n" + user_prompt),
                "response": content,
                "usage": usage_dict(response),
            }
            append_jsonl(output_dir / "raw_llm_calls.jsonl", record)
            return record
        except Exception as exc:
            last_error = exc
            append_jsonl(output_dir / "llm_errors.jsonl", {
                "meeting": meeting, "round": round_no, "role": role,
                "attempt": attempt, "started_at": started,
                "failed_at": utc_now(), "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if attempt >= max_attempts or not retryable(exc):
                raise
            wait_seconds = min(5 * (2 ** (attempt - 1)), 40)
            print(f"[retry] {meeting} round={round_no} role={role} {type(exc).__name__}; wait {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError(f"LLM call failed: {last_error}")


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response contains no JSON object")
        obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM response JSON must be an object")
    return obj


def compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_agent_weights(raw: Mapping[str, Any], metrics: Sequence[str]) -> tuple[Dict[str, float], Dict[str, Any]]:
    aliases = {compact_key(metric): metric for metric in metrics}
    aliases.update({
        "accuracy": "ACC", "balancedaccuracy": "BalancedAccuracy",
        "brierscore": "BrierScore", "f1": "F1-Score", "f1score": "F1-Score",
        "sensitivity": "Recall", "recallsensitivity": "Recall",
    })
    mapped: Dict[str, float] = {}
    for key, value in raw.items():
        metric = aliases.get(compact_key(key))
        if metric not in metrics:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            mapped[metric] = number
    missing = [metric for metric in metrics if metric not in mapped]
    if missing:
        raise ValueError("Missing metric weights: " + ", ".join(missing))
    out_of_range = [metric for metric, value in mapped.items() if value < 0.005 or value > 0.35]
    if out_of_range:
        raise ValueError("Metric weights outside [0.005, 0.35]: " + ", ".join(out_of_range))
    original_sum = sum(mapped.values())
    if original_sum <= 0:
        raise ValueError("Weight sum is not positive")
    normalized = {metric: mapped[metric] / original_sum for metric in metrics}
    return normalized, {
        "original_sum": original_sum,
        "normalization_applied": not math.isclose(original_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6),
    }


def parse_weight_response(record: Mapping[str, Any], field: str, metrics: Sequence[str]) -> Dict[str, Any]:
    obj = extract_json(str(record["response"]))
    raw = obj.get(field)
    if not isinstance(raw, dict):
        raise ValueError(f"Response missing object field {field}")
    weights, audit = normalize_agent_weights(raw, metrics)
    obj[field] = weights
    obj["weight_validation"] = audit
    return obj


def validate_object_contract(
    obj: Any,
    *,
    required_types: Mapping[str, type | tuple[type, ...]],
    contract_name: str,
) -> Dict[str, Any]:
    """Validate non-numeric Agent output structure before it enters meeting state."""
    if not isinstance(obj, dict):
        raise ValueError(f"{contract_name} response must be a JSON object")
    missing = [key for key in required_types if key not in obj]
    if missing:
        raise ValueError(f"{contract_name} response missing fields: {', '.join(missing)}")
    wrong = [key for key, expected in required_types.items() if not isinstance(obj.get(key), expected)]
    if wrong:
        raise ValueError(f"{contract_name} response has invalid field types: {', '.join(wrong)}")
    return dict(obj)


def repair_weight_response(
    *, output_dir: Path, meeting: str, round_no: int, role: str,
    system_prompt: str, bad_record: Mapping[str, Any], error: Exception,
    field: str, metrics: Sequence[str], previous_weights: Mapping[str, float] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    repair_prompt = json.dumps({
        "task": "Repair the prior response into valid JSON only.",
        "validation_error": str(error),
        "required_weight_field": field,
        "exact_metric_keys": list(metrics),
        "weight_bounds": [0.005, 0.35],
        "sum": 1.0,
        "previous_weights": previous_weights,
        "prior_response": bad_record.get("response"),
    }, ensure_ascii=False, indent=2)
    repaired = call_llm(
        output_dir=output_dir, meeting=meeting, round_no=round_no,
        role=role + "_repair", system_prompt=system_prompt,
        user_prompt=repair_prompt, max_tokens=1000, temperature=0.0,
    )
    return repaired, parse_weight_response(repaired, field, metrics)


def dataset_profiles(results_dir: Path) -> List[Dict[str, Any]]:
    profiles = []
    for folder in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        csv_path = folder / "final_results_with_predictions.csv"
        if not csv_path.exists():
            continue
        rows = 0
        positives = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                try:
                    positives += int(float(row.get("True_Label", 0)) > 0.5)
                except (TypeError, ValueError):
                    pass
        profiles.append({
            "dataset": folder.name,
            "rows": rows,
            "positives": positives,
            "negatives": rows - positives,
            "positive_fraction": positives / rows if rows else None,
            "formal_independence_status": "not confirmed; model-associated provenance and homology gates remain pending",
        })
    return profiles


def literature_snapshot(results_dir: Path) -> Dict[str, Any]:
    pool_path = ROOT / "data" / "evidence_pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else {}
    return {
        "generated_at": utc_now(),
        "evidence_pool_path": "data/evidence_pool.json",
        "evidence_pool_sha256": sha256_text(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else None,
        "paper_count": pool.get("paper_count"),
        "evidence_batch_count": pool.get("evidence_batch_count"),
        "source_counts": pool.get("source_counts"),
        "metric_consensus": LITERATURE_CONSENSUS,
        "dataset_profiles": dataset_profiles(results_dir),
        "metric_definitions": METRIC_GUIDE,
    }


def build_shared_payload(
    *, meeting_kind: str, round_no: int, metrics: Sequence[str],
    snapshot: Mapping[str, Any], previous_weights: Mapping[str, float] | None,
    sampled_datasets: Sequence[str] | None,
    metric_evidence: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    # Data-minimization boundary for external LLM calls. Full provenance remains
    # in the local snapshot, but prompts contain only anonymous aggregate facts.
    alias_map = {
        profile["dataset"]: f"Dataset_{chr(65 + index)}"
        for index, profile in enumerate(snapshot["dataset_profiles"])
    }
    anonymous_profiles = [
        {
            "dataset": alias_map[profile["dataset"]],
            "rows": profile["rows"],
            "positives": profile["positives"],
            "negatives": profile["negatives"],
            "positive_fraction": profile["positive_fraction"],
            "formal_independence_status": "not confirmed; provenance and homology gates remain pending",
        }
        for profile in snapshot["dataset_profiles"]
    ]
    consensus = dict(snapshot["metric_consensus"])
    consensus.pop("verified_dataset_sources", None)
    consensus["source_id"] = "retrieved_literature_metric_consensus"
    return {
        "meeting_kind": meeting_kind,
        "round": round_no,
        "scientific_task": "Select metric weights for an AMP binary-classification benchmark spanning distinct class prevalences.",
        "exact_metric_keys": list(metrics),
        "metric_definitions": {metric: METRIC_GUIDE.get(metric, "") for metric in metrics},
        "literature_consensus": consensus,
        "evidence_pool_summary": {
            "paper_count": snapshot.get("paper_count"),
            "evidence_batch_count": snapshot.get("evidence_batch_count"),
        },
        "dataset_profiles": anonymous_profiles,
        "sampled_datasets_this_round": [alias_map.get(name, "Dataset_unknown") for name in (sampled_datasets or [])],
        "metric_evidence_this_round": metric_evidence or {},
        "previous_accepted_weights": previous_weights,
        "blinding": "No model names, model scores or Top3 identities are provided to weight-setting Agents.",
        "methodological_caveat": "The stored datasets have unresolved independence/homology gates; this meeting is exploratory post-hoc analysis, not a leakage-free preregistered benchmark.",
        "external_data_minimization": "Only anonymous aggregate dataset profiles, metric-level evidence and a compact literature consensus are sent. No sequences, file paths, DOI list, model names, scores or leaderboard positions are included.",
    }


def expert_prompt(role: str, payload: Mapping[str, Any]) -> str:
    schema = {
        "role": role,
        "analysis": "<=140 words",
        "proposed_weights": {metric: "number" for metric in payload["exact_metric_keys"]},
        "evidence_links": [{"source_type": "literature|benchmark|llm_prior", "source_id": "string", "claim": "string"}],
        "changes_from_previous": {metric: "increase|decrease|hold + reason" for metric in payload["exact_metric_keys"]},
        "uncertainties": ["string"],
    }
    return (
        ROLE_INSTRUCTIONS[role]
        + "\n\nReturn this exact JSON structure:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\nMeeting evidence:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def reviewer_prompt(payload: Mapping[str, Any], expert_outputs: Mapping[str, Any]) -> str:
    return json.dumps({
        "task": "Audit the three blinded expert weight proposals. Do not produce final weights.",
        "meeting_evidence": payload,
        "expert_outputs": expert_outputs,
        "required_output_schema": {
            "analysis": "<=160 words",
            "criticisms": ["string"],
            "required_changes": {"metric_or_issue": "required revision"},
            "preferred_directions": {"metric": "increase|decrease|hold"},
            "leakage_check": "pass|fail with reason",
            "unresolved_risks": ["string"],
        },
    }, ensure_ascii=False, indent=2)


def chief_prompt(
    payload: Mapping[str, Any], expert_outputs: Mapping[str, Any], reviewer_output: Mapping[str, Any]
) -> str:
    return json.dumps({
        "task": "Issue the accepted consensus weights after reconciling experts and Reviewer.",
        "meeting_evidence": payload,
        "expert_outputs": expert_outputs,
        "reviewer_output": reviewer_output,
        "required_output_schema": {
            "analysis": "<=180 words",
            "final_weights": {metric: "number" for metric in payload["exact_metric_keys"]},
            "consensus_summary": "string",
            "agent_disagreements": ["string"],
            "reviewer_responses": ["string"],
            "evidence_sources_by_metric": {metric: ["source_id"] for metric in payload["exact_metric_keys"]},
            "confidence": "number 0..1",
            "remaining_uncertainties": ["string"],
        },
        "hard_constraints": {
            "weight_bounds": [0.005, 0.35],
            "sum": 1.0,
            "max_l1_change_from_previous": 0.30 if payload.get("previous_accepted_weights") else None,
            "no_model_specific_priority": True,
        },
    }, ensure_ascii=False, indent=2)


def run_agent_meeting(
    *, output_dir: Path, meeting_kind: str, round_no: int,
    metrics: Sequence[str], payload: Mapping[str, Any], max_workers: int,
) -> Dict[str, Any]:
    prompts = {role: expert_prompt(role, payload) for role in ROLE_NAMES}
    raw_experts: Dict[str, Dict[str, Any]] = {}
    print(f"[meeting] {meeting_kind} round={round_no}: launching 3 expert Agents", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                call_llm,
                output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
                role=role, system_prompt=SHARED_SYSTEM, user_prompt=prompt,
                max_tokens=900, temperature=0.25,
            ): role
            for role, prompt in prompts.items()
        }
        for future in as_completed(futures):
            role = futures[future]
            raw_experts[role] = future.result()
            print(f"[meeting] {meeting_kind} round={round_no}: {role} completed", flush=True)

    experts: Dict[str, Dict[str, Any]] = {}
    for role in ROLE_NAMES:
        try:
            experts[role] = parse_weight_response(raw_experts[role], "proposed_weights", metrics)
        except Exception as exc:
            repaired_record, repaired = repair_weight_response(
                output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
                role=role, system_prompt=SHARED_SYSTEM,
                bad_record=raw_experts[role], error=exc,
                field="proposed_weights", metrics=metrics,
                previous_weights=payload.get("previous_accepted_weights"),
            )
            raw_experts[role] = repaired_record
            experts[role] = repaired

    review_prompt = reviewer_prompt(payload, experts)
    reviewer_record = call_llm(
        output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
        role="reviewer_agent", system_prompt=REVIEWER_SYSTEM,
        user_prompt=review_prompt, max_tokens=900, temperature=0.15,
    )
    try:
        reviewer = validate_object_contract(
            extract_json(reviewer_record["response"]),
            required_types={
                "analysis": str,
                "criticisms": list,
                "required_changes": dict,
                "preferred_directions": dict,
                "leakage_check": str,
                "unresolved_risks": list,
            },
            contract_name="Reviewer",
        )
    except Exception as exc:
        reviewer_record = call_llm(
            output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
            role="reviewer_agent_repair", system_prompt=REVIEWER_SYSTEM,
            user_prompt=json.dumps({
                "task": "Repair the prior Reviewer response into one valid JSON object only.",
                "validation_error": str(exc),
                "required_keys": ["analysis", "criticisms", "required_changes", "preferred_directions", "leakage_check", "unresolved_risks"],
                "prior_response": reviewer_record.get("response"),
            }, ensure_ascii=False, indent=2),
            max_tokens=900, temperature=0.0,
        )
        reviewer = validate_object_contract(
            extract_json(reviewer_record["response"]),
            required_types={
                "analysis": str,
                "criticisms": list,
                "required_changes": dict,
                "preferred_directions": dict,
                "leakage_check": str,
                "unresolved_risks": list,
            },
            contract_name="Reviewer repair",
        )
    print(f"[meeting] {meeting_kind} round={round_no}: reviewer completed", flush=True)

    final_prompt = chief_prompt(payload, experts, reviewer)
    chief_record = call_llm(
        output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
        role="chief_agent", system_prompt=CHIEF_SYSTEM,
        user_prompt=final_prompt, max_tokens=1100, temperature=0.15,
    )
    try:
        chief = parse_weight_response(chief_record, "final_weights", metrics)
    except Exception as exc:
        chief_record, chief = repair_weight_response(
            output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
            role="chief_agent", system_prompt=CHIEF_SYSTEM,
            bad_record=chief_record, error=exc,
            field="final_weights", metrics=metrics,
            previous_weights=payload.get("previous_accepted_weights"),
        )

    previous = payload.get("previous_accepted_weights")
    if previous:
        l1_change = sum(abs(chief["final_weights"][metric] - float(previous[metric])) for metric in metrics)
        if l1_change > 0.300001:
            error = ValueError(f"Chief L1 change {l1_change:.6f} exceeds 0.30")
            chief_record, chief = repair_weight_response(
                output_dir=output_dir, meeting=meeting_kind, round_no=round_no,
                role="chief_agent_delta", system_prompt=CHIEF_SYSTEM,
                bad_record=chief_record, error=error,
                field="final_weights", metrics=metrics,
                previous_weights=previous,
            )
            l1_change = sum(abs(chief["final_weights"][metric] - float(previous[metric])) for metric in metrics)
            if l1_change > 0.300001:
                raise ValueError(f"Repaired Chief L1 change remains too large: {l1_change:.6f}")
    else:
        l1_change = None
    print(f"[meeting] {meeting_kind} round={round_no}: chief completed", flush=True)

    return {
        "meeting_kind": meeting_kind,
        "round": round_no,
        "model": DEFAULT_MODEL,
        "completed_at": utc_now(),
        "input_payload": payload,
        "expert_prompts": prompts,
        "experts": experts,
        "reviewer_prompt": review_prompt,
        "reviewer": reviewer,
        "chief_prompt": final_prompt,
        "chief": chief,
        "l1_change_from_previous": l1_change,
        "accepted_weights": chief["final_weights"],
        "raw_call_ids": {
            **{role: raw_experts[role].get("request_id") for role in ROLE_NAMES},
            "reviewer_agent": reviewer_record.get("request_id"),
            "chief_agent": chief_record.get("request_id"),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_eval_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for folder in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        path = folder / "eval_result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for model, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    rows.append({
                        "dataset": folder.name,
                        "model": str(model),
                        "metric": str(metric),
                        "metric_key": str(metric).strip().lower(),
                        "value": number,
                    })
    return rows


def ranking_from_rounds(
    round_records: Sequence[Mapping[str, Any]], models: Sequence[str], rounds: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    score_rows: List[Dict[str, Any]] = []
    for record in round_records:
        for row in record["model_scores"]:
            score_rows.append(dict(row))
    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    ranks_by_model: Dict[str, List[int]] = defaultdict(list)
    top3_counts = Counter()
    for record in round_records:
        top3_counts.update(record["top3"])
    for row in score_rows:
        scores_by_model[str(row["model"])].append(float(row["score"]))
        ranks_by_model[str(row["model"])].append(int(row["rank"]))
    ranking = []
    for model in models:
        values = np.asarray(scores_by_model[model], dtype=float)
        ranks = np.asarray(ranks_by_model[model], dtype=float)
        ranking.append({
            "model": model,
            "median_score": round(float(np.median(values)), 8),
            "mean_score": round(float(np.mean(values)), 8),
            "score_iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 8),
            "median_rank": round(float(np.median(ranks)), 4),
            "mean_rank": round(float(np.mean(ranks)), 4),
            "top3_frequency": round(top3_counts[model] / rounds, 6),
            "rounds": rounds,
        })
    ranking.sort(key=lambda row: (-row["median_score"], row["mean_rank"], row["score_iqr"], row["model"].lower()))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    return ranking, score_rows


def plot_outputs(output_dir: Path, ranking: Sequence[Mapping[str, Any]], score_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    ordered = [str(row["model"]) for row in ranking]
    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    for row in score_rows:
        scores_by_model[str(row["model"])].append(float(row["score"]))
    positions = np.arange(1, len(ordered) + 1)
    height = max(6.8, 0.34 * len(ordered) + 1.6)
    fig, (ax_box, ax_bubble) = plt.subplots(
        1, 2, figsize=(12.2, height), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True
    )
    palette = plt.get_cmap("tab20")
    colors = [palette(i % 20) for i in range(len(ordered))]
    boxes = ax_box.boxplot(
        [scores_by_model[model] for model in ordered], positions=positions,
        vert=False, widths=0.62, patch_artist=True, showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#6B7280", "linewidth": 0.8},
        capprops={"color": "#6B7280", "linewidth": 0.8},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#4B5563")
        patch.set_linewidth(0.7)
    ax_box.set_yticks(positions)
    ax_box.set_yticklabels([f"{i}. {model}" for i, model in enumerate(ordered, 1)])
    ax_box.invert_yaxis()
    ax_box.set_xlabel("Weighted rank score across 50 LLM-Agent deliberation rounds")
    ax_box.set_title("A. Ranking uncertainty across Agent-selected weights", loc="left", fontweight="bold")
    ax_box.grid(axis="x", color="#E5E7EB", linewidth=0.7)

    x = np.asarray([float(row["median_score"]) for row in ranking])
    y = np.asarray([int(row["rank"]) for row in ranking])
    top3 = np.asarray([float(row["top3_frequency"]) for row in ranking])
    iqr = np.asarray([float(row["score_iqr"]) for row in ranking])
    sizes = 65.0 + 620.0 * top3
    scatter = ax_bubble.scatter(x, y, s=sizes, c=iqr, cmap="viridis_r", alpha=0.82,
                                edgecolors="#374151", linewidths=0.65)
    for row in ranking:
        ax_bubble.annotate(str(row["model"]), (float(row["median_score"]), int(row["rank"])),
                           xytext=(6, 0), textcoords="offset points", va="center", fontsize=7)
    ax_bubble.invert_yaxis()
    ax_bubble.set_yticks(positions)
    ax_bubble.set_xlabel("Median weighted rank score")
    ax_bubble.set_ylabel("Final rank")
    ax_bubble.set_title("B. Consensus rank and Top3 confidence", loc="left", fontweight="bold")
    ax_bubble.grid(color="#E5E7EB", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=ax_bubble, fraction=0.045, pad=0.03)
    cbar.set_label("Score IQR (lower is more stable)")
    for frequency in (0.25, 0.50, 0.75, 1.0):
        ax_bubble.scatter([], [], s=65.0 + 620.0 * frequency, c="#9CA3AF", alpha=0.65,
                          edgecolors="#374151", linewidths=0.65, label=f"Top3 {frequency:.0%}")
    ax_bubble.legend(title="Bubble size", loc="upper left", frameon=False, fontsize=7, title_fontsize=7)
    paths = {}
    for ext, dpi in (("png", 300), ("svg", None), ("pdf", None), ("tiff", 600)):
        path = output_dir / f"llm_agent_weight_ranking.{ext}"
        kwargs: Dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def write_trace(output_dir: Path, initial: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]], final_weights: Mapping[str, float], ranking: Sequence[Mapping[str, Any]]) -> Path:
    lines = [
        "# LLM multi-Agent metric-weight meeting trace",
        "",
        f"- Model: `{DEFAULT_MODEL}`",
        "- Roles: Literature Evidence Agent, Benchmark Statistics Agent, AMP Screening Agent, Reviewer Agent, Chief Agent",
        "- Blinding: weight-setting Agents did not receive model names, scores or Top3 identities.",
        "- Caveat: current stored datasets have unresolved independence and homology gates; this is exploratory post-hoc analysis.",
        "",
        "## Initial literature meeting",
        "",
        f"- Accepted weights: `{json.dumps(initial['accepted_weights'], ensure_ascii=False)}`",
        f"- Chief consensus: {initial['chief'].get('consensus_summary', '')}",
        "",
    ]
    for record in rounds:
        lines.extend([
            f"## Round {record['round']}", "",
            f"- Sampled datasets: {', '.join(record['sampled_datasets'])}",
            f"- Weights before: `{json.dumps(record['weights_before'], ensure_ascii=False)}`",
            f"- Literature Agent: {record['meeting']['experts']['literature_agent'].get('analysis', '')}",
            f"- Statistics Agent: {record['meeting']['experts']['statistics_agent'].get('analysis', '')}",
            f"- Screening Agent: {record['meeting']['experts']['screening_agent'].get('analysis', '')}",
            f"- Reviewer: {record['meeting']['reviewer'].get('analysis', '')}",
            f"- Chief: {record['meeting']['chief'].get('consensus_summary', '')}",
            f"- Accepted weights: `{json.dumps(record['weights_after'], ensure_ascii=False)}`",
            f"- Top3 after weights were frozen for this round: {', '.join(record['top3'])}",
            "",
        ])
    lines.extend([
        "## Final median Agent weights", "",
        f"`{json.dumps(final_weights, ensure_ascii=False)}`", "",
        "## Final model ranking", "",
        "| Rank | Model | Median score | Score IQR | Top3 frequency |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in ranking:
        lines.append(f"| {row['rank']} | {row['model']} | {row['median_score']:.4f} | {row['score_iqr']:.4f} | {row['top3_frequency']:.2%} |")
    path = output_dir / "llm_agent_weight_meeting_50_rounds.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_final_report(output_dir: Path, snapshot: Mapping[str, Any], final_weights: Mapping[str, float], ranking: Sequence[Mapping[str, Any]]) -> Path:
    system = _load_weight_prompt("research_advisor")
    payload = {
        "final_agent_weights": final_weights,
        "ranking": list(ranking),
        "dataset_profiles": snapshot["dataset_profiles"],
        "method": "initial literature meeting + 50 real LLM multi-Agent bootstrap meetings",
        "model": DEFAULT_MODEL,
        "required_sections": [
            "当前证据范围", "Agent权重共识", "跨数据集排名", "Top3候选",
            "方法学风险", "后续验证与集成建议",
        ],
    }
    record = call_llm(
        output_dir=output_dir, meeting="final_research_advisor", round_no=51,
        role="research_advisor", system_prompt=system,
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        max_tokens=2600, temperature=0.2,
    )
    path = output_dir / "amp_future_directions_report_llm_agent.md"
    path.write_text(record["response"].strip() + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rounds").mkdir(exist_ok=True)

    rows = collect_eval_rows(results_dir)
    eligible_rows, resource_gate = apply_resource_gate(rows, load_model_resource_policy())
    resource_audit_files = write_resource_gate_audit(output_dir, resource_gate)
    prepared = _prepare(eligible_rows)
    if len(prepared["datasets"]) != 3 or len(prepared["models"]) < 3 or len(prepared["metrics"]) != 12:
        raise ValueError(
            f"Expected 3 datasets, at least 3 resource-eligible models and 12 eligible metrics; got "
            f"{len(prepared['datasets'])}, {len(prepared['models'])}, {len(prepared['metrics'])}"
        )
    metrics = [prepared["names"][metric] for metric in prepared["metrics"]]
    snapshot = literature_snapshot(results_dir)
    write_json(output_dir / "literature_evidence_snapshot.json", snapshot)
    run_config = {
        "started_at": utc_now(), "model": DEFAULT_MODEL, "rounds": args.rounds,
        "seed": args.seed, "roles": list(ROLE_NAMES) + ["reviewer_agent", "chief_agent"],
        "results_dir": str(results_dir), "output_dir": str(output_dir),
        "metrics": metrics, "datasets": prepared["datasets"], "models": len(prepared["models"]),
        "blinding": "weight-setting Agents do not receive model identities or rankings",
        "resource_gate": resource_gate,
        "resource_gate_audit_files": resource_audit_files,
        "resume": args.resume,
    }
    write_json(output_dir / "run_config.json", run_config)

    initial_path = output_dir / "initial_agent_weight_meeting.json"
    if initial_path.exists() and args.resume:
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        print("[resume] initial Agent meeting loaded", flush=True)
    else:
        initial_payload = build_shared_payload(
            meeting_kind="initial_literature_meeting", round_no=0,
            metrics=metrics, snapshot=snapshot, previous_weights=None,
            sampled_datasets=None, metric_evidence=None,
        )
        initial = run_agent_meeting(
            output_dir=output_dir, meeting_kind="initial_literature_meeting",
            round_no=0, metrics=metrics, payload=initial_payload,
            max_workers=args.max_workers,
        )
        write_json(initial_path, initial)
    weights = {metric: float(initial["accepted_weights"][metric]) for metric in metrics}

    rng = np.random.default_rng(args.seed)
    # Advance the RNG deterministically for already completed round files.
    round_records: List[Dict[str, Any]] = []
    for round_no in range(1, args.rounds + 1):
        sampled = rng.choice(prepared["datasets"], size=len(prepared["datasets"]), replace=True).tolist()
        round_path = output_dir / "rounds" / f"round_{round_no:03d}.json"
        if round_path.exists() and args.resume:
            record = json.loads(round_path.read_text(encoding="utf-8"))
            weights = {metric: float(record["weights_after"][metric]) for metric in metrics}
            round_records.append(record)
            print(f"[resume] round {round_no}/50 loaded", flush=True)
            continue

        raw_evidence = _review_metric_evidence(prepared, sampled)
        evidence_by_name = {prepared["names"][key]: value for key, value in raw_evidence.items()}
        weights_before = dict(weights)
        payload = build_shared_payload(
            meeting_kind="bootstrap_weight_meeting", round_no=round_no,
            metrics=metrics, snapshot=snapshot, previous_weights=weights_before,
            sampled_datasets=sampled, metric_evidence=evidence_by_name,
        )
        meeting = run_agent_meeting(
            output_dir=output_dir, meeting_kind="bootstrap_weight_meeting",
            round_no=round_no, metrics=metrics, payload=payload,
            max_workers=args.max_workers,
        )
        weights = {metric: float(meeting["accepted_weights"][metric]) for metric in metrics}
        internal_weights = {
            key: weights[prepared["names"][key]] for key in prepared["metrics"]
        }
        scores = _score_models(prepared, sampled, internal_weights)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
        model_scores = [
            {"round": round_no, "model": model, "score": round(float(score), 10), "rank": rank}
            for rank, (model, score) in enumerate(ranked, 1)
        ]
        record = {
            "round": round_no,
            "sampled_datasets": sampled,
            "weights_before": weights_before,
            "metric_evidence": evidence_by_name,
            "meeting": meeting,
            "weights_after": weights,
            "model_scores": model_scores,
            "top3": [model for model, _ in ranked[:3]],
        }
        write_json(round_path, record)
        round_records.append(record)
        write_json(output_dir / "checkpoint.json", {
            "completed_at": utc_now(), "completed_rounds": round_no,
            "last_weights": weights, "last_round_file": str(round_path),
        })
        print(f"[complete] round {round_no}/{args.rounds}", flush=True)

    weight_rows = []
    for record in round_records:
        for metric in metrics:
            weight_rows.append({
                "round": record["round"], "metric": metric,
                "weight": record["weights_after"][metric],
                **record["metric_evidence"][metric],
            })
    final_weights = {}
    for metric in metrics:
        final_weights[metric] = statistics.median(
            float(row["weight"]) for row in weight_rows if row["metric"] == metric
        )
    final_weights = _normalize(final_weights, metrics)
    ranking, score_rows = ranking_from_rounds(round_records, prepared["models"], args.rounds)

    write_csv(output_dir / "llm_agent_metric_weights_50_rounds.csv", weight_rows,
              ["round", "metric", "weight", "coverage", "separation", "consistency", "consensus", "uniqueness", "committee_support"])
    write_csv(output_dir / "llm_agent_model_scores_50_rounds.csv", score_rows,
              ["round", "model", "score", "rank"])
    write_csv(output_dir / "llm_agent_model_ranking_50_rounds.csv", ranking,
              ["rank", "model", "median_score", "mean_score", "score_iqr", "median_rank", "mean_rank", "top3_frequency", "rounds"])
    figure_files = plot_outputs(output_dir, ranking, score_rows)
    trace_path = write_trace(output_dir, initial, round_records, final_weights, ranking)
    report_path = None if args.no_final_report else generate_final_report(output_dir, snapshot, final_weights, ranking)

    result = {
        "method": "literature- and LLM-knowledge-grounded five-role Agent deliberation with blinded dataset bootstrap",
        "model": DEFAULT_MODEL,
        "rounds": args.rounds,
        "seed": args.seed,
        "datasets": prepared["datasets"],
        "models": prepared["models"],
        "eligible_metrics": metrics,
        "resource_gate": resource_gate,
        "resource_gate_audit_files": resource_audit_files,
        "initial_agent_weights": initial["accepted_weights"],
        "final_weights_median": final_weights,
        "final_ranking": ranking,
        "figure_files": figure_files,
        "trace_file": str(trace_path),
        "research_advisor_report": str(report_path) if report_path else None,
        "raw_calls_file": str(output_dir / "raw_llm_calls.jsonl"),
        "round_files": [str(output_dir / "rounds" / f"round_{i:03d}.json") for i in range(1, args.rounds + 1)],
        "scientific_caveat": "Current dataset independence and homology gates are unresolved; results are exploratory post-hoc analysis.",
    }
    write_json(output_dir / "llm_agent_weight_meeting_50_rounds.json", result)
    write_json(output_dir / "checkpoint.json", {
        "completed_at": utc_now(), "completed_rounds": args.rounds,
        "status": "complete", "result_file": str(output_dir / "llm_agent_weight_meeting_50_rounds.json"),
    })
    print(json.dumps({
        "status": "complete", "top3": [row["model"] for row in ranking[:3]],
        "output_dir": str(output_dir),
    }, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real LLM five-role metric-weight meeting")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data" / "results_manual")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "results_manual" / "llm_agent_weight_meeting")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-final-report", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
