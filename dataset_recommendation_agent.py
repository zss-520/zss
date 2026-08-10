"""Evidence-aware dataset recommendation agent for the AMP benchmark.

The agent deliberately separates literature nomination from scientific
selection.  Literature-only candidates can enter an acquisition queue, but a
dataset can enter the final three-dataset strategy only after its real
sequences have been standardized and audited against the preregistered rules.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MEMORY_PATH = DATA_DIR / "literature_deep_research_memory.json"
EVIDENCE_POOL_PATH = DATA_DIR / "evidence_pool.json"
REQUIRED_DATASET_SEEDS_PATH = DATA_DIR / "required_benchmark_dataset_seeds.json"
POLICY_PATH = DATA_DIR / "dataset_selection_policy.json"
METADATA_PATH = DATA_DIR / "dataset_metadata.json"
SOURCE_LOCK_PATH = DATA_DIR / "dataset_source_lock.json"
POOL_PATH = DATA_DIR / "dataset_candidate_pool.json"
POOL_MD_PATH = DATA_DIR / "dataset_candidate_pool.md"
RECOMMENDATION_PATH = DATA_DIR / "dataset_agent_recommendation.json"
RECOMMENDATION_MD_PATH = DATA_DIR / "dataset_agent_recommendation.md"
AGENT_STRATEGY_PATH = DATA_DIR / "benchmark_strategy.agent.json"
MANUAL_RESULTS_DIR = DATA_DIR / "results_manual"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _missing(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, (list, dict)):
        return not value
    return str(value).strip().lower() in {
        "", "none", "null", "n/a", "na", "unknown", "not reported",
        "not_reported", "not_reported_in_available_evidence",
    }


def _first(*values: Any) -> Any:
    return next((value for value in values if not _missing(value)), "")


def _extract_url(value: Any) -> str:
    match = re.search(r"https?://[^\s<>\]\[()]+", str(value or ""))
    return match.group(0).rstrip(".,;:") if match else ""


def _canonical(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    for token in ("dataset", "data set", "benchmark", "test set", "github", "zenodo"):
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def _candidate_name_keys(candidate: dict[str, Any]) -> set[str]:
    values = [candidate.get("dataset_name"), candidate.get("canonical_name"), *(candidate.get("aliases") or [])]
    return {key for key in (_canonical(value) for value in values) if key}


def _normalize_download_url(value: Any) -> str:
    url = str(value or "").strip().rstrip(".,;:")
    zenodo_doi = re.search(r"doi\.org/10\.5281/zenodo\.(\d+)", url, flags=re.IGNORECASE)
    if zenodo_doi:
        return f"https://zenodo.org/records/{zenodo_doi.group(1)}"
    return url


def _safe_name(value: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value.strip())
    return name.replace(" ", "_") or "Unknown_Dataset"


def _normalize_candidate(raw: dict[str, Any], origin: str, *, rank: int | None = None) -> dict[str, Any]:
    name = str(_first(raw.get("dataset_name"), raw.get("name"), raw.get("dataset"), raw.get("dataset_source"), "unnamed_dataset"))
    source_text = _first(
        raw.get("dataset_source_or_link"), raw.get("dataset_url"), raw.get("download_url"),
        raw.get("source_url"), raw.get("url"), raw.get("dataset_source"),
    )
    url = _first(raw.get("dataset_url"), raw.get("download_url"), raw.get("source_url"), raw.get("url"), _extract_url(source_text))
    citation = {
        "doi": _first(raw.get("source_doi"), raw.get("doi")),
        "pmid": _first(raw.get("source_pmid"), raw.get("pmid")),
        "year": _first(raw.get("source_year"), raw.get("year")),
    }
    citation = {key: value for key, value in citation.items() if not _missing(value)}
    linked_models = raw.get("linked_models") if isinstance(raw.get("linked_models"), list) else []
    linked_model = _first(raw.get("linked_model"), raw.get("model_name"), linked_models[0] if linked_models else "")
    return {
        "candidate_id": _safe_name(name),
        "dataset_name": name,
        "canonical_name": _canonical(name),
        "source_url": _normalize_download_url(url),
        "dataset_source": str(source_text or ""),
        "linked_model": linked_model,
        "linked_models": linked_models or ([linked_model] if linked_model else []),
        "aliases": list(raw.get("aliases") or []),
        "dataset_role": _first(raw.get("recommended_role"), raw.get("dataset_role"), raw.get("role")),
        "positive_samples_evidence": _first(raw.get("positive_samples"), raw.get("positive_count")),
        "negative_samples_evidence": _first(raw.get("negative_samples"), raw.get("negative_count")),
        "deduplication_evidence": _first(raw.get("deduplication_method"), raw.get("required_cleaning")),
        "split_method": _first(raw.get("split_method")),
        "citation": citation,
        "evidence_level": _first(raw.get("evidence_level"), "unknown"),
        "quality_status": _first(raw.get("quality_status"), raw.get("status")),
        "description": _first(raw.get("why_selected"), raw.get("description"), raw.get("evidence"), raw.get("quality_notes")),
        "source_type": _first(raw.get("source_type")),
        "dataset_version": _first(raw.get("dataset_version")),
        "license": _first(raw.get("license")),
        "expected_files": list(raw.get("expected_files") or []),
        "expected_sheets": list(raw.get("expected_sheets") or []),
        "class_profile_evidence": _first(raw.get("class_profile_evidence")),
        "length_evidence": _first(raw.get("length_evidence")),
        "independent_external_test": raw.get("independent_external_test"),
        "independence_scope": _first(raw.get("independence_scope")),
        "origin": origin,
        "literature_rank": rank,
    }


def _merge_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "origin":
            origins = set(str(merged.get("origin") or "").split("+")) | set(str(value or "").split("+"))
            merged[key] = "+".join(sorted(item for item in origins if item))
        elif key == "citation":
            citation = dict(merged.get("citation") or {})
            citation.update({k: v for k, v in (value or {}).items() if not _missing(v)})
            merged[key] = citation
        elif key in {"aliases", "linked_models", "expected_files", "expected_sheets"}:
            merged[key] = sorted({str(item) for item in [*(merged.get(key) or []), *(value or [])] if str(item).strip()})
        elif key == "dataset_role" and not _missing(value):
            current = str(merged.get(key) or "").lower()
            if _missing(merged.get(key)) or current in {"training_or_benchmark_source", "dataset source"}:
                merged[key] = value
        elif key == "independent_external_test" and value is not None:
            merged[key] = value
        elif _missing(merged.get(key)) and not _missing(value):
            merged[key] = value
        elif key == "literature_rank" and value is not None:
            old = merged.get(key)
            merged[key] = min(int(old), int(value)) if old is not None else int(value)
    return merged


def _is_amp_binary_candidate(candidate: dict[str, Any]) -> bool:
    text = " ".join(str(candidate.get(key) or "") for key in (
        "dataset_name", "dataset_source", "linked_model", "dataset_role", "description"
    )).lower()
    excluded = ("anticancer", "anti-cancer", "antiviral", "anti-viral", "antifungal", "anti-fungal", "toxicity", "hemolysis", "mic regression")
    if any(term in text for term in excluded):
        return False
    amp_signal = any(term in text for term in ("amp", "antimicrobial peptide", "antimicrobial", "antibacterial peptide"))
    return amp_signal or candidate.get("literature_rank") is not None


def _evidence_score(candidate: dict[str, Any]) -> float:
    score = 0.0
    rank = candidate.get("literature_rank")
    if rank is not None:
        score += max(12.0, 24.0 - 3.0 * int(rank))
    url = str(candidate.get("source_url") or "").lower()
    if any(host in url for host in ("zenodo", "figshare", "dryad", "dataverse", "osf.io")):
        score += 6
    elif "github.com" in url or "gitlab" in url:
        score += 4
    elif url:
        score += 2
    if candidate.get("citation"):
        score += 3
    if str(candidate.get("evidence_level") or "").lower() in {"fulltext", "dataset_repository", "official"}:
        score += 3
    if "official" in str(candidate.get("evidence_level") or "").lower():
        score += 3
    if "required_verified_seed" in str(candidate.get("origin") or ""):
        # Curated seeds require both a primary-paper identity and an official
        # repository/release location.  They should outrank legacy rows whose
        # only advantage is a hard-coded literature rank, while remaining
        # acquisition candidates until real-sequence audit passes.
        score += 15
    if not _missing(candidate.get("positive_samples_evidence")):
        score += 2
    if not _missing(candidate.get("negative_samples_evidence")):
        score += 2
    if not _missing(candidate.get("deduplication_evidence")):
        score += 1
    role_text = str(candidate.get("dataset_role") or "").lower()
    if "external" in role_text or "benchmark" in role_text or "test" in role_text:
        score += 2
    if "training" in role_text and "test" not in role_text:
        score -= 4
    candidate["evidence_score"] = round(score, 3)
    return score


def discover_candidates(root: Path = ROOT) -> list[dict[str, Any]]:
    memory = _load_json(root / MEMORY_PATH.relative_to(ROOT), {})
    evidence_pool = _load_json(root / EVIDENCE_POOL_PATH.relative_to(ROOT), {})
    required_seeds = _load_json(root / REQUIRED_DATASET_SEEDS_PATH.relative_to(ROOT), {})
    inputs: list[dict[str, Any]] = []
    for index, row in enumerate(memory.get("final_recommended_datasets", []), 1):
        if isinstance(row, dict):
            is_meeting = row.get("recommendation_origin") == "literature_global_meeting_consensus"
            inputs.append(
                _normalize_candidate(
                    row,
                    "literature_meeting_shortlist" if is_meeting else "legacy_static_recommendation",
                    rank=int(row.get("dataset_rank") or index) if is_meeting else None,
                )
            )
    for row in memory.get("datasets", []):
        if isinstance(row, dict):
            inputs.append(_normalize_candidate(row, "literature_memory"))
    for row in evidence_pool.get("external_datasets", []):
        if isinstance(row, dict):
            inputs.append(_normalize_candidate(row, "evidence_pool"))
    for row in required_seeds.get("datasets", []):
        if isinstance(row, dict):
            inputs.append(_normalize_candidate(row, "required_verified_seed"))

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in inputs:
        is_meeting_shortlist = "literature_meeting_shortlist" in str(candidate.get("origin") or "")
        # A literature meeting may legitimately nominate a dataset from a primary
        # paper before a direct archive/download URL has been recovered.  Keep
        # those rows in the acquisition queue so the download gate can expose the
        # missing URL as a blocker; do not silently erase the meeting decision.
        if not _is_amp_binary_candidate(candidate) or (
            not candidate.get("source_url") and not is_meeting_shortlist
        ):
            continue
        key = candidate.get("source_url", "").lower().rstrip("/") or candidate.get("canonical_name")
        if not key:
            continue
        deduped[key] = _merge_candidate(deduped[key], candidate) if key in deduped else candidate
    rows = list(deduped.values())
    for row in rows:
        _evidence_score(row)
    return sorted(rows, key=lambda row: (-float(row.get("evidence_score", 0)), str(row.get("dataset_name", "")).casefold()))


def _read_ground_truth(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sequence", "label"}.issubset(reader.fieldnames):
            return []
        rows = []
        for raw in reader:
            sequence = re.sub(r"\s+", "", str(raw.get("sequence") or "")).upper()
            label = str(raw.get("label") or "").strip()
            if sequence and label in {"0", "1", "0.0", "1.0"}:
                rows.append({"sequence": sequence, "label": int(float(label))})
        return rows


def _audit_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_standardized", "blockers": ["ground_truth_missing_or_invalid"]}
    labels = [row["label"] for row in rows]
    positive = sum(labels)
    negative = len(labels) - positive
    lengths = [len(row["sequence"]) for row in rows]
    by_label = {label: [len(row["sequence"]) for row in rows if row["label"] == label] for label in (0, 1)}
    majority = max(positive, negative)
    minority = min(positive, negative)
    ratio = minority / majority if majority else 0.0
    threshold = float(policy.get("balanced_minority_majority_ratio_min", 0.70))
    profile = "balanced" if ratio >= threshold else "imbalanced"

    def fraction(values: Iterable[int]) -> float:
        values = list(values)
        return sum(10 <= value <= 50 for value in values) / len(values) if values else 0.0

    audit = {
        "status": "audited",
        "row_count": len(rows),
        "positive_count": positive,
        "negative_count": negative,
        "positive_fraction": positive / len(rows),
        "minority_fraction": minority / len(rows),
        "minority_majority_ratio": ratio,
        "observed_profile": profile,
        "length": {
            "min_aa": min(lengths),
            "max_aa": max(lengths),
            "median_aa": statistics.median(lengths),
            "fraction_10_50_aa": fraction(lengths),
            "positive_median_aa": statistics.median(by_label[1]),
            "negative_median_aa": statistics.median(by_label[0]),
            "positive_fraction_10_50_aa": fraction(by_label[1]),
            "negative_fraction_10_50_aa": fraction(by_label[0]),
        },
        "within_dataset_duplicate_count": len(rows) - len({row["sequence"] for row in rows}),
        "blockers": [],
    }
    blockers = audit["blockers"]
    if len(rows) < int(policy.get("min_total_samples", 500)):
        blockers.append("insufficient_total_samples")
    if minority < int(policy.get("min_samples_per_class", 100)):
        blockers.append("insufficient_class_samples")
    if min(fraction(lengths), fraction(by_label[0]), fraction(by_label[1])) < float(policy.get("min_primary_length_fraction", 0.80)):
        blockers.append("primary_length_coverage_below_threshold")
    if min(lengths) < int(policy.get("absolute_min_length_aa", 5)) or max(lengths) > int(policy.get("absolute_max_length_aa", 100)):
        blockers.append("absolute_length_out_of_range")
    median_gap = abs(statistics.median(by_label[1]) - statistics.median(by_label[0]))
    if median_gap > float(policy.get("max_class_median_length_gap_aa", 15)):
        blockers.append("class_length_distribution_mismatch")
    if audit["within_dataset_duplicate_count"]:
        blockers.append("within_dataset_duplicates")
    audit["structural_status"] = "passed" if not blockers else "failed"
    return audit


def _local_audit(dataset_dir: Path, policy: dict[str, Any]) -> dict[str, Any]:
    return _audit_rows(_read_ground_truth(dataset_dir / "ground_truth.csv"), policy)


def _read_manual_prediction_ground_truth(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        sequence_column = next(
            (field for field in fields if _canonical(field) in {"standard id", "sequence", "seq"}),
            None,
        )
        label_column = next(
            (field for field in fields if _canonical(field) in {"true label", "label", "amp label", "y true"}),
            None,
        )
        if not sequence_column or not label_column:
            return []
        rows = []
        for raw in reader:
            sequence = re.sub(r"\s+", "", str(raw.get(sequence_column) or "")).upper()
            label = str(raw.get(label_column) or "").strip()
            if sequence and label in {"0", "1", "0.0", "1.0"}:
                rows.append({"sequence": sequence, "label": int(float(label))})
        return rows


def _metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(root / METADATA_PATH.relative_to(ROOT), {})
    rows = payload.get("datasets", []) if isinstance(payload, dict) else []
    return {
        _canonical(_first(row.get("local_dataset_name"), row.get("dataset_name"))): row
        for row in rows if isinstance(row, dict)
    }


def _source_locks(root: Path) -> dict[str, Any]:
    payload = _load_json(root / SOURCE_LOCK_PATH.relative_to(ROOT), {})
    return payload.get("datasets", {}) if isinstance(payload, dict) else {}


def audit_local_candidates(root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    datasets_dir = root / "data" / "datasets"
    metadata = _metadata_index(root)
    locks = _source_locks(root)
    outputs: list[dict[str, Any]] = []
    if not datasets_dir.is_dir():
        return outputs
    for directory in sorted((path for path in datasets_dir.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        meta = metadata.get(_canonical(directory.name), {})
        candidate = _normalize_candidate({**meta, "dataset_name": directory.name}, "local_audit")
        candidate["local_dataset_name"] = directory.name
        candidate["local_path"] = directory.relative_to(root).as_posix()
        candidate["expected_sha256"] = meta.get("expected_sha256") or locks.get(directory.name)
        candidate["audit"] = _local_audit(directory, policy)
        candidate["selection_profile"] = candidate["audit"].get("observed_profile")
        outputs.append(candidate)
    return outputs


def audit_manual_evaluated_candidates(root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    results_root = root / MANUAL_RESULTS_DIR.relative_to(ROOT)
    seeds_doc = _load_json(root / REQUIRED_DATASET_SEEDS_PATH.relative_to(ROOT), {})
    seeds = [row for row in seeds_doc.get("datasets", []) if isinstance(row, dict)]
    alias_index: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for key in _candidate_name_keys(seed):
            alias_index[key] = seed
    outputs: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return outputs
    for directory in sorted((path for path in results_root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        seed = alias_index.get(_canonical(directory.name))
        prediction_path = directory / "final_results_with_predictions.csv"
        if seed is None or not prediction_path.is_file():
            continue
        candidate = _normalize_candidate(
            {**seed, "dataset_name": directory.name},
            "manual_evaluation_audit+required_verified_seed",
        )
        candidate["local_dataset_name"] = directory.name
        candidate["local_path"] = prediction_path.relative_to(root).as_posix()
        candidate["audit"] = _audit_rows(_read_manual_prediction_ground_truth(prediction_path), policy)
        candidate["selection_profile"] = candidate["audit"].get("observed_profile")
        candidate["manual_evaluation_available"] = True
        candidate["evaluation_scope"] = (
            "Empirically evaluated candidate; model-specific training overlap and homology audits remain mandatory."
        )
        outputs.append(candidate)
    return outputs


def _formal_metadata_blockers(candidate: dict[str, Any], root: Path, policy: dict[str, Any]) -> list[str]:
    blockers = list((candidate.get("audit") or {}).get("blockers") or [])
    requirements = (
        ("require_source_url", "source_url", "source_url_missing"),
        ("require_citation", "citation", "citation_missing"),
        ("require_version", "dataset_version", "version_missing"),
        ("require_retrieval_date", "retrieved_at_utc", "retrieval_date_missing"),
        ("require_license", "license", "license_missing"),
        ("require_expected_sha256", "expected_sha256", "sha256_missing"),
        ("require_training_references", "training_reference_paths", "training_references_missing"),
        ("require_label_definition", "label_definition", "label_definition_missing"),
        ("require_negative_sampling_strategy", "negative_sampling_strategy", "negative_sampling_strategy_missing"),
    )
    for policy_key, field, issue in requirements:
        if policy.get(policy_key, False) and _missing(candidate.get(field)):
            blockers.append(issue)
    if policy.get("require_independent_external_test", False) and candidate.get("independent_external_test") is not True:
        blockers.append("independent_external_test_not_confirmed")
    if policy.get("require_low_homology_report", False):
        raw = str(candidate.get("homology_report_path") or "").strip()
        path = Path(raw) if raw else None
        if path and not path.is_absolute():
            path = root / path
        if not path or not path.is_file():
            blockers.append("low_homology_report_missing")
    return sorted(set(blockers))


def _best_three(candidates: list[dict[str, Any]], policy: dict[str, Any], *, formal: bool) -> list[dict[str, Any]]:
    balanced = [row for row in candidates if (row.get("audit") or {}).get("observed_profile") == "balanced"]
    imbalanced = [row for row in candidates if (row.get("audit") or {}).get("observed_profile") == "imbalanced"]
    if formal:
        balanced = [row for row in balanced if not row.get("formal_blockers")]
        imbalanced = [row for row in imbalanced if not row.get("formal_blockers")]
    else:
        balanced = [row for row in balanced if (row.get("audit") or {}).get("status") == "audited"]
        imbalanced = [row for row in imbalanced if (row.get("audit") or {}).get("status") == "audited"]
    min_gap = float(policy.get("min_imbalanced_minority_fraction_gap", 0.10))
    combinations: list[tuple[float, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = []
    for one, pair in itertools.product(balanced, itertools.combinations(imbalanced, 2)):
        gap = abs(float(pair[0]["audit"]["minority_fraction"]) - float(pair[1]["audit"]["minority_fraction"]))
        if gap < min_gap:
            continue
        rows = (one, pair[0], pair[1])
        penalty = sum(5 * len(row.get("formal_blockers") or []) for row in rows)
        measurable_penalty = sum(15 * len((row.get("audit") or {}).get("blockers") or []) for row in rows)
        score = sum(float(row.get("evidence_score", 0)) for row in rows) + gap * 20 - penalty - measurable_penalty
        combinations.append((score, rows))
    if not combinations:
        return []
    _, best = max(combinations, key=lambda item: item[0])
    return [dict(row) for row in best]


def _strategy_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": candidate.get("local_dataset_name") or candidate.get("dataset_name"),
        "role": f"{candidate.get('selection_profile')}_external_test",
        "selection_profile": candidate.get("selection_profile"),
        "description": candidate.get("description") or "Selected by Dataset Recommendation Agent from audited candidates.",
        "download_url": candidate.get("source_url") or "",
        "expected_sha256": candidate.get("expected_sha256"),
        "citation": candidate.get("citation"),
        "dataset_version": candidate.get("dataset_version") or "",
        "retrieved_at_utc": candidate.get("retrieved_at_utc") or "",
        "license": candidate.get("license") or "",
        "independent_external_test": candidate.get("independent_external_test"),
        "label_definition": candidate.get("label_definition") or "",
        "negative_sampling_strategy": candidate.get("negative_sampling_strategy") or "",
        "homology_report_path": candidate.get("homology_report_path") or "",
        "training_reference_paths": candidate.get("training_reference_paths") or [],
        "agent_selection_evidence": {
            "observed_profile": (candidate.get("audit") or {}).get("observed_profile"),
            "minority_majority_ratio": (candidate.get("audit") or {}).get("minority_majority_ratio"),
            "fraction_10_50_aa": ((candidate.get("audit") or {}).get("length") or {}).get("fraction_10_50_aa"),
        },
    }


def recommend(root: Path = ROOT) -> dict[str, Any]:
    policy_payload = _load_json(root / POLICY_PATH.relative_to(ROOT), {})
    policy = policy_payload.get("dataset_selection_policy", policy_payload)
    if not isinstance(policy, dict) or not policy.get("enabled"):
        raise RuntimeError(f"数据集选择策略未启用: {root / POLICY_PATH.relative_to(ROOT)}")

    literature = discover_candidates(root)
    local = audit_local_candidates(root, policy) + audit_manual_evaluated_candidates(root, policy)
    literature_by_name: dict[str, dict[str, Any]] = {}
    for evidence_row in literature:
        for key in _candidate_name_keys(evidence_row):
            current = literature_by_name.get(key)
            if current is None or float(evidence_row.get("evidence_score", 0)) > float(current.get("evidence_score", 0)):
                literature_by_name[key] = evidence_row
    for row in local:
        local_name = row.get("local_dataset_name")
        evidence = next((literature_by_name.get(key) for key in _candidate_name_keys(row) if literature_by_name.get(key)), None)
        if evidence:
            merged = _merge_candidate(evidence, row)
            row.clear()
            row.update(merged)
            row["dataset_name"] = local_name
            row["local_dataset_name"] = local_name
        _evidence_score(row)
        row["formal_blockers"] = _formal_metadata_blockers(row, root, policy)
        row["formal_eligible"] = not row["formal_blockers"]

    meeting_shortlist = sorted(
        [row for row in literature if "literature_meeting_shortlist" in str(row.get("origin") or "")],
        key=lambda row: (int(row.get("literature_rank") or 999), -float(row.get("evidence_score", 0))),
    )
    meeting_name_keys = set().union(*(_candidate_name_keys(row) for row in meeting_shortlist)) if meeting_shortlist else set()
    meeting_urls = {str(row.get("source_url") or "").lower().rstrip("/") for row in meeting_shortlist}
    meeting_local = [
        row for row in local
        if _candidate_name_keys(row) & meeting_name_keys
        or str(row.get("source_url") or "").lower().rstrip("/") in meeting_urls
    ]
    formal_selection = _best_three(meeting_local, policy, formal=True)
    provisional_selection = _best_three(meeting_local, policy, formal=False)
    empirically_evaluated = [row for row in local if row.get("manual_evaluation_available")]
    empirically_evaluated_top3 = _best_three(empirically_evaluated, policy, formal=False)
    remaining_queue = [
        row for row in literature
        if row.get("source_url") and row.get("canonical_name") not in {candidate.get("canonical_name") for candidate in local}
        and "literature_meeting_shortlist" not in str(row.get("origin") or "")
    ]
    acquisition_queue = (meeting_shortlist + remaining_queue)[:10]
    provisional_top3 = meeting_shortlist[:3]
    strategy_written = False
    strategy_path = root / AGENT_STRATEGY_PATH.relative_to(ROOT)
    if formal_selection:
        strategy = {
            "schema_version": 2,
            "selection_origin": "literature_global_meeting_consensus_then_dataset_recommendation_agent_audit",
            "generated_at_utc": _utc_now(),
            "candidate_pool_path": POOL_PATH.relative_to(ROOT).as_posix(),
            "recommendation_manifest_path": RECOMMENDATION_PATH.relative_to(ROOT).as_posix(),
            "include_unlisted_local_datasets": False,
            "dataset_selection_policy": policy,
            "recommended_datasets": [_strategy_row(row) for row in formal_selection],
        }
        _write_json(strategy_path, strategy)
        strategy_written = True
    elif strategy_path.exists():
        strategy_path.unlink()

    payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "selection_method": "literature global meeting shortlist -> evidence enrichment -> real-sequence audit -> constrained 1 balanced + 2 imbalanced optimization",
        "policy_path": POLICY_PATH.relative_to(ROOT).as_posix(),
        "candidate_pool_size": len(literature) + len(local),
        "literature_candidate_count": len(literature),
        "audited_local_candidate_count": len(local),
        "recommendation_origin": "literature_global_meeting_consensus",
        "meeting_shortlist_status": "ready_for_acquisition" if len(meeting_shortlist) == 3 else "missing_or_incomplete",
        "meeting_shortlist_count": len(meeting_shortlist),
        "meeting_shortlist": meeting_shortlist,
        "formal_selection_status": (
            "selected"
            if formal_selection
            else "blocked_meeting_shortlist_missing_or_incomplete"
            if len(meeting_shortlist) != 3
            else "blocked_no_three_formally_eligible_meeting_datasets"
        ),
        "formal_selection": formal_selection,
        "provisional_audited_selection": provisional_selection,
        "empirically_evaluated_top3": empirically_evaluated_top3,
        "empirically_evaluated_top3_status": (
            "selected_pending_independence_and_homology_gates"
            if len(empirically_evaluated_top3) == 3
            else "insufficient_complementary_manual_datasets"
        ),
        "provisional_acquisition_top3": provisional_top3,
        "acquisition_queue": acquisition_queue,
        "strategy_written": strategy_written,
        "strategy_path": AGENT_STRATEGY_PATH.relative_to(ROOT).as_posix() if strategy_written else None,
        "audited_candidates": local,
    }
    pool_payload = {
        "schema_version": 1,
        "generated_at_utc": payload["generated_at_utc"],
        "literature_candidates": literature,
        "audited_local_candidates": local,
    }
    _write_json(root / POOL_PATH.relative_to(ROOT), pool_payload)
    _write_json(root / RECOMMENDATION_PATH.relative_to(ROOT), payload)
    _write_markdown(root, pool_payload, payload)
    return payload


def _write_markdown(root: Path, pool: dict[str, Any], result: dict[str, Any]) -> None:
    literature = pool.get("literature_candidates", [])
    pool_lines = [
        "# 数据集候选池",
        "",
        f"Generated: `{pool.get('generated_at_utc')}`",
        "",
        "| 排名 | 数据集 | URL | 证据分 | 状态 |",
        "|---:|---|---|---:|---|",
    ]
    for index, row in enumerate(literature[:100], 1):
        pool_lines.append(f"| {index} | {row.get('dataset_name')} | {row.get('source_url')} | {row.get('evidence_score')} | 需要下载与真实序列审计 |")
    (root / POOL_MD_PATH.relative_to(ROOT)).write_text("\n".join(pool_lines) + "\n", encoding="utf-8")

    lines = [
        "# Dataset Agent 数据集推荐",
        "",
        f"- 推荐来源：**{result.get('recommendation_origin')}**",
        f"- 文献会议 Top 3 状态：**{result.get('meeting_shortlist_status')}**（{result.get('meeting_shortlist_count', 0)}/3）",
        f"- 正式选集状态：**{result.get('formal_selection_status')}**",
        f"- 候选池规模：{result.get('candidate_pool_size')}",
        f"- 已审计本地候选：{result.get('audited_local_candidate_count')}",
        f"- 是否生成 Agent strategy：{result.get('strategy_written')}",
        "",
        "## 已有评测结果支持的互补三数据集",
        "",
        f"状态：**{result.get('empirically_evaluated_top3_status')}**。该组合由真实标签分布动态选择，不是固定名称模板；正式独立外测仍需训练重叠和同源性门禁。",
        "",
    ]
    for index, row in enumerate(result.get("empirically_evaluated_top3", []), 1):
        audit = row.get("audit") or {}
        lines.append(
            f"{index}. **{row.get('dataset_name')}** — profile={audit.get('observed_profile')}, "
            f"positive_fraction={audit.get('positive_fraction')}, ratio={audit.get('minority_majority_ratio')}"
        )
    lines.extend([
        "",
        "## 优先下载与审计的 3 个候选",
        "",
        "这些是下载和真实序列审计优先项，尚不是正式 benchmark 数据集。",
        "",
    ])
    for index, row in enumerate(result.get("provisional_acquisition_top3", []), 1):
        lines.append(f"{index}. **{row.get('dataset_name')}** — {row.get('source_url')} (evidence score={row.get('evidence_score')})")
    if not result.get("provisional_acquisition_top3"):
        lines.append("- 文献会议尚未生成可用 Top 3；请先重新运行 literature global meeting。核验种子不会自动替代会议决定。")
    lines.extend(["", "## 已审计本地候选", ""])
    for row in result.get("audited_candidates", []):
        audit = row.get("audit") or {}
        lines.append(
            f"- **{row.get('dataset_name')}**: profile={audit.get('observed_profile')}, "
            f"ratio={audit.get('minority_majority_ratio')}, 10-50 aa={((audit.get('length') or {}).get('fraction_10_50_aa'))}, "
            f"formal blockers={', '.join(row.get('formal_blockers') or []) or 'none'}"
        )
    (root / RECOMMENDATION_MD_PATH.relative_to(ROOT)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(result: dict[str, Any]) -> None:
    print("\n========== Dataset Recommendation Agent ==========")
    print("候选池:", result.get("candidate_pool_size"))
    print("推荐来源:", result.get("recommendation_origin"))
    print("文献会议 Top 3:", f"{result.get('meeting_shortlist_count', 0)}/3", result.get("meeting_shortlist_status"))
    print("正式三数据集状态:", result.get("formal_selection_status"))
    print("正式 strategy:", result.get("strategy_path") or "未生成")
    print("\n已有评测结果支持的互补三数据集:", result.get("empirically_evaluated_top3_status"))
    for index, row in enumerate(result.get("empirically_evaluated_top3", []), 1):
        audit = row.get("audit") or {}
        print(
            f" {index}. {row.get('dataset_name')} | {audit.get('observed_profile')} "
            f"ratio={audit.get('minority_majority_ratio', 0):.3f}"
        )
    print("\n优先下载与审计的 3 个候选:")
    for index, row in enumerate(result.get("provisional_acquisition_top3", []), 1):
        print(f" {index}. {row.get('dataset_name')} | score={row.get('evidence_score')} | {row.get('source_url')}")
    if not result.get("provisional_acquisition_top3"):
        print(" - 尚无会议生成的 Top 3。请先运行文献菜单 16 重新开会；固定种子不会顶替。")
    print("\n已审计本地候选:")
    for row in result.get("audited_candidates", []):
        audit = row.get("audit") or {}
        print(
            f" - {row.get('dataset_name')}: {audit.get('observed_profile')} "
            f"ratio={audit.get('minority_majority_ratio', 0):.3f} "
            f"len10-50={((audit.get('length') or {}).get('fraction_10_50_aa') or 0):.2%} "
            f"blockers={len(row.get('formal_blockers') or [])}"
        )
    print("报告:", RECOMMENDATION_MD_PATH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMP benchmark dataset recommendation agent")
    parser.add_argument("command", choices=["recommend", "status"], nargs="?", default="recommend")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "status":
        result = _load_json(root / RECOMMENDATION_PATH.relative_to(ROOT), {})
        if not result:
            print("尚未生成 Dataset Agent 推荐；请先运行 recommend。")
            return 2
        _print_summary(result)
        return 0
    result = recommend(root)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
