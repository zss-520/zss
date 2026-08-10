"""Deterministic, tiered model selection for an AMP benchmark portfolio."""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Callable, Iterable


CLASSIC_BASELINE_ANCHORS = [
    {
        "name": "AMP Scanner v2",
        "aliases": ["AMP Scanner", "AMPScanner V2", "Antimicrobial Peptide Scanner vr.2"],
        "year": 2018,
        "repository": "https://github.com/dan-veltri/amp-scanner-v2",
        "reason": "经典 CNN/LSTM 序列基线",
    },
    {
        "name": "Macrel",
        "aliases": ["BigDataBiology/macrel"],
        "year": 2020,
        "repository": "https://github.com/BigDataBiology/macrel",
        "reason": "可解释理化特征与随机森林工程基线",
    },
    {
        "name": "amPEPpy",
        "aliases": ["amPEPpy 1.0"],
        "year": 2020,
        "repository": "https://github.com/tlawrence3/amPEPpy",
        "reason": "便携式随机森林基线",
    },
    {
        "name": "AMPlify",
        "aliases": ["bcgsc/AMPlify"],
        "year": 2021,
        "repository": "https://github.com/bcgsc/AMPlify",
        "reason": "BiLSTM 与注意力经典深度学习基线",
    },
    {
        "name": "Co-AMPpred",
        "aliases": ["CoAMPpred", "onkarS23/CoAMPpred"],
        "year": 2021,
        "repository": "https://github.com/onkarS23/CoAMPpred",
        "reason": "手工特征与传统机器学习对照",
    },
    {
        "name": "Deep-AmPEP30",
        "aliases": ["DeepAmPEP30"],
        "year": 2020,
        "reason": "短肽 CNN 经典基线",
    },
]


# This is a search/selection watchlist, not a declaration that a model remains SOTA.
# Entries must still have local evidence and runnable code before formal selection.
RECENT_SOTA_WATCHLIST = [
    {
        "name": "CG-AMP",
        "aliases": ["ghli16/CG-AMP"],
        "year": 2025,
        "source_doi": "10.1038/s41598-025-29666-z",
        "repository": "https://github.com/ghli16/CG-AMP",
    },
    {
        "name": "deepAMPNet",
        "aliases": ["Iseeu233/deepAMPNet"],
        "year": 2024,
        "source_pmid": "39040937",
        "repository": "https://github.com/Iseeu233/deepAMPNet",
    },
    {
        "name": "UniproLcad",
        "aliases": ["harkic/UniproLcad"],
        "year": 2024,
        "source_doi": "10.3390/sym16040464",
        "repository": "https://github.com/harkic/UniproLcad",
    },
    {
        "name": "PepNet",
        "aliases": [],
        "year": 2024,
        "source_doi": "10.1038/s42003-024-06911-1",
        "repository": "",
    },
]


REQUIRED_ARCHITECTURES = [
    "machine_learning_models",
    "cnn_dominant_models",
    "rnn_lstm_dominant_models",
    "cnn_rnn_hybrid_models",
    "transformer_llm_dominant_models",
    "gnn_models",
    "pipeline_or_ensemble_frameworks",
]


ROLE_LABELS = {
    "verified_required_core": "已核验核心候选",
    "classic_baseline": "经典基线",
    "recent_sota_candidate": "近期 SOTA 候选",
    "architecture_representative": "架构代表",
    "evidence_ranked_fill": "证据排序补位",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _row_name(row: dict[str, Any]) -> str:
    return str(row.get("model_name") or row.get("canonical_name") or row.get("name") or "").strip()


def _row_keys(row: dict[str, Any]) -> set[str]:
    values = [_row_name(row), row.get("canonical_name"), row.get("code_repository_url")]
    keys = {_norm(value) for value in values if _norm(value)}
    repo = str(row.get("code_repository_url") or row.get("repo_url") or "").rstrip("/")
    if repo:
        keys.add(_norm(repo.split("/")[-1]))
    return keys


def _anchor_for(row: dict[str, Any], anchors: list[dict[str, Any]]) -> dict[str, Any] | None:
    keys = _row_keys(row)
    for anchor in anchors:
        anchor_keys = {_norm(anchor["name"]), *[_norm(x) for x in anchor.get("aliases", [])]}
        repository = str(anchor.get("repository") or "").rstrip("/")
        if repository:
            anchor_keys.add(_norm(repository.split("/")[-1]))
        if keys & {x for x in anchor_keys if x}:
            return anchor
    return None


def _direct_anchor_name_match(row: dict[str, Any], anchor: dict[str, Any]) -> bool:
    name_key = _norm(_row_name(row))
    return name_key in {
        _norm(anchor.get("name")),
        *[_norm(alias) for alias in anchor.get("aliases", [])],
    }


def _selection_key(row: dict[str, Any]) -> str:
    anchor = _anchor_for(row, CLASSIC_BASELINE_ANCHORS + RECENT_SOTA_WATCHLIST)
    if anchor:
        return "anchor:" + _norm(anchor.get("name"))
    return "model:" + _norm(_row_name(row))


def publication_year(row: dict[str, Any]) -> int | None:
    for key in ["source_year", "publication_year", "year", "published_year", "publication_date"]:
        value = row.get(key)
        if value is None:
            continue
        match = re.search(r"(?:19|20)\d{2}", str(value))
        if match:
            return int(match.group(0))
    return None


def _default_score(row: dict[str, Any]) -> float:
    for key in ["deployment_selection_score", "evidence_score", "score", "article_impact_score"]:
        try:
            return float(row.get(key) or 0.0)
        except Exception:
            continue
    return 0.0


def _has_code(row: dict[str, Any]) -> bool:
    value = str(row.get("code_repository_url") or row.get("repo_url") or "").strip().lower()
    return value not in {"", "none", "null", "unknown", "not_reported_in_available_evidence"}


def _has_verified_code(row: dict[str, Any], watch: dict[str, Any] | None = None) -> bool:
    if not _has_code(row):
        return False
    repo = str(row.get("code_repository_url") or row.get("repo_url") or "")
    official_repo = str((watch or {}).get("repository") or "")
    if official_repo and _norm(official_repo) in _norm(repo):
        return True
    blockers = " ".join(str(x) for x in (row.get("blocking_issues") or [])).casefold()
    evidence_level = str(row.get("evidence_level") or "").casefold()
    if row.get("needs_manual_verification") is True:
        return False
    if any(
        token in blockers
        for token in [
            "no code",
            "no_code",
            "code not",
            "code missing",
            "candidate_requires_manual_verification",
        ]
    ):
        return False
    if evidence_level in {"github_search", "qwen_max_web_search", "web_search_candidate"}:
        return False
    return True


def _sota_claim_evidence(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in ["candidate_reason", "evidence", "benchmark_implications", "agent_registry_note"]
    ).casefold()
    claim = any(token in text for token in ["state-of-the-art", "state of the art", "sota", "outperform"])
    independent = any(token in text for token in ["independent", "external", "held-out", "held out"])
    return claim and independent


def _is_amp_binary_prediction(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in [
            "model_name",
            "canonical_name",
            "task_type",
            "method_family",
            "candidate_reason",
            "architecture_or_algorithm",
        ]
    ).casefold()
    if any(
        token in text
        for token in [
            "generation",
            "generative",
            "generator",
            "design",
            "mic regression",
            "mic prediction",
            "toxicity",
            "hemolysis",
            "anticancer",
            "antiviral",
            "antifungal",
        ]
    ):
        return False
    has_amp = "antimicrobial peptide" in text or re.search(r"\bamp\b", text) is not None or "amp" in _norm(_row_name(row))
    has_prediction_task = any(
        token in text
        for token in ["prediction", "predictor", "classification", "classifier", "identification", "recognition"]
    )
    return bool(has_amp and has_prediction_task)


def build_benchmark_portfolio(
    models: Iterable[dict[str, Any]],
    *,
    current_year: int | None = None,
    max_models: int = 20,
    classic_min: int = 3,
    recent_sota_min: int = 3,
    recent_window_years: int = 2,
    score_fn: Callable[[dict[str, Any]], float] | None = None,
    required_core_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Select a benchmark portfolio with hard baseline and recent-model layers."""
    current_year = current_year or _dt.datetime.now().year
    recent_cutoff = current_year - recent_window_years
    classic_cutoff = current_year - 5
    score_fn = score_fn or _default_score

    deduped_by_key: dict[str, dict[str, Any]] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        key = _selection_key(item)
        if not _norm(_row_name(item)):
            continue
        old = deduped_by_key.get(key)
        quality = (
            int(_has_verified_code(item, _anchor_for(item, CLASSIC_BASELINE_ANCHORS + RECENT_SOTA_WATCHLIST))),
            int(_is_amp_binary_prediction(item)),
            score_fn(item),
        )
        old_quality = (
            int(_has_verified_code(old, _anchor_for(old, CLASSIC_BASELINE_ANCHORS + RECENT_SOTA_WATCHLIST))),
            int(_is_amp_binary_prediction(old)),
            score_fn(old),
        ) if old else (-1, -1, -1.0)
        if old is None or quality > old_quality:
            deduped_by_key[key] = dict(item)
    deduped = list(deduped_by_key.values())

    selected: list[dict[str, Any]] = []
    selected_by_key: dict[str, dict[str, Any]] = {}
    required_core_names = [str(name).strip() for name in (required_core_names or []) if str(name).strip()]

    def add(row: dict[str, Any], role: str, reason: str) -> None:
        key = _selection_key(row)
        if not key:
            return
        if key in selected_by_key:
            existing = selected_by_key[key]
            roles = existing.setdefault("benchmark_roles", [])
            if role not in roles:
                roles.append(role)
            return
        if len(selected) >= max_models:
            return
        out = dict(row)
        out["benchmark_role"] = role
        out["benchmark_role_label"] = ROLE_LABELS[role]
        out["benchmark_roles"] = [role]
        out["benchmark_role_reason"] = reason
        out["publication_year"] = publication_year(out)
        selected.append(out)
        selected_by_key[key] = out

    required_selected: set[str] = set()
    for wanted in required_core_names:
        wanted_key = _norm(wanted)
        matches = [
            row for row in deduped
            if wanted_key in _row_keys(row)
            and _is_amp_binary_prediction(row)
            and _has_verified_code(row)
        ]
        if not matches:
            continue
        row = sorted(matches, key=score_fn, reverse=True)[0]
        add(
            row,
            "verified_required_core",
            "用户指定的核心覆盖候选，已通过科学身份与代码证据门禁；名次仍由统一评分决定",
        )
        required_selected.add(wanted)

    classic_rows = []
    for row in deduped:
        anchor = _anchor_for(row, CLASSIC_BASELINE_ANCHORS)
        if anchor and _direct_anchor_name_match(row, anchor) and _has_verified_code(row, anchor):
            priority = next(
                (i for i, item in enumerate(CLASSIC_BASELINE_ANCHORS) if item is anchor),
                len(CLASSIC_BASELINE_ANCHORS),
            )
            classic_rows.append((row, anchor, priority))
    classic_rows.sort(key=lambda item: (item[2], -score_fn(item[0])))
    for row, anchor, _ in classic_rows[:classic_min]:
        classic_row = dict(row)
        # A classic anchor identifies the original model publication.  A later
        # paper that evaluates or reuses the model must not replace that year.
        classic_row["publication_year"] = anchor.get("year") or publication_year(row)
        add(
            classic_row,
            "classic_baseline",
            anchor.get("reason") or f"发表于 {publication_year(row)} 年且有可运行代码的稳定历史基线",
        )

    recent_rows = []
    for row in deduped:
        watch = _anchor_for(row, RECENT_SOTA_WATCHLIST)
        year = publication_year(row) or ((watch or {}).get("year"))
        if not (watch or _is_amp_binary_prediction(row)) or not _has_verified_code(row, watch) or year is None or year < recent_cutoff:
            continue
        if watch or _sota_claim_evidence(row):
            recent_rows.append((row, watch, year))
    recent_rows.sort(key=lambda item: (item[2], score_fn(item[0])), reverse=True)
    for row, watch, year in recent_rows[:recent_sota_min]:
        recent_row = dict(row)
        recent_row["publication_year"] = year
        add(
            recent_row,
            "recent_sota_candidate",
            f"{year} 年近期模型，有代码及 SOTA/外部测试证据线索；SOTA 身份须由本 benchmark 重新验证",
        )

    for architecture in REQUIRED_ARCHITECTURES:
        matches = [
            row
            for row in deduped
            if row.get("architecture_category") == architecture
            and _is_amp_binary_prediction(row)
            and _has_verified_code(row)
        ]
        if not matches:
            continue
        matches.sort(key=score_fn, reverse=True)
        add(matches[0], "architecture_representative", f"覆盖架构分层：{architecture}")

    for row in sorted(deduped, key=score_fn, reverse=True):
        if (
            _selection_key(row) not in selected_by_key
            and _is_amp_binary_prediction(row)
            and _has_verified_code(row)
        ):
            add(row, "evidence_ranked_fill", "在满足基线、近期模型和架构覆盖后按复现证据与影响力补位")

    # Quotas decide membership, not rank. Re-sort by the common evidence score
    # so required-core models receive a slot without being pinned to positions 1-3.
    selected.sort(key=lambda row: (score_fn(row), _norm(_row_name(row))), reverse=True)

    role_counts = {
        role: sum(1 for row in selected if role in row.get("benchmark_roles", []))
        for role in ROLE_LABELS
    }
    covered_architectures = sorted(
        {str(row.get("architecture_category")) for row in selected if row.get("architecture_category")}
    )
    gaps: list[dict[str, Any]] = []
    if role_counts["classic_baseline"] < classic_min:
        gaps.append(
            {
                "type": "classic_baseline_shortfall",
                "required": classic_min,
                "selected": role_counts["classic_baseline"],
                "recommended_search_names": [item["name"] for item in CLASSIC_BASELINE_ANCHORS],
            }
        )
    if role_counts["recent_sota_candidate"] < recent_sota_min:
        gaps.append(
            {
                "type": "recent_sota_candidate_shortfall",
                "required": recent_sota_min,
                "selected": role_counts["recent_sota_candidate"],
                "recent_cutoff_year": recent_cutoff,
                "recommended_search_names": [item["name"] for item in RECENT_SOTA_WATCHLIST],
            }
        )
    missing_architectures = [x for x in REQUIRED_ARCHITECTURES if x not in covered_architectures]
    if missing_architectures:
        gaps.append({"type": "architecture_coverage_shortfall", "missing": missing_architectures})
    required_missing = [name for name in required_core_names if name not in required_selected]
    if required_missing:
        gaps.append({
            "type": "verified_required_core_shortfall",
            "required": len(required_core_names),
            "selected": len(required_selected),
            "missing": required_missing,
        })

    return {
        "policy_version": "1.0",
        "generated_for_year": current_year,
        "classic_cutoff_year": classic_cutoff,
        "recent_sota_window": [recent_cutoff, current_year],
        "quotas": {
            "classic_baseline_min": classic_min,
            "recent_sota_candidate_min": recent_sota_min,
            "verified_required_core_names": required_core_names,
            "max_models": max_models,
        },
        "sota_semantics": "论文中的 SOTA 仅作为候选证据；必须经统一 benchmark 后才能确认",
        "role_counts": role_counts,
        "covered_architectures": covered_architectures,
        "gaps": gaps,
        "selected_models": selected,
    }
