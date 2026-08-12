#!/usr/bin/env python
"""Five-batch LLM nomination followed by evidence-gated online verification.

The nomination stage intentionally has no browsing/search capability.  Its
output is quarantined and cannot enter the evidence pool.  The verification
stage uses Crossref/OpenAlex plus publisher-curated local JIF metadata.  Only
paper-backed AMP prediction models are integrated into the formal evidence
pool and recommendation memory.
"""

import argparse
import csv
import datetime as dt
import math
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import deep_research_literature_agent as lit


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NOMINATIONS_JSON = DATA / "llm_top_journal_model_nominations.json"
NOMINATIONS_MD = DATA / "llm_top_journal_model_nominations.md"
VERIFICATION_JSON = DATA / "llm_top_journal_model_verification.json"
VERIFICATION_MD = DATA / "llm_top_journal_model_verification.md"

BATCH_FOCUS = [
    "优先提名发表在本领域高水平期刊、论文影响力高、适合作为 AMP benchmark 主表的模型",
    "补充经典基线、传统机器学习与早期深度学习模型，不能与前一批重复",
    "补充 2022 年以来的近期 SOTA 候选，强调独立测试、外部验证和可复现代码",
    "补齐 CNN/RNN/GNN/Transformer/蛋白语言模型/集成学习等架构多样性",
    "补充前四批遗漏但具有公开论文或代码证据的长尾模型，保持任务与年代多样性",
]

NOMINATION_FIELDS = [
    "model_name", "aliases", "paper_title", "publication_year", "authors",
    "source_journal", "claimed_journal_impact_factor", "claimed_jif_data_year",
    "claimed_citation_count", "source_doi", "source_pmid", "paper_url",
    "code_repository_url", "web_server_url", "dataset_source_or_link",
    "task_type", "model_architecture", "input_representation", "training_data",
    "reported_metrics", "benchmark_role", "why_recommended", "uncertainties",
]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _model_key(value: Any) -> str:
    return lit.normalize_key(lit.canonicalize_model_name(value))


def _title_similarity(a: Any, b: Any) -> float:
    aa = lit.normalize_key(a)
    bb = lit.normalize_key(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def _nomination_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("models") or payload.get("candidates") or payload.get("recommendations") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _normalize_nomination(row: Dict[str, Any], batch_no: int) -> Dict[str, Any]:
    out = {field: row.get(field) for field in NOMINATION_FIELDS}
    out["model_name"] = _clean_text(out.get("model_name") or row.get("canonical_name") or row.get("name"))
    out["aliases"] = lit.ensure_list(out.get("aliases"))
    out["authors"] = lit.ensure_list(out.get("authors"))
    out["reported_metrics"] = out.get("reported_metrics") if isinstance(out.get("reported_metrics"), dict) else {}
    out["uncertainties"] = lit.ensure_list(out.get("uncertainties"))
    out["nomination_batch"] = batch_no
    out["nomination_status"] = "unverified_llm_nomination"
    out["eligible_for_evidence_pool"] = False
    out["generated_at"] = lit.now_str()
    return out


def _render_nomination_md(doc: Dict[str, Any]) -> str:
    lines = [
        "# LLM AMP Model Nominations (Unverified)", "",
        "> 这些条目只是无联网搜索能力的大模型提名，不是证据，禁止直接进入正式 benchmark。", "",
        "- Created: " + str(doc.get("created_at") or ""),
        "- Provider: " + str(doc.get("provider") or ""),
        "- Target: " + str(doc.get("target") or ""),
        "- Unique nominations: " + str(len(lit.ensure_list(doc.get("models")))), "",
    ]
    for batch in lit.ensure_list(doc.get("batches")):
        if not isinstance(batch, dict):
            continue
        lines += ["## Batch " + str(batch.get("batch_no")), "", str(batch.get("focus") or ""), ""]
        for row in lit.ensure_list(batch.get("models")):
            if not isinstance(row, dict):
                continue
            lines.append(
                "- **{name}** | paper={paper} | journal={journal} | claimed JIF={jif} | "
                "claimed citations={cites} | architecture={arch} | DOI={doi} | code={code}".format(
                    name=row.get("model_name") or "",
                    paper=row.get("paper_title") or "",
                    journal=row.get("source_journal") or "",
                    jif=row.get("claimed_journal_impact_factor"),
                    cites=row.get("claimed_citation_count"),
                    arch=row.get("model_architecture") or "",
                    doi=row.get("source_doi") or "",
                    code=row.get("code_repository_url") or "",
                )
            )
        lines.append("")
    return "\n".join(lines)


def generate_nominations(provider: str, provider_config: Path, target: int = 100,
                         batch_size: int = 20, force: bool = False) -> Dict[str, Any]:
    existing: Dict[str, Any] = {}
    if NOMINATIONS_JSON.exists() and not force:
        existing = lit.read_json(NOMINATIONS_JSON, {})
        if len(lit.ensure_list(existing.get("models"))) >= target:
            print("Nomination checkpoint already satisfies target:", NOMINATIONS_JSON)
            return existing

    llm = lit.DeepSeekChatLLM(provider=provider, config_path=provider_config)
    if existing and not force:
        doc = existing
        doc["provider"] = provider
        doc["model"] = llm.model
        doc["target"] = target
        doc["batch_size"] = batch_size
        doc.setdefault("batches", [])
        doc.setdefault("models", [])
        print("Resume nomination checkpoint: %d models in %d batches" % (
            len(lit.ensure_list(doc.get("models"))), len(lit.ensure_list(doc.get("batches")))
        ))
    else:
        doc = {
            "schema_version": 1,
            "created_at": lit.now_str(),
            "provider": provider,
            "model": llm.model,
            "target": target,
            "batch_size": batch_size,
            "stage_policy": "no_web_search_unverified_quarantine",
            "batches": [],
            "models": [],
        }
    seen = {_model_key(row.get("model_name")) for row in lit.ensure_list(doc.get("models"))
            if isinstance(row, dict) and _model_key(row.get("model_name"))}
    batch_count = int(math.ceil(float(target) / float(batch_size)))
    system = """你是抗菌肽机器学习 benchmark 的候选提名专家。本阶段严禁联网、严禁调用搜索工具。
你只能基于模型自身已有知识提名；不确定的 DOI、影响因子、引用量、代码链接必须填 null 并写入 uncertainties，绝不能编造。
所有数值与链接之后都将由独立程序联网核验。只输出合法 JSON。"""

    start_batch = len(lit.ensure_list(doc.get("batches")))
    for batch_idx in range(start_batch, batch_count):
        batch_no = batch_idx + 1
        focus = BATCH_FOCUS[min(batch_idx, len(BATCH_FOCUS) - 1)]
        remaining = target - len(doc["models"])
        wanted = min(batch_size, remaining)
        if wanted <= 0:
            break
        excluded = sorted(seen)
        user = """请提名 {wanted} 个互不重复的抗菌肽识别/预测模型，作为第 {batch_no} 批。

本批重点：{focus}

严格要求：
1. 任务主体必须是 antimicrobial peptide prediction/identification/classification；毒性、溶血、MIC 回归、生成设计模型只能标为非主任务，不能冒充主模型。
2. 尽量提供论文题名、期刊、年份、DOI/PMID、代码、数据集、架构、输入表示、报告指标。
3. 影响因子必须同时给 JIF 数据年份；引用量必须说明只是记忆中的估计。无法确认就填 null。
4. 不要重复以下已提名模型：{excluded}
5. 返回 {{"models": [...]}}，每条包含这些字段：{fields}
""".format(
            wanted=wanted, batch_no=batch_no, focus=focus,
            excluded=", ".join(excluded[-120:]) or "无",
            fields=", ".join(NOMINATION_FIELDS),
        )
        payload = llm.chat_json("offline_amp_model_nomination_batch_%d" % batch_no, system, user)
        accepted: List[Dict[str, Any]] = []
        for raw in _nomination_rows(payload):
            row = _normalize_nomination(raw, batch_no)
            key = _model_key(row.get("model_name"))
            if not key or key in seen:
                continue
            seen.add(key)
            accepted.append(row)
            doc["models"].append(row)
            if len(doc["models"]) >= target:
                break
        doc["batches"].append({"batch_no": batch_no, "focus": focus, "models": accepted})
        lit.write_json(NOMINATIONS_JSON, doc)
        NOMINATIONS_MD.write_text(_render_nomination_md(doc), encoding="utf-8")
        print("Batch %d: accepted %d, cumulative %d/%d" % (batch_no, len(accepted), len(doc["models"]), target))

    doc["completed_at"] = lit.now_str()
    doc["shortfall"] = max(0, target - len(doc["models"]))
    lit.write_json(NOMINATIONS_JSON, doc)
    NOMINATIONS_MD.write_text(_render_nomination_md(doc), encoding="utf-8")
    return doc


def _load_jif_metadata() -> Dict[str, Dict[str, Any]]:
    path = DATA / "journal_impact_factors.csv"
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = lit.normalize_key(row.get("verification_status"))
            if not status.startswith("verified"):
                continue
            keys = [row.get("journal")]
            keys.extend(re.split(r"[;|]", str(row.get("aliases") or "")))
            for name in keys:
                key = lit.normalize_key(name)
                if key:
                    out[key] = dict(row)
    return out


def _merge_source_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = lit.normalize_key(row.get("doi")) or lit.normalize_key(row.get("title"))
        if not key:
            continue
        current = merged.get(key, {})
        combined = lit.merge_candidate(current, row)
        combined["sources"] = sorted(set(lit.ensure_list(current.get("sources")) + lit.ensure_list(row.get("sources"))))
        combined["cited_by_count"] = max(
            lit._safe_float(current.get("cited_by_count"), 0.0),
            lit._safe_float(row.get("cited_by_count") or row.get("citation_count"), 0.0),
        )
        merged[key] = combined
    return list(merged.values())


def _paper_match_score(candidate: Dict[str, Any], paper: Dict[str, Any]) -> float:
    claimed_doi = lit.normalize_key(candidate.get("source_doi"))
    found_doi = lit.normalize_key(paper.get("doi"))
    if claimed_doi and found_doi and claimed_doi == found_doi:
        return 1.0
    score = _title_similarity(candidate.get("paper_title"), paper.get("title"))
    text = lit.normalize_key(" ".join([
        _clean_text(paper.get("title")), _clean_text(paper.get("abstract")),
    ]))
    model_key = _model_key(candidate.get("model_name"))
    if model_key and len(model_key) >= 4 and model_key in text:
        score += 0.12
    if "antimicrobial peptide" in text or "antibacterial peptide" in text:
        score += 0.08
    return min(score, 1.0)


def _architecture_evidence(claim: Any, paper: Dict[str, Any]) -> Tuple[str, List[str]]:
    claim_key = lit.normalize_key(claim)
    text = lit.normalize_key(" ".join([_clean_text(paper.get("title")), _clean_text(paper.get("abstract"))]))
    vocabulary = [
        "support vector machine", "svm", "random forest", "xgboost", "cnn",
        "convolutional", "rnn", "lstm", "gru", "attention", "transformer",
        "bert", "protein language model", "graph neural network", "gnn", "gat",
        "ensemble", "logistic regression", "gradient boosting", "deep learning",
        "machine learning",
    ]
    claimed_tokens = [token for token in vocabulary if token in claim_key]
    hits = sorted(set(token for token in claimed_tokens if token in text))
    if hits:
        return "verified_from_title_or_abstract", hits
    return "unverified_architecture_claim", []


def _verify_repository(url: Any) -> Tuple[Optional[str], str]:
    value = _clean_text(url)
    match = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)", value, flags=re.I)
    if not match:
        return None, "missing_or_not_github"
    owner, repo = match.group(1), match.group(2).replace(".git", "")
    try:
        data = lit.HTTP.get_json("https://api.github.com/repos/%s/%s" % (owner, repo))
        if isinstance(data, dict) and data.get("html_url") and not data.get("disabled"):
            return data.get("html_url"), "verified_github_api"
    except Exception as exc:
        return None, "github_unverified:%s" % type(exc).__name__
    return None, "github_unverified"


def verify_one(candidate: Dict[str, Any], crossref: Any, openalex: Any,
               jif_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    queries = []
    for value in [candidate.get("source_doi"), candidate.get("paper_title"),
                  "%s antimicrobial peptide" % _clean_text(candidate.get("model_name"))]:
        value = _clean_text(value)
        if value and value not in queries:
            queries.append(value)
    found: List[Dict[str, Any]] = []
    errors: List[str] = []
    for query in queries[:3]:
        try:
            found.extend(crossref.search(query, rows=5))
        except Exception as exc:
            errors.append("crossref:%s" % type(exc).__name__)
        try:
            found.extend(openalex.search(query, rows=5))
        except Exception as exc:
            errors.append("openalex:%s" % type(exc).__name__)
        claimed_doi = lit.normalize_key(candidate.get("source_doi"))
        if claimed_doi and any(lit.normalize_key(row.get("doi")) == claimed_doi for row in found if isinstance(row, dict)):
            # An exact DOI match is decisive; avoid two more broad network
            # searches, which also reduce reliability under rate limits.
            break
    papers = _merge_source_records(found)
    ranked = sorted((( _paper_match_score(candidate, paper), paper) for paper in papers),
                    key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] < 0.62:
        return {
            "model_name": candidate.get("model_name"),
            "nomination_batch": candidate.get("nomination_batch"),
            "verification_status": "rejected_no_matching_paper",
            "eligible_for_evidence_pool": False,
            "best_match_score": ranked[0][0] if ranked else 0.0,
            "verification_errors": sorted(set(errors)),
            "verified_at": lit.now_str(),
            "nomination": candidate,
        }

    match_score, paper = ranked[0]
    relevant = lit.looks_relevant(paper)
    paper_text = lit.normalize_key(" ".join([_clean_text(paper.get("title")), _clean_text(paper.get("abstract"))]))
    model_key = _model_key(candidate.get("model_name"))
    explicit_model = bool(model_key and len(model_key) >= 4 and model_key in paper_text)
    exact_claimed_paper = _title_similarity(candidate.get("paper_title"), paper.get("title")) >= 0.85
    model_supported = relevant and (explicit_model or exact_claimed_paper)
    journal = paper.get("journal") or paper.get("venue") or candidate.get("source_journal")
    jif = jif_meta.get(lit.normalize_key(journal), {})
    arch_status, arch_hits = _architecture_evidence(candidate.get("model_architecture"), paper)
    verified_repo, repo_status = _verify_repository(candidate.get("code_repository_url"))
    citations = int(lit._safe_float(paper.get("cited_by_count") or paper.get("citation_count"), 0.0))
    jif_numeric = lit._safe_float(jif.get("impact_factor"), 0.0)
    jif_value = jif_numeric if jif_numeric > 0 else None
    status = "verified" if model_supported else "rejected_not_verified_amp_model"
    sources = list(paper.get("urls") or [])
    if paper.get("doi"):
        sources.append("https://doi.org/%s" % paper.get("doi"))
    if paper.get("openalex_id"):
        sources.append(paper.get("openalex_id"))
    if jif.get("source_url"):
        sources.append(jif.get("source_url"))
    if verified_repo:
        sources.append(verified_repo)
    return {
        "model_name": candidate.get("model_name"),
        "canonical_name": lit.canonicalize_model_name(candidate.get("model_name")),
        "nomination_batch": candidate.get("nomination_batch"),
        "verification_status": status,
        "eligible_for_evidence_pool": status == "verified",
        "best_match_score": round(match_score, 4),
        "paper_title": paper.get("title"),
        "publication_year": paper.get("year"),
        "source_journal": journal,
        "source_doi": paper.get("doi"),
        "source_pmid": paper.get("pmid"),
        "source_openalex_id": paper.get("openalex_id"),
        "citation_count": citations,
        "citation_count_status": "verified_from_openalex" if paper.get("openalex_id") else "missing_openalex_citation",
        "citation_evidence_source": "OpenAlex" if paper.get("openalex_id") else None,
        "citation_snapshot_date": dt.date.today().isoformat(),
        "journal_impact_factor": jif_value,
        "journal_impact_factor_status": jif.get("verification_status") or "missing_curated_jif_mapping",
        "jif_data_year": jif.get("jif_data_year"),
        "jif_source_url": jif.get("source_url"),
        "top_journal_level": bool(jif_numeric >= 5.0),
        "model_architecture": candidate.get("model_architecture") if arch_status.startswith("verified") else None,
        "architecture_claim": candidate.get("model_architecture"),
        "architecture_verification_status": arch_status,
        "architecture_evidence_terms": arch_hits,
        "input_representation": candidate.get("input_representation"),
        "task_type": candidate.get("task_type") or "AMP prediction/classification",
        "code_repository_url": verified_repo,
        "code_repository_verification_status": repo_status,
        "dataset_source_or_link": candidate.get("dataset_source_or_link"),
        "web_server_url": candidate.get("web_server_url"),
        "reported_metrics": candidate.get("reported_metrics") or {},
        "online_verification_sources": sorted(set(str(x) for x in sources if x)),
        "verification_errors": sorted(set(errors)),
        "verified_at": lit.now_str(),
        "paper_record": paper,
        "nomination": candidate,
    }


def _verified_score(row: Dict[str, Any]) -> float:
    citations = max(0.0, lit._safe_float(row.get("citation_count"), 0.0))
    jif = max(0.0, lit._safe_float(row.get("journal_impact_factor"), 0.0))
    score = min(math.log10(citations + 1.0), 3.0)
    score += min(math.log10(jif + 1.0) * 1.5, 2.0)
    score += 1.0 if row.get("code_repository_url") else 0.0
    score += 0.5 if str(row.get("architecture_verification_status") or "").startswith("verified") else 0.0
    year = int(lit._safe_float(row.get("publication_year"), 0.0))
    if year >= dt.date.today().year - 2:
        score += 0.5
    return round(score, 4)


def _post_audit_result(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reject real database/supplement records that are not runnable models."""
    out = dict(row)
    if out.get("verification_status") != "verified":
        return out
    if (lit._safe_float(out.get("journal_impact_factor"), 0.0) <= 0
            and not lit.normalize_key(out.get("journal_impact_factor_status")).startswith("verified")):
        out["journal_impact_factor"] = None
    title = lit.normalize_key(out.get("paper_title"))
    doi = lit.normalize_key(out.get("source_doi"))
    model = _model_key(out.get("model_name"))
    paper = out.get("paper_record") if isinstance(out.get("paper_record"), dict) else {}
    paper_text = lit.normalize_key(" ".join([
        _clean_text(paper.get("title") or out.get("paper_title")),
        _clean_text(paper.get("abstract")),
    ]))
    database_names = {
        "apd3", "apd3 prediction", "dbaasp", "dbaasp prediction", "dramp",
        "dramp 2 0 prediction", "dbamp", "dbamp 2 0 prediction", "bactibase prediction",
        "camp r2", "camp r3", "campr4 prediction",
    }
    database_phrases = [
        "antimicrobial peptide database", "database of antimicrobial", "data repository of antimicrobial",
        "updated data repository", "database as a tool", "database as a resource",
    ]
    supplementary = bool(
        re.search(r"(?:^|[ ./_-])(supp|supplement|mm\d+|s\d{3})(?:$|[ ./_-])", title + " " + doi)
        or title.endswith(" docx") or title.endswith(" pdf")
    )
    if model in database_names or any(phrase in title for phrase in database_phrases):
        out["verification_status"] = "rejected_database_or_platform_not_model"
    elif supplementary:
        out["verification_status"] = "rejected_supplement_not_primary_paper"
    elif not model or len(model) < 4 or model not in paper_text:
        out["verification_status"] = "rejected_model_name_not_supported_by_paper"
    elif "antimicrobial region" in title and "peptide" not in title:
        out["verification_status"] = "rejected_related_non_main_amp_task"
    if out.get("verification_status") != "verified":
        out["eligible_for_evidence_pool"] = False
        out["verified_recommendation_score"] = 0.0
        out["post_audit_at"] = lit.now_str()
    else:
        out["post_audit_status"] = "passed_primary_model_paper_gate"
    return out


def _refresh_openalex_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a verified DOI in OpenAlex, with Crossref citation fallback."""
    out = dict(row)
    if out.get("verification_status") != "verified" or not out.get("source_doi"):
        return out
    if out.get("source_openalex_id") and out.get("citation_count_status") == "verified_from_openalex":
        return out
    if (out.get("citation_count_status") == "verified_from_crossref_is_referenced_by_count"
            and out.get("citation_snapshot_date") == dt.date.today().isoformat()):
        return out
    doi_url = "https://doi.org/%s" % _clean_text(out.get("source_doi"))
    endpoint = "https://api.openalex.org/works/%s" % urllib.parse.quote(doi_url, safe="")
    prior_errors = lit.ensure_list(out.get("verification_errors"))
    data = None
    if not any(str(err).startswith("openalex_doi_refresh:") for err in prior_errors):
        try:
            data = lit.HTTP.get_json(endpoint)
        except Exception as exc:
            errors = lit.ensure_list(out.get("verification_errors"))
            errors.append("openalex_doi_refresh:%s" % type(exc).__name__)
            out["verification_errors"] = sorted(set(errors))
    if isinstance(data, dict) and _title_similarity(out.get("paper_title"), data.get("title")) >= 0.72:
        out["source_openalex_id"] = data.get("id")
        out["citation_count"] = int(lit._safe_float(data.get("cited_by_count"), 0.0))
        out["citation_count_status"] = "verified_from_openalex"
        out["citation_evidence_source"] = "OpenAlex DOI lookup"
        out["citation_snapshot_date"] = dt.date.today().isoformat()
        sources = lit.ensure_list(out.get("online_verification_sources"))
        if data.get("id"):
            sources.append(data.get("id"))
        out["online_verification_sources"] = sorted(set(str(x) for x in sources if x))
        paper = dict(out.get("paper_record") or {})
        paper["openalex_id"] = data.get("id")
        paper["cited_by_count"] = out["citation_count"]
        paper["sources"] = sorted(set(lit.ensure_list(paper.get("sources")) + ["openalex_doi_lookup"]))
        out["paper_record"] = paper
        out["verified_recommendation_score"] = _verified_score(out)
        return out

    crossref_endpoint = "https://api.crossref.org/works/%s" % urllib.parse.quote(
        _clean_text(out.get("source_doi")), safe=""
    )
    try:
        crossref_data = lit.HTTP.get_json(crossref_endpoint)
        message = crossref_data.get("message") if isinstance(crossref_data, dict) else {}
    except Exception as exc:
        errors = lit.ensure_list(out.get("verification_errors"))
        errors.append("crossref_doi_refresh:%s" % type(exc).__name__)
        out["verification_errors"] = sorted(set(errors))
        return out
    crossref_title = ((message.get("title") or [""])[0]) if isinstance(message, dict) else ""
    if _title_similarity(out.get("paper_title"), crossref_title) < 0.72:
        return out
    out["citation_count"] = int(lit._safe_float(message.get("is-referenced-by-count"), 0.0))
    out["citation_count_status"] = "verified_from_crossref_is_referenced_by_count"
    out["citation_evidence_source"] = "Crossref DOI lookup"
    out["citation_snapshot_date"] = dt.date.today().isoformat()
    sources = lit.ensure_list(out.get("online_verification_sources"))
    sources.append(crossref_endpoint)
    out["online_verification_sources"] = sorted(set(str(x) for x in sources if x))
    paper = dict(out.get("paper_record") or {})
    paper["citation_count"] = out["citation_count"]
    paper["sources"] = sorted(set(lit.ensure_list(paper.get("sources")) + ["crossref_doi_lookup"]))
    out["paper_record"] = paper
    out["verified_recommendation_score"] = _verified_score(out)
    return out


def _render_verification_md(doc: Dict[str, Any]) -> str:
    rows = lit.ensure_list(doc.get("results"))
    verified = [row for row in rows if isinstance(row, dict) and row.get("verification_status") == "verified"]
    rejected = [row for row in rows if isinstance(row, dict) and row.get("verification_status") != "verified"]
    lines = [
        "# Online Verification of LLM AMP Model Nominations", "",
        "- Verified at: " + str(doc.get("updated_at") or ""),
        "- Checked: " + str(len(rows)),
        "- Verified AMP models: " + str(len(verified)),
        "- Rejected/unresolved: " + str(len(rejected)), "",
        "## Verified recommendation ranking", "",
        "| Rank | Model | Paper | Year | Journal | JIF (year) | Citations | Architecture | Code | Score |",
        "|---:|---|---|---:|---|---:|---:|---|---|---:|",
    ]
    for rank, row in enumerate(sorted(verified, key=_verified_score, reverse=True), 1):
        lines.append("| {rank} | {model} | {paper} | {year} | {journal} | {jif} ({jyear}) | {cites} | {arch} | {code} | {score} |".format(
            rank=rank, model=row.get("model_name") or "", paper=row.get("paper_title") or "",
            year=row.get("publication_year") or "", journal=row.get("source_journal") or "",
            jif=row.get("journal_impact_factor") or "", jyear=row.get("jif_data_year") or "",
            cites=row.get("citation_count") or 0, arch=row.get("model_architecture") or "unverified",
            code=row.get("code_repository_url") or "unverified", score=_verified_score(row),
        ))
    lines += ["", "## Rejected or unresolved nominations", ""]
    for row in rejected:
        lines.append("- {name}: {status} (best_match_score={score})".format(
            name=row.get("model_name") or "", status=row.get("verification_status") or "",
            score=row.get("best_match_score") or 0,
        ))
    return "\n".join(lines)


def verify_nominations(force: bool = False, limit: int = 0, workers: int = 4) -> Dict[str, Any]:
    nominations = lit.read_json(NOMINATIONS_JSON, {})
    candidates = [row for row in lit.ensure_list(nominations.get("models")) if isinstance(row, dict)]
    if not candidates:
        raise FileNotFoundError("No nominations found; run --stage nominate first")
    existing = {} if force else lit.read_json(VERIFICATION_JSON, {})
    candidate_keys = {_model_key(row.get("model_name")) for row in candidates if _model_key(row.get("model_name"))}
    previous_results = [] if force else [row for row in lit.ensure_list(existing.get("results")) if isinstance(row, dict)]
    superseded_results = [row for row in previous_results if _model_key(row.get("model_name")) not in candidate_keys]
    results = [row for row in previous_results if _model_key(row.get("model_name")) in candidate_keys]
    completed = {_model_key(row.get("model_name")) for row in results}
    crossref = lit.CrossrefClient()
    openalex = lit.OpenAlexClient()
    jif_meta = _load_jif_metadata()
    pending: List[Dict[str, Any]] = []
    for candidate in candidates:
        key = _model_key(candidate.get("model_name"))
        if not key or key in completed:
            continue
        if limit and len(pending) >= limit:
            break
        pending.append(candidate)

    def run_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return verify_one(candidate, crossref, openalex, jif_meta)
        except Exception as exc:
            return {
                "model_name": candidate.get("model_name"),
                "nomination_batch": candidate.get("nomination_batch"),
                "verification_status": "verification_error",
                "eligible_for_evidence_pool": False,
                "verification_errors": ["%s:%s" % (type(exc).__name__, str(exc)[:300])],
                "verified_at": lit.now_str(),
                "nomination": candidate,
            }

    max_workers = max(1, min(int(workers or 1), 8))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for index, candidate in enumerate(pending, 1):
            print("Queue verify %d/%d: %s" % (len(results) + index, len(candidates), candidate.get("model_name")))
            future_map[executor.submit(run_candidate, candidate)] = candidate
        for future in as_completed(future_map):
            candidate = future_map[future]
            result = future.result()
            result["verified_recommendation_score"] = _verified_score(result) if result.get("verification_status") == "verified" else 0.0
            results.append(result)
            completed.add(_model_key(candidate.get("model_name")))
            doc = {
                "schema_version": 1, "created_at": existing.get("created_at") or lit.now_str(),
                "updated_at": lit.now_str(), "nomination_file": str(NOMINATIONS_JSON.relative_to(ROOT)),
                "results": results,
            }
            lit.write_json(VERIFICATION_JSON, doc)
            VERIFICATION_MD.write_text(_render_verification_md(doc), encoding="utf-8")
            print("Completed %d/%d: %s -> %s" % (
                len(results), len(candidates), result.get("model_name"), result.get("verification_status")
            ))
    results = [_post_audit_result(row) for row in results]
    results = [_refresh_openalex_metrics(row) for row in results]
    doc = {
        "schema_version": 1, "created_at": existing.get("created_at") or lit.now_str(),
        "updated_at": lit.now_str(), "nomination_file": str(NOMINATIONS_JSON.relative_to(ROOT)),
        "checked": len(results),
        "verified_count": sum(row.get("verification_status") == "verified" for row in results),
        "superseded_result_count": len(superseded_results),
        "superseded_results": superseded_results,
        "results": results,
    }
    lit.write_json(VERIFICATION_JSON, doc)
    VERIFICATION_MD.write_text(_render_verification_md(doc), encoding="utf-8")
    return doc


def _model_from_verified(row: Dict[str, Any]) -> Dict[str, Any]:
    nomination = row.get("nomination") if isinstance(row.get("nomination"), dict) else {}
    return {
        "model_name": row.get("model_name"),
        "canonical_name": row.get("canonical_name") or row.get("model_name"),
        "publication_year": row.get("publication_year"),
        "paper_title": row.get("paper_title"),
        "task_type": row.get("task_type") or "AMP prediction/classification",
        "method_family": row.get("model_architecture") or "architecture_not_verified",
        "architecture_or_algorithm": row.get("model_architecture"),
        "input_representation": row.get("input_representation"),
        "source_journal": row.get("source_journal"),
        "journal_impact_factor": row.get("journal_impact_factor"),
        "journal_impact_factor_status": row.get("journal_impact_factor_status"),
        "citation_count": row.get("citation_count"),
        "citation_count_status": row.get("citation_count_status"),
        "citation_evidence_source": row.get("citation_evidence_source"),
        "source_doi": row.get("source_doi"),
        "source_pmid": row.get("source_pmid"),
        "code_repository_url": row.get("code_repository_url"),
        "web_server_url": row.get("web_server_url"),
        "dataset_source_or_link": row.get("dataset_source_or_link"),
        "reported_metrics": row.get("reported_metrics") or {},
        "candidate_reason": nomination.get("why_recommended") or "LLM nomination independently verified online",
        "evidence_level": "crossref_openalex_verified_llm_nomination",
        "confidence": min(1.0, max(0.0, lit._safe_float(row.get("best_match_score"), 0.0))),
        "online_verification_sources": row.get("online_verification_sources") or [],
        "verification_status": "verified_before_evidence_pool_integration",
        "provenance": "llm_nomination_then_crossref_openalex_verification",
    }


def integrate_verified() -> Dict[str, Any]:
    verification = lit.read_json(VERIFICATION_JSON, {})
    verified = [row for row in lit.ensure_list(verification.get("results"))
                if isinstance(row, dict) and row.get("verification_status") == "verified"
                and row.get("eligible_for_evidence_pool") is True]
    if not verified:
        raise RuntimeError("No verified models are eligible for integration")
    models = [_model_from_verified(row) for row in verified]

    pool = lit.read_json(lit.EVIDENCE_POOL_JSON, {})
    papers = lit.ensure_list(pool.get("papers"))
    for row in verified:
        paper = dict(row.get("paper_record") or {})
        if not paper:
            continue
        paper["sources"] = sorted(set(lit.ensure_list(paper.get("sources")) + ["llm_nomination_online_verified"]))
        paper["candidate_key"] = lit.candidate_key(paper)
        paper["verification_status"] = "verified_before_evidence_pool_integration"
        papers.append(paper)
    papers = lit.dedupe_candidates([row for row in papers if isinstance(row, dict)])
    batches = lit.ensure_list(pool.get("evidence_batches"))
    batches = [row for row in batches if not (isinstance(row, dict) and row.get("_stage") == "llm_nomination_online_verified")]
    batches.append({
        "_stage": "llm_nomination_online_verified",
        "_batch_no": "llm_verified_%s" % dt.date.today().isoformat(),
        "models": models,
        "papers": [row.get("paper_record") for row in verified if row.get("paper_record")],
        "important_evidence": ["Only Crossref/OpenAlex-verified AMP model nominations were integrated."],
        "uncertainties": ["JIF is available only for journals present in the curated local official-source mapping."],
    })
    lit.save_evidence_pool(
        batches, papers,
        lit.ensure_list(pool.get("external_repositories")),
        lit.ensure_list(pool.get("external_datasets")),
    )

    memory = lit.read_json(lit.MEMORY_JSON, {})
    memory["all_candidate_models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("all_candidate_models")) + models)
    memory["models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("models")) + models)
    code_models = [row for row in models if row.get("code_repository_url")]
    memory["benchmark_ready_models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("benchmark_ready_models")) + code_models)
    memory["papers"] = lit.dedupe_candidates(lit.ensure_list(memory.get("papers")) + papers)
    memory.setdefault("runs", []).append({
        "time": lit.now_str(), "mode": "llm_nomination_online_verified_integration",
        "verified_models_added": len(models), "verified_code_models_added": len(code_models),
        "verification_file": str(VERIFICATION_JSON.relative_to(ROOT)),
    })
    lit.write_json(lit.MEMORY_JSON, memory)
    refreshed = lit.refresh_memory_views_only()
    return {
        "verified_models_integrated": len(models),
        "verified_code_models_integrated": len(code_models),
        "final_deployment_models": len(lit.ensure_list(refreshed.get("final_deployment_models"))),
        "evidence_pool": str(lit.EVIDENCE_POOL_JSON.relative_to(ROOT)),
        "memory": str(lit.MEMORY_JSON.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LLM 5x20 AMP nomination, online verification, evidence integration")
    parser.add_argument("--stage", choices=["nominate", "verify", "integrate", "all"], default="nominate")
    parser.add_argument("--provider", default="dashscope")
    parser.add_argument("--provider-config", default="llm_providers.json")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--verify-limit", type=int, default=0, help="0 verifies all remaining nominations")
    parser.add_argument("--verify-workers", type=int, default=4, help="Concurrent candidate verifications (1-8)")
    parser.add_argument("--force-nominations", action="store_true")
    parser.add_argument("--force-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.provider_config)
    if args.stage in {"nominate", "all"}:
        doc = generate_nominations(args.provider, config, args.target, args.batch_size, args.force_nominations)
        print("Nominations:", len(lit.ensure_list(doc.get("models"))), NOMINATIONS_JSON)
    if args.stage in {"verify", "all"}:
        doc = verify_nominations(force=args.force_verification, limit=args.verify_limit, workers=args.verify_workers)
        print("Verified:", doc.get("verified_count"), "/", doc.get("checked"), VERIFICATION_JSON)
    if args.stage in {"integrate", "all"}:
        print("Integration:", lit.json_dumps(integrate_verified(), 2))


if __name__ == "__main__":
    main()
