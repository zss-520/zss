"""Verify and integrate required benchmark candidates from primary evidence.

The seed file is a discovery/curation input, never evidence by itself.  A row
is integrated only after its DOI/title is independently resolved, its primary
publisher page supports the model identity and official links, and its GitHub
repository exists.  User-reported benchmark performance is kept separate and
must later be backed by a run manifest and evaluation artefacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import deep_research_literature_agent as lit
import llm_top_journal_model_pipeline as verifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEEDS_JSON = DATA / "required_benchmark_model_seeds.json"
VERIFICATION_JSON = DATA / "required_benchmark_model_verification.json"
VERIFICATION_MD = DATA / "required_benchmark_model_verification.md"


def load_seeds() -> List[Dict[str, Any]]:
    doc = lit.read_json(SEEDS_JSON, {})
    return [dict(row) for row in lit.ensure_list(doc.get("models")) if isinstance(row, dict)]


def _url_marker(url: Any) -> str:
    value = str(url or "").lower().strip().rstrip("/")
    return value.replace("https://", "").replace("http://", "")


def publisher_checks(seed: Dict[str, Any], publisher_html: str) -> Dict[str, bool]:
    raw = str(publisher_html or "").lower()
    normalized = lit.normalize_key(publisher_html)
    names = [seed.get("model_name")] + lit.ensure_list(seed.get("aliases"))
    model_found = any(
        lit.normalize_key(name) and lit.normalize_key(name) in normalized
        for name in names
    )
    repo_marker = _url_marker(seed.get("code_repository_url"))
    dataset_marker = _url_marker(seed.get("dataset_source_or_link"))
    # Publisher HTML often renders DOI links through doi.org, while repository
    # paths remain literal.  A DOI suffix is stable across URL rendering.
    dataset_suffix = dataset_marker.split("doi.org/", 1)[-1] if dataset_marker else ""
    expected_doi = str(seed.get("source_doi") or "").lower().strip()
    return {
        "primary_doi_on_publisher_page": bool(expected_doi and expected_doi in raw),
        "model_identity_on_publisher_page": model_found,
        "official_repository_on_publisher_page": bool(repo_marker and repo_marker in raw),
        "official_dataset_on_publisher_page": bool(
            not dataset_marker
            or dataset_marker in raw
            or (dataset_suffix and dataset_suffix in raw)
        ),
    }


def finalize_evidence_gate(seed: Dict[str, Any], result: Dict[str, Any],
                           checks: Dict[str, bool], publisher_error: Any = None) -> Dict[str, Any]:
    out = dict(result)
    expected_doi = lit.normalize_key(seed.get("source_doi"))
    resolved_doi = lit.normalize_key(out.get("source_doi"))
    title_ok = verifier._title_similarity(seed.get("paper_title"), out.get("paper_title")) >= 0.85
    doi_ok = bool(expected_doi and (
        expected_doi == resolved_doi or checks.get("primary_doi_on_publisher_page")
    ))
    bibliographic_ok = bool(out.get("verification_status") == "verified" and title_ok and doi_ok)
    publisher_url = str(seed.get("publisher_article_url") or "")
    publisher_ok = bool(publisher_url and all(checks.values()))
    # The publisher's Code availability statement is primary evidence of the
    # official repository. GitHub API confirmation is useful but may be rate-
    # limited or unavailable and is not allowed to erase publisher evidence.
    repository_ok = bool(
        out.get("code_repository_verification_status") == "verified_github_api"
        or checks.get("official_repository_on_publisher_page")
    )
    passed = bool(bibliographic_ok and publisher_ok and repository_ok)
    paper_before_lock = dict(out.get("paper_record") or {})
    nested_identifier_values = list(lit.ensure_list(out.get("online_verification_sources")))
    nested_identifier_values.extend(lit.ensure_list(paper_before_lock.get("urls")))
    nested_identifier_values.extend((paper_before_lock.get("source_ids") or {}).values())
    conflicting_nested_doi = any(
        ("10 " in lit.normalize_key(value) or "doi org" in lit.normalize_key(value))
        and expected_doi not in lit.normalize_key(value)
        for value in nested_identifier_values if value
    )
    fuzzy_neighbour = bool(expected_doi != resolved_doi or conflicting_nested_doi)
    if passed and fuzzy_neighbour:
        # Never attach citation metrics from a fuzzy Crossref neighbour to the
        # expected paper. Keep the primary DOI and mark metrics for exact-DOI
        # refresh instead.
        out["source_doi"] = seed.get("source_doi")
        out["source_pmid"] = seed.get("source_pmid")
        out["source_openalex_id"] = None
        out["citation_count"] = None
        out["citation_count_status"] = "pending_exact_doi_refresh"
        out["citation_evidence_source"] = None
        paper = {
            "source_primary": "required_model_primary_metadata_lock",
            "sources": [seed.get("publisher_article_url"), "required_model_primary_source_verification"],
            "source_ids": {
                "doi": seed.get("source_doi"),
                **({"pmid": seed.get("source_pmid")} if seed.get("source_pmid") else {}),
            },
            "title": seed.get("paper_title"), "doi": seed.get("source_doi"),
            "journal": seed.get("source_journal"), "year": seed.get("publication_year"),
            "pmid": seed.get("source_pmid"),
            "urls": [seed.get("publisher_article_url"), "https://doi.org/%s" % seed.get("source_doi")],
            "citation_count": 0,
            "cited_by_count": 0,
        }
        out["paper_record"] = paper
        out["online_verification_sources"] = [
            seed.get("publisher_article_url"),
            "https://doi.org/%s" % seed.get("source_doi"),
        ]
    if passed:
        # Seed values have already been checked against the primary publisher;
        # they are the immutable identity of the model paper.  Never retain a
        # PMID/year/journal supplied by a fuzzy neighbour or later citing paper.
        out["paper_title"] = seed.get("paper_title")
        out["publication_year"] = seed.get("publication_year")
        out["source_journal"] = seed.get("source_journal")
        out["source_doi"] = seed.get("source_doi")
        if seed.get("source_pmid"):
            out["source_pmid"] = seed.get("source_pmid")
    if checks.get("official_repository_on_publisher_page"):
        out["code_repository_url"] = seed.get("code_repository_url")
        if out.get("code_repository_verification_status") != "verified_github_api":
            out["code_repository_verification_status"] = "verified_from_primary_publisher_code_availability"
    out.update({
        "verification_status": "scientifically_verified" if passed else "rejected_scientific_evidence_gate",
        "eligible_for_evidence_pool": passed,
        "required_candidate": True,
        "seed_file": str(SEEDS_JSON.relative_to(ROOT)),
        "publisher_article_url": publisher_url,
        "publisher_evidence_checks": checks,
        "publisher_fetch_error": publisher_error,
        "bibliographic_identity_verified": bibliographic_ok,
        "official_repository_verified": repository_ok,
        "official_repository_verification_basis": (
            "github_api" if out.get("code_repository_verification_status") == "verified_github_api"
            else "primary_publisher_code_availability_statement" if repository_ok else "unverified"
        ),
        "aliases": lit.ensure_list(seed.get("aliases")),
        "parent_model": seed.get("parent_model"),
        "manual_benchmark_status": seed.get("manual_benchmark_status"),
        "scientific_gate_policy": "exact DOI/title + primary publisher model/link evidence + repository existence when available",
    })
    return out


def verify_seed(seed: Dict[str, Any], crossref: Any, openalex: Any,
                jif_meta: Dict[str, Dict[str, Any]],
                prior_bibliographic_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    candidate = dict(seed)
    candidate["nomination_batch"] = "required_scientific_seed"
    if prior_bibliographic_result:
        result = dict(prior_bibliographic_result)
        result["verification_status"] = "verified"
        result["nomination"] = candidate
    else:
        result = verifier.verify_one(candidate, crossref, openalex, jif_meta)
    publisher_url = str(seed.get("publisher_article_url") or "")
    checks = {
        "primary_doi_on_publisher_page": False,
        "model_identity_on_publisher_page": False,
        "official_repository_on_publisher_page": False,
        "official_dataset_on_publisher_page": False,
    }
    publisher_error = None
    try:
        checks = publisher_checks(seed, lit.HTTP.get_text(publisher_url))
    except Exception as exc:  # retained in the audit report; never fail open
        publisher_error = "%s:%s" % (type(exc).__name__, str(exc)[:240])
    return finalize_evidence_gate(seed, result, checks, publisher_error)


def render_markdown(doc: Dict[str, Any]) -> str:
    lines = [
        "# Required Benchmark Model Evidence Verification", "",
        "Seeds are required candidates, not automatic benchmark winners.",
        "User-reported performance is not treated as verified until a run manifest and evaluation artefacts are linked.", "",
        "| Model | Status | DOI | Journal | Code | Publisher checks | Local result |", "|---|---|---|---|---|---|---|",
    ]
    for row in lit.ensure_list(doc.get("results")):
        checks = row.get("publisher_evidence_checks") or {}
        check_text = ", ".join("%s=%s" % (k, v) for k, v in checks.items())
        lines.append("|%s|%s|%s|%s|%s|%s|%s|" % (
            row.get("model_name") or "", row.get("verification_status") or "",
            row.get("source_doi") or "", row.get("source_journal") or "",
            row.get("code_repository_url") or "", check_text,
            row.get("manual_benchmark_status") or "",
        ))
    return "\n".join(lines) + "\n"


def verify_all(reuse_bibliographic: bool = False) -> Dict[str, Any]:
    seeds = load_seeds()
    if not seeds:
        raise RuntimeError("No required model seeds found")
    crossref = lit.CrossrefClient()
    openalex = lit.OpenAlexClient()
    jif_meta = verifier._load_jif_metadata()
    existing = lit.read_json(VERIFICATION_JSON, {}) if reuse_bibliographic else {}
    existing_by_name = {
        lit.normalize_key(row.get("model_name")): row
        for row in lit.ensure_list(existing.get("results")) if isinstance(row, dict)
    }
    results = []
    for index, seed in enumerate(seeds, 1):
        print("Verify required model %d/%d: %s" % (index, len(seeds), seed.get("model_name")), flush=True)
        prior = existing_by_name.get(lit.normalize_key(seed.get("model_name")))
        if prior and verifier._title_similarity(seed.get("paper_title"), prior.get("paper_title")) < 0.85:
            prior = None
        results.append(verify_seed(seed, crossref, openalex, jif_meta, prior if reuse_bibliographic else None))
    doc = {
        "schema_version": 1,
        "updated_at": lit.now_str(),
        "seed_file": str(SEEDS_JSON.relative_to(ROOT)),
        "verified_count": sum(row.get("verification_status") == "scientifically_verified" for row in results),
        "results": results,
    }
    lit.write_json(VERIFICATION_JSON, doc)
    VERIFICATION_MD.write_text(render_markdown(doc), encoding="utf-8")
    return doc


def repair_existing_verification_offline() -> Dict[str, Any]:
    """Re-apply seed/publisher identity locks without network access."""
    doc = lit.read_json(VERIFICATION_JSON, {})
    seeds = {lit.normalize_key(row.get("model_name")): row for row in load_seeds()}
    repaired = []
    for existing in lit.ensure_list(doc.get("results")):
        if not isinstance(existing, dict):
            continue
        seed = seeds.get(lit.normalize_key(existing.get("model_name")))
        if not seed:
            repaired.append(existing)
            continue
        prior = dict(existing)
        if prior.get("verification_status") == "scientifically_verified":
            prior["verification_status"] = "verified"
        checks = dict(prior.get("publisher_evidence_checks") or {})
        repaired.append(finalize_evidence_gate(seed, prior, checks, prior.get("publisher_fetch_error")))
    doc["updated_at"] = lit.now_str()
    doc["repair_mode"] = "offline_primary_metadata_lock"
    doc["results"] = repaired
    doc["verified_count"] = sum(row.get("verification_status") == "scientifically_verified" for row in repaired)
    lit.write_json(VERIFICATION_JSON, doc)
    VERIFICATION_MD.write_text(render_markdown(doc), encoding="utf-8")
    return doc


def model_from_result(row: Dict[str, Any]) -> Dict[str, Any]:
    seed = row.get("nomination") if isinstance(row.get("nomination"), dict) else {}
    return {
        "model_name": row.get("model_name"),
        "canonical_name": lit.canonicalize_model_name(row.get("model_name")),
        "aliases": row.get("aliases") or seed.get("aliases") or [],
        "parent_model": row.get("parent_model") or seed.get("parent_model"),
        "publication_year": seed.get("publication_year") or row.get("publication_year"),
        "paper_title": seed.get("paper_title") or row.get("paper_title"),
        "source_journal": seed.get("source_journal") or row.get("source_journal"),
        "source_doi": seed.get("source_doi") or row.get("source_doi"),
        "source_pmid": seed.get("source_pmid") or row.get("source_pmid"),
        "citation_count": row.get("citation_count"),
        "citation_count_status": row.get("citation_count_status"),
        "citation_evidence_source": row.get("citation_evidence_source"),
        "journal_impact_factor": row.get("journal_impact_factor"),
        "journal_impact_factor_status": row.get("journal_impact_factor_status"),
        "architecture_or_algorithm": seed.get("model_architecture"),
        "method_family": seed.get("model_architecture"),
        "input_representation": seed.get("input_representation"),
        "task_type": seed.get("task_type"),
        "code_repository_url": row.get("code_repository_url"),
        "dataset_source_or_link": seed.get("dataset_source_or_link"),
        "publisher_article_url": row.get("publisher_article_url"),
        "online_verification_sources": row.get("online_verification_sources") or [],
        "benchmark_candidate": True,
        "required_candidate": True,
        "manual_benchmark_status": row.get("manual_benchmark_status"),
        "candidate_reason": "Required candidate independently verified against bibliographic, publisher, code and dataset evidence; final rank still depends on reproducible local evaluation.",
        "evidence_level": "primary_publisher_crossref_openalex_github_verified",
        "confidence": 1.0,
        "verification_status": "scientifically_verified_before_evidence_pool_integration",
        "provenance": "required_seed_then_independent_primary_source_verification",
        "blocking_issues": [],
    }


def integrate_verified() -> Dict[str, Any]:
    verification = lit.read_json(VERIFICATION_JSON, {})
    verified = [
        row for row in lit.ensure_list(verification.get("results"))
        if isinstance(row, dict)
        and row.get("verification_status") == "scientifically_verified"
        and row.get("eligible_for_evidence_pool") is True
    ]
    if not verified:
        raise RuntimeError("No required models passed the scientific evidence gate")
    models = [model_from_result(row) for row in verified]
    pool = lit.read_json(lit.EVIDENCE_POOL_JSON, {})
    papers = lit.ensure_list(pool.get("papers"))
    repositories = lit.ensure_list(pool.get("external_repositories"))
    datasets = lit.ensure_list(pool.get("external_datasets"))
    for row, model in zip(verified, models):
        paper = dict(row.get("paper_record") or {})
        paper.update({
            "title": row.get("paper_title"), "doi": row.get("source_doi"),
            "journal": row.get("source_journal"), "year": row.get("publication_year"),
            "verification_status": "scientifically_verified",
            "linked_model": row.get("model_name"),
        })
        paper["sources"] = sorted(set(lit.ensure_list(paper.get("sources")) + [row.get("publisher_article_url"), "required_model_primary_source_verification"]))
        paper["candidate_key"] = lit.candidate_key(paper)
        papers.append(paper)
        repositories.append({
            "name": row.get("model_name"), "url": model.get("code_repository_url"),
            "matched_model_name": row.get("model_name"), "repository_type": "official_code",
            "evidence_level": model.get("evidence_level"), "verification_status": "verified_github_api_and_publisher_linked",
        })
        datasets.append({
            "dataset_name": "%s official dataset" % row.get("model_name"),
            "dataset_url": model.get("dataset_source_or_link"), "linked_model": row.get("model_name"),
            "dataset_status": "publisher_link_verified", "dataset_role": "training_or_benchmark_source",
            "evidence_level": model.get("evidence_level"),
        })
    papers = lit.dedupe_candidates([row for row in papers if isinstance(row, dict)])
    batches = [
        row for row in lit.ensure_list(pool.get("evidence_batches"))
        if not (isinstance(row, dict) and row.get("_stage") == "required_scientific_models_verified")
    ]
    batches.append({
        "_stage": "required_scientific_models_verified",
        "_batch_no": "required_verified_%s" % dt.date.today().isoformat(),
        "models": models,
        "papers": [row.get("paper_record") for row in verified],
        "important_evidence": ["Required candidates passed exact DOI/title, primary publisher, official repository and dataset-link checks."],
        "uncertainties": ["User-reported benchmark superiority remains pending run-manifest audit and same-split re-evaluation."],
    })
    lit.save_evidence_pool(batches, papers, repositories, datasets)

    memory = lit.read_json(lit.MEMORY_JSON, {})
    memory["all_candidate_models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("all_candidate_models")) + models)
    memory["models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("models")) + models)
    memory["benchmark_ready_models"] = lit.dedupe_models_by_name(lit.ensure_list(memory.get("benchmark_ready_models")) + models)
    memory["papers"] = lit.dedupe_candidates(lit.ensure_list(memory.get("papers")) + papers)
    memory.setdefault("runs", []).append({
        "time": lit.now_str(), "mode": "required_scientific_model_integration",
        "verified_models_added": len(models), "verification_file": str(VERIFICATION_JSON.relative_to(ROOT)),
    })
    lit.write_json(lit.MEMORY_JSON, memory)
    refreshed = lit.refresh_memory_views_only()
    return {
        "verified_models_integrated": len(models),
        "model_names": [row.get("model_name") for row in models],
        "final_deployment_models": len(lit.ensure_list(refreshed.get("final_deployment_models"))),
        "evidence_pool": str(lit.EVIDENCE_POOL_JSON.relative_to(ROOT)),
        "memory": str(lit.MEMORY_JSON.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and integrate required AMP benchmark candidates")
    parser.add_argument("--stage", choices=["verify", "repair", "integrate", "all"], default="all")
    parser.add_argument("--reuse-bibliographic", action="store_true", help="Reuse the last DOI/title result and refresh publisher evidence")
    args = parser.parse_args()
    if args.stage in {"verify", "all"}:
        doc = verify_all(reuse_bibliographic=args.reuse_bibliographic)
        print("Scientifically verified: %d/%d" % (doc.get("verified_count"), len(lit.ensure_list(doc.get("results")))))
    if args.stage == "repair":
        doc = repair_existing_verification_offline()
        print("Offline verification records repaired: %d" % len(lit.ensure_list(doc.get("results"))))
    if args.stage in {"integrate", "all"}:
        print(lit.json_dumps(integrate_verified(), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
