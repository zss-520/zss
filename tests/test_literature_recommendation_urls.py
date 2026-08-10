from pathlib import Path

from deep_research_literature_agent import (
    ARCHITECTURE_CATEGORY_DEFS,
    _attach_article_impact,
    _build_paper_impact_index,
    _paper_key_values,
    _select_representatives,
    _strict_main_deployment_candidate,
    classify_architecture_item,
    classify_representation_item,
    dedupe_models_by_name,
    has_code_repository_url,
    is_missing_value,
    merge_candidate,
    normalize_code_repository_url,
    read_jsonl,
)


def test_normalized_not_reported_sentinel_is_missing():
    assert is_missing_value("not_reported_in_available_evidence")
    assert is_missing_value("not reported in available evidence")


def test_real_repository_replaces_missing_sentinel_during_merge():
    merged = merge_candidate(
        {"model_name": "ExampleAMP", "code_repository_url": "not_reported_in_available_evidence"},
        {"model_name": "ExampleAMP", "code_repository_url": "https://github.com/example/amp"},
    )
    assert merged["code_repository_url"] == "https://github.com/example/amp"
    assert has_code_repository_url(merged)


def test_repository_url_drops_trailing_sentence_punctuation():
    assert normalize_code_repository_url("https://github.com/example/amp.") == "https://github.com/example/amp"
    assert normalize_code_repository_url("https://github.com/example/amp.git") == "https://github.com/example/amp.git"


def test_representative_selection_excludes_models_without_repository_url():
    spec = ARCHITECTURE_CATEGORY_DEFS[0]
    rows = [
        {
            "model_name": "NoCodeAMP",
            "method_family": "random forest machine learning",
            "source_doi": "10.1000/no-code",
            "journal_impact_factor": 100,
            "code_repository_url": "not_reported_in_available_evidence",
        },
        {
            "model_name": "CodeBackedAMP",
            "method_family": "random forest machine learning",
            "source_doi": "10.1000/with-code",
            "journal_impact_factor": 1,
            "code_repository_url": "https://github.com/example/code-backed-amp",
        },
    ]
    selected = _select_representatives(rows, spec)
    assert [row["model_name"] for row in selected] == ["CodeBackedAMP"]


def test_read_jsonl_supports_concatenated_pretty_json_documents(tmpdir):
    path = Path(str(tmpdir)) / "legacy.jsonl"
    path.write_text('{\n  "pmid": "1"\n}\n{\n  "pmid": "2"\n}\n', encoding="utf-8")
    assert [row["pmid"] for row in read_jsonl(path)] == ["1", "2"]


def test_paper_keys_expand_list_identifiers():
    keys = _paper_key_values({"source_pmid": ["1", "2"], "source_doi": ["10.1/a", "10.1/b"]})
    assert keys == ["pmid:1", "pmid:2", "doi:10 1 a", "doi:10 1 b"]


def test_rich_local_paper_metadata_replaces_zero_placeholders():
    data = {
        "papers": [{"pmid": "42", "title": "Example AMP", "citation_count": 0}],
        "records": [{
            "pmid": "42",
            "title": "Example AMP",
            "journal": "Example Journal",
            "citation_count": 17,
            "sources": ["semantic_scholar"],
        }],
    }
    journal_map = {"example journal": 4.2}
    index = _build_paper_impact_index(data, journal_map)
    row = _attach_article_impact(
        {"model_name": "ExampleAMP", "source_pmid": "42", "citation_count": 0, "journal_impact_factor": 0},
        index,
        journal_map,
    )
    assert row["source_journal"] == "Example Journal"
    assert row["citation_count"] == 17
    assert row["journal_impact_factor"] == 4.2
    assert row["citation_evidence_source"] == "semantic_scholar"


def test_primary_registry_prevents_secondary_paper_metadata_from_replacing_amp_scanner():
    row = dedupe_models_by_name([{
        "model_name": "AMP Scanner v2",
        "publication_year": 2025,
        "source_doi": ["10.1007/s00248-025-02620-2", "10.1128/spectrum.01504-25"],
        "source_pmid": ["41315055", "40891852"],
        "method_family": "machine learning",
        "representation_category": "traditional_physicochemical_statistical_features",
        "architecture_category": "machine_learning_models",
        "code_repository_url": "https://github.com/dan-veltri/amp-scanner-v2",
    }])[0]
    assert row["publication_year"] == 2018
    assert row["source_doi"] == "10.1093/bioinformatics/bty179"
    assert row["source_pmid"] == "29590297"
    assert row["representation_category"] == "sequence_encoding_representation"
    assert row["architecture_category"] == "cnn_rnn_hybrid_models"
    assert "10.1007/s00248-025-02620-2" in row["secondary_evidence_dois"]


def test_primary_registry_corrects_c_amps_predict_pmid_and_preserves_locked_taxonomy():
    row = dedupe_models_by_name([{
        "model_name": "C_AMPs-predict",
        "source_doi": "10.1038/s41587-022-01226-0",
        "source_pmid": "41164228",
        "publication_year": 2025,
        "code_repository_url": "https://github.com/mayuefine/c_AMPs-prediction",
        "evidence_level": "primary_publisher_verified",
    }])[0]
    assert row["source_pmid"] == "35241840"
    assert row["publication_year"] == 2022
    assert "41164228" in row["secondary_evidence_pmids"]
    assert classify_representation_item(row) == "protein_language_model_representation"
    assert classify_architecture_item(row) == "rnn_lstm_dominant_models"


def test_out_of_scope_and_regression_models_cannot_enter_main_deployment():
    common = {
        "code_repository_url": "https://github.com/example/model",
        "evidence_level": "fulltext",
        "confidence": 1.0,
    }
    allopipe = dedupe_models_by_name([dict(common, model_name="Allopipe")])[0]
    eippred = dedupe_models_by_name([dict(common, model_name="EIPpred")])[0]
    assert allopipe["benchmark_candidate"] is False
    assert eippred["benchmark_candidate"] is False
    assert not _strict_main_deployment_candidate(allopipe)
    assert not _strict_main_deployment_candidate(eippred)


def test_ambiguous_generic_amp_name_cannot_enter_main_deployment():
    row = {
        "model_name": "AMP",
        "task_type": "AMP binary prediction/classification",
        "source_doi": "10.1000/example",
        "code_repository_url": "https://github.com/example/amp",
        "evidence_level": "fulltext",
        "confidence": 1.0,
    }
    assert not _strict_main_deployment_candidate(row)
