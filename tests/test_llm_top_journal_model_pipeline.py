import unittest

import llm_top_journal_model_pipeline as pipeline


class LlmTopJournalPipelineTests(unittest.TestCase):
    def test_title_similarity_normalizes_punctuation(self):
        self.assertGreater(
            pipeline._title_similarity("AMP-BERT: Prediction of AMPs", "AMP BERT prediction of AMPs"),
            0.95,
        )

    def test_local_jif_aliases_are_available(self):
        mapping = pipeline._load_jif_metadata()
        row = mapping[pipeline.lit.normalize_key("Bioinformatics (Oxford, England)")]
        self.assertEqual(float(row["impact_factor"]), 5.5)

    def test_unverified_nomination_is_quarantined(self):
        row = pipeline._normalize_nomination({"model_name": "ExampleAMP"}, 2)
        self.assertEqual(row["nomination_status"], "unverified_llm_nomination")
        self.assertFalse(row["eligible_for_evidence_pool"])

    def test_verified_score_prefers_real_evidence(self):
        weak = {"citation_count": 0, "journal_impact_factor": 0}
        strong = {
            "citation_count": 100,
            "journal_impact_factor": 7.3,
            "code_repository_url": "https://github.com/example/model",
            "architecture_verification_status": "verified_from_title_or_abstract",
        }
        self.assertGreater(pipeline._verified_score(strong), pipeline._verified_score(weak))

    def test_model_keys_deduplicate_punctuation_variants(self):
        self.assertEqual(pipeline._model_key("AMP-BERT"), pipeline._model_key("AMP BERT"))

    def test_post_audit_rejects_database_as_model(self):
        row = pipeline._post_audit_result({
            "model_name": "APD3 Prediction",
            "paper_title": "APD3: the antimicrobial peptide database as a tool for research and education",
            "source_doi": "10.1093/nar/gkv1278",
            "verification_status": "verified",
            "eligible_for_evidence_pool": True,
            "paper_record": {"title": "APD3: the antimicrobial peptide database as a tool for research and education"},
        })
        self.assertEqual(row["verification_status"], "rejected_database_or_platform_not_model")
        self.assertFalse(row["eligible_for_evidence_pool"])

    def test_post_audit_requires_model_name_in_paper(self):
        row = pipeline._post_audit_result({
            "model_name": "AMP-Context",
            "paper_title": "DeepALM: A Context-Aware Deep Learning Framework for Antimicrobial Peptide Prediction",
            "verification_status": "verified",
            "eligible_for_evidence_pool": True,
            "paper_record": {"title": "DeepALM: A Context-Aware Deep Learning Framework for Antimicrobial Peptide Prediction"},
        })
        self.assertEqual(row["verification_status"], "rejected_model_name_not_supported_by_paper")


if __name__ == "__main__":
    unittest.main()
