import unittest
import xml.etree.ElementTree as ET

import deep_research_literature_agent as literature
import scientific_model_evidence as scientific


class PubMedPrimaryIdentifierTests(unittest.TestCase):
    def test_reference_ids_do_not_overwrite_primary_article_ids(self):
        xml = """
        <PubmedArticle>
          <MedlineCitation><PMID>41776033</PMID><Article>
            <ArticleTitle>HMD-AMP paper</ArticleTitle>
            <Journal><Title>Nature Biomedical Engineering</Title><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
            <Abstract><AbstractText>Here we introduce HMD-AMP.</AbstractText></Abstract>
          </Article></MedlineCitation>
          <PubmedData>
            <ArticleIdList>
              <ArticleId IdType="pubmed">41776033</ArticleId>
              <ArticleId IdType="doi">10.1038/s41551-026-01630-w</ArticleId>
              <ArticleId IdType="pii">10.1038/s41551-026-01630-w</ArticleId>
            </ArticleIdList>
            <ReferenceList><Reference><ArticleIdList>
              <ArticleId IdType="pubmed">36370099</ArticleId>
              <ArticleId IdType="doi">10.1093/nar/gkac1040</ArticleId>
              <ArticleId IdType="pmc">PMC9825490</ArticleId>
            </ArticleIdList></Reference></ReferenceList>
          </PubmedData>
        </PubmedArticle>
        """
        row = literature.PubMedClient().parse_article(ET.fromstring(xml))
        self.assertEqual(row["pmid"], "41776033")
        self.assertEqual(row["doi"], "10.1038/s41551-026-01630-w")
        self.assertIsNone(row["pmcid"])


class RequiredScientificEvidenceTests(unittest.TestCase):
    def test_required_seed_file_contains_three_distinct_benchmark_units(self):
        rows = scientific.load_seeds()
        self.assertEqual({row["model_name"] for row in rows}, {"C_AMPs-predict", "AMPSorter", "HMD-AMP"})
        ampsorter = next(row for row in rows if row["model_name"] == "AMPSorter")
        self.assertEqual(ampsorter["parent_model"], "ProteoGPT")

    def test_publisher_gate_requires_model_repository_and_dataset(self):
        seed = scientific.load_seeds()[2]
        html = """
        <html>HMD-AMP
        10.1038/s41551-026-01630-w
        https://github.com/ml4bio/HMD-AMP
        https://doi.org/10.5281/zenodo.15622525
        </html>
        """
        self.assertTrue(all(scientific.publisher_checks(seed, html).values()))
        self.assertFalse(scientific.publisher_checks(seed, "HMD-AMP only")["official_repository_on_publisher_page"])

    def test_aliases_separate_parent_proteogpt_from_classifier(self):
        self.assertEqual(literature.canonicalize_model_name("AMPSorter"), "AMPSorter")
        self.assertEqual(literature.canonicalize_model_name("ProteoGPT"), "ProteoGPT")
        self.assertEqual(literature.canonicalize_model_name("c_AMPs-prediction"), "C_AMPs-predict")

    def test_fuzzy_bibliographic_neighbour_cannot_supply_primary_ids_or_citations(self):
        seed = scientific.load_seeds()[0]
        result = {
            "verification_status": "verified",
            "paper_title": seed["paper_title"],
            # The top-level DOI may already have been repaired while nested
            # Crossref identifiers still point at the fuzzy neighbour.
            "source_doi": "10.1038/s41587-022-01226-0",
            "source_pmid": "41164228",
            "citation_count": 6,
            "paper_record": {
                "doi": "10.1038/s41587-022-01226-0",
                "pmid": "41164228",
                "citation_count": 6,
                "source_ids": {"crossref": "10.1038/s41587-022-01230-4"},
                "urls": ["https://doi.org/10.1038/s41587-022-01230-4"],
            },
            "online_verification_sources": ["https://doi.org/10.1038/s41587-022-01230-4"],
        }
        checks = {
            "primary_doi_on_publisher_page": True,
            "model_identity_on_publisher_page": True,
            "official_repository_on_publisher_page": True,
            "official_dataset_on_publisher_page": True,
        }
        fixed = scientific.finalize_evidence_gate(seed, result, checks)
        self.assertEqual(fixed["source_doi"], "10.1038/s41587-022-01226-0")
        self.assertEqual(fixed["source_pmid"], "35241840")
        self.assertIsNone(fixed["citation_count"])
        self.assertEqual(fixed["paper_record"]["citation_count"], 0)
        self.assertNotIn("01230-4", " ".join(fixed["online_verification_sources"]))


if __name__ == "__main__":
    unittest.main()
