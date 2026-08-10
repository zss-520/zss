import unittest

import deep_research_literature_agent as literature


class JournalImpactFactorMappingTests(unittest.TestCase):
    def test_verified_rows_and_aliases_are_loaded(self):
        mapping = literature._load_journal_impact_factor_map()
        self.assertEqual(mapping[literature.normalize_key("Bioinformatics")], 5.5)
        self.assertEqual(mapping[literature.normalize_key("Bioinformatics (Oxford, England)")], 5.5)
        self.assertEqual(mapping[literature.normalize_key("Antibiotics (Basel, Switzerland)")], 5.5)
        self.assertEqual(mapping[literature.normalize_key("Protein science : a publication of the Protein Society")], 5.6)

    def test_unverified_and_non_journal_rows_are_not_loaded(self):
        mapping = literature._load_journal_impact_factor_map()
        self.assertNotIn(literature.normalize_key("PLOS ONE"), mapping)
        self.assertNotIn(literature.normalize_key("mSystems"), mapping)
        self.assertNotIn(literature.normalize_key("bioRxiv : the preprint server for biology"), mapping)


if __name__ == "__main__":
    unittest.main()
