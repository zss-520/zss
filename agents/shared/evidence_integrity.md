# Shared policy: evidence integrity and provenance

- Use only evidence present in the supplied context. Never invent a paper, DOI, PMID, URL, repository, dataset, weight file, experiment or result.
- Label statements derived from model knowledge as `llm_prior`; they are hypotheses, not retrieved citations.
- Preserve provenance fields and distinguish metadata, abstract, full text, repository, dataset repository, web-search candidate and local benchmark evidence.
- Web-search and repository-search hits are provisional until their authorship, relation to the paper, downloadable assets and inference entry point are verified.
- When evidence is missing or contradictory, return `defer`, `uncertain` or `not_reported_in_available_evidence` and state the verification task.
- Do not silently delete dissent, failed verification or unresolved evidence.
