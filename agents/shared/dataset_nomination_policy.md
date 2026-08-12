# Shared policy: dataset nomination, not executable approval

- Agents nominate and discuss dataset candidates; only the deterministic dataset gate may mark a dataset eligible for execution.
- For each candidate, preserve positive/negative-label evidence, source provenance, intended role, length evidence, negative construction, deduplication or homology evidence, model-specific independence limits and required cleaning.
- A positive-only AMP database is not a binary gold-standard test set. An unlabeled sequence is not automatically a verified negative.
- Prefer permanent and auditable sources, but do not invent a direct download URL when only a database or supplementary-material reference is reported.
- Use `accept`, `reject` or `defer` as a scientific recommendation. Mark unmeasured properties as requiring real-sequence audit.
- Do not replace a rejected candidate with a hard-coded or invented dataset.
