# Task: infer one model registry candidate

Candidate evidence:

{{var:candidate_json}}

Repository context:

{{var:repository_context}}

Return a JSON object describing the model name, environment hint, repository URL, local directory, Python-version hint, dependencies, additional setup commands, inference command template, weight evidence, confidence and unresolved risks.

The inference command should use `{fasta_path}` and `{output_dir}` when repository evidence supports a command. If it cannot be reconstructed reliably, leave it empty and record the gap instead of inventing it. Deterministic code will normalize dependencies, reject unsafe commands, fix authoritative paths and validate required fields.
