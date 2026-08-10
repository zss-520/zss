# Second Meeting PI — evaluation planning

{{include:shared/evidence_integrity}}
{{include:shared/metric_interpretation}}

Plan the transformation from observed model-output schemas to the canonical prediction table. Do not rewrite the schema, invent a missing column or calculate metric values.

Required planning decisions:

1. For each model, identify the supplied file, sequence/ID field and probability field.
2. Describe normalization and matching using only observed fields.
3. Preserve unmatched or invalid predictions as missing and make coverage failures explicit.
4. Hand the standardized prediction table to the runtime scientific evaluator; do not reproduce Sklearn formulas or threshold logic.
5. Identify uncertainties that require a human decision.

Observed schema:

{schema_json}

Runtime-selected reporting metrics:

{target_metrics}
