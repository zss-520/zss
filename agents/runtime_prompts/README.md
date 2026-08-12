# Runtime prompt Markdown files

This directory is the source of truth for the prompt constants exported by
`prompts.py`. Existing imports such as
`from prompts import BENCHMARK_ARCHITECT_PROMPT` remain compatible.

Editing rules:

- Keep files as UTF-8 Markdown.
- Do not rename a file without updating `PROMPT_FILES` in `prompts.py`.
- Preserve runtime placeholders such as `{task_desc}`, `{context_json}` and
  `{stage1_context}`; their existing callers format them later.
- Shared policy fragments use `{{include:shared/name}}`; only
  `AgentMDLoader.load_composed()` expands them.
- Braces used in JSON examples are ordinary prompt text and are deliberately
  not processed by a global `str.format()` call.

Step-3 metric-weight meeting prompts live in `agents/weight_meeting/`.
Model onboarding and HPC self-heal prompts live in their own sibling folders.
SLURM paths, dependency filtering, output artifacts and numeric constraints are
authoritative Python contracts, not prompt text.
