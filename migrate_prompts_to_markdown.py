"""Compatibility verifier for the completed Markdown prompt migration.

Prompt extraction was completed in 2026. This command now performs a read-only
audit so it cannot overwrite composed role prompts or shared policy fragments.
"""

from __future__ import annotations

from agent_md_loader import AgentMDLoader
import prompts


def main() -> int:
    loader = AgentMDLoader(prompts.PROMPT_DIR)
    failures: list[str] = []
    for constant_name, stem in prompts.PROMPT_FILES.items():
        try:
            expected = loader.load_composed(stem)
        except Exception as exc:
            failures.append(f"{constant_name}: {exc}")
            continue
        actual = getattr(prompts, constant_name, None)
        role_composed = constant_name in {
            "MULTI_AGENT_SCOUT_PROMPT",
            "MULTI_AGENT_METRICS_PROMPT",
            "MULTI_AGENT_CRITIC_PROMPT",
        }
        if (role_composed and not str(actual or "").startswith(expected.strip())) or (not role_composed and actual != expected):
            failures.append(f"{constant_name}: runtime value differs from {stem}.md")
    if failures:
        for failure in failures:
            print("ERROR", failure)
        return 1
    print(f"OK: verified {len(prompts.PROMPT_FILES)} composed runtime prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
