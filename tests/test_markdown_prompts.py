from __future__ import annotations

import unittest

import prompts
from agent_md_loader import AgentMDLoader


class MarkdownPromptTests(unittest.TestCase):
    def test_every_runtime_prompt_has_a_markdown_source(self) -> None:
        loader = AgentMDLoader(prompts.PROMPT_DIR)
        self.assertTrue(prompts.PROMPT_DIR.is_absolute())
        self.assertEqual(len(prompts.PROMPT_FILES), 20)

        for constant_name, stem in prompts.PROMPT_FILES.items():
            expected = loader.load_composed(stem)
            actual = getattr(prompts, constant_name)
            if constant_name in {
                "MULTI_AGENT_SCOUT_PROMPT",
                "MULTI_AGENT_METRICS_PROMPT",
                "MULTI_AGENT_CRITIC_PROMPT",
            }:
                self.assertTrue(actual.startswith(expected.strip()))
            else:
                self.assertEqual(actual, expected)

    def test_reusable_meeting_roles_are_composed_in_code(self) -> None:
        self.assertEqual(prompts.MULTI_AGENT_SCOUT_PROMPT.count("# Scout role"), 1)
        self.assertEqual(prompts.MULTI_AGENT_SCOUT_REBUTTAL_PROMPT.count("# Scout role"), 1)
        self.assertIn("initial incremental proposal", prompts.MULTI_AGENT_SCOUT_PROMPT)
        self.assertIn("response to Reviewer", prompts.MULTI_AGENT_SCOUT_REBUTTAL_PROMPT)
        self.assertIn("{{include:", AgentMDLoader(prompts.PROMPT_DIR).load("multi_agent_scout"))
        self.assertNotIn("{{include:", prompts.MULTI_AGENT_SCOUT_PROMPT)

    def test_prompt_builders_still_render_runtime_context(self) -> None:
        model = {
            "model_name": "demo-model",
            "env_name": "demo-env",
            "inference_cmd_template": "python predict.py --input {fasta_path} --out {output_dir}",
        }
        first = prompts.build_first_meeting_agenda([model])
        self.assertIn("demo-model", first)
        self.assertIn("demo-env", first)

        second = prompts.build_second_meeting_agenda([model], "stage-one-context", "id,sequence,label")
        self.assertIn("stage-one-context", second)
        self.assertIn("id,sequence,label", second)

        advisor = prompts.build_amp_research_advisor_prompt("{\"ok\": true}", "AUPRC=0.2")
        self.assertIn("{\"ok\": true}", advisor)
        self.assertIn("AUPRC=0.2", advisor)
        self.assertIn("Top3", advisor)

    def test_loader_rejects_path_escape(self) -> None:
        loader = AgentMDLoader(prompts.PROMPT_DIR)
        with self.assertRaises(ValueError):
            loader.load("../outside")

    def test_loader_renders_explicit_variables_without_touching_json_braces(self) -> None:
        loader = AgentMDLoader(prompts.PROMPT_DIR.parent / "model_onboarding")
        with self.assertRaises(KeyError):
            loader.render("repository_inspector_task", {})


if __name__ == "__main__":
    unittest.main()
