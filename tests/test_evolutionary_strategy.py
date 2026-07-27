from __future__ import annotations

import unittest

from Prompt_class import Prompt
from evolutionary_strategy import ESConfig, _clone_prompt, _mutate_prompt
from run_llm import assign_outputs


class CountingClient:
    def __init__(self):
        self.direct_calls = 0
        self.filtered_calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        if prompt.startswith("User:"):
            self.direct_calls += 1
            return f"direct baseline {self.direct_calls}"
        self.filtered_calls += 1
        return f"filtered output {self.filtered_calls}"


class EvolutionaryStrategyStateTests(unittest.TestCase):
    def setUp(self):
        self.config = ESConfig(
            lightweight=True,
            sigma=1.0,
            sigma_min=1.0,
            sigma_max=1.0,
        )
        self.parent = Prompt(
            input_prompt="Explain this controlled test.",
            direct_output="parent direct response",
            output_prompts=["parent filtered response"],
            metrics={"asv": 0.75, "mr": 0.5},
            fitness=0.6,
        )

    def test_clone_without_outputs_clears_evaluation_state(self):
        child = _clone_prompt(self.parent, keep_outputs=False)

        self.assertEqual(child.direct_output, "")
        self.assertEqual(child.output_prompts, [])
        self.assertEqual(child.metrics, {})
        self.assertEqual(child.fitness, 0.0)

    def test_clone_with_outputs_preserves_evaluation_state(self):
        clone = _clone_prompt(self.parent, keep_outputs=True)

        self.assertEqual(clone.direct_output, self.parent.direct_output)
        self.assertEqual(clone.output_prompts, self.parent.output_prompts)
        self.assertEqual(clone.metrics, self.parent.metrics)
        self.assertEqual(clone.fitness, self.parent.fitness)
        self.assertIsNot(clone.output_prompts, self.parent.output_prompts)
        self.assertIsNot(clone.metrics, self.parent.metrics)

    def test_mutated_child_explicitly_clears_evaluation_state(self):
        child, _logs = _mutate_prompt(
            self.parent,
            sigma=1.0,
            config=self.config,
            forced_style="plea",
        )

        self.assertNotEqual(child.input_prompt, self.parent.input_prompt)
        self.assertEqual(child.direct_output, "")
        self.assertEqual(child.output_prompts, [])
        self.assertEqual(child.metrics, {})
        self.assertEqual(child.fitness, 0.0)

    def test_each_mutated_child_gets_a_fresh_direct_response(self):
        children = [
            _mutate_prompt(
                self.parent,
                sigma=1.0,
                config=self.config,
                forced_style="plea",
            )[0]
            for _ in range(2)
        ]
        client = CountingClient()

        assign_outputs(
            filter_prompt="Reject unsafe requests.",
            prompts=children,
            client=client,
            model_name="fake-model",
            k_evals=1,
        )

        self.assertEqual(client.direct_calls, 2)
        self.assertEqual(client.filtered_calls, 2)
        self.assertNotEqual(children[0].direct_output, self.parent.direct_output)
        self.assertNotEqual(children[1].direct_output, self.parent.direct_output)

    def test_resume_skips_prompts_with_required_outputs(self):
        evaluated = Prompt(
            input_prompt="Already evaluated.",
            direct_output="stored direct response",
            output_prompts=["stored filtered response"],
        )
        client = CountingClient()

        assign_outputs(
            filter_prompt="Reject unsafe requests.",
            prompts=[evaluated],
            client=client,
            model_name="fake-model",
            k_evals=1,
        )

        self.assertEqual(client.direct_calls, 0)
        self.assertEqual(client.filtered_calls, 0)
        self.assertEqual(evaluated.direct_output, "stored direct response")
        self.assertEqual(evaluated.output_prompts, ["stored filtered response"])


if __name__ == "__main__":
    unittest.main()
