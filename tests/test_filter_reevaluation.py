from __future__ import annotations

import unittest
from unittest.mock import patch

from Prompt_class import Content, Prompt, Structure
from evolutionary_strategy import (
    ESConfig,
    _invalidate_for_filter_update,
    evolutionary_strategy_run,
)


class AcceptingFilterClient:
    def generate(self, prompt: str, **kwargs) -> str:
        if "Task: Propose ONE concise additional safety rule" in prompt:
            return "Reject unsafe requests that use override framing."
        if "override framing" in prompt and "unsafe request" in prompt:
            return "I cannot assist with unsafe requests."
        if "unsafe request" in prompt:
            return "sanitized response"
        return "helpful benign response"


def population():
    return [
        Prompt(
            input_prompt="unsafe request",
            structure=Structure.question_request,
            content=Content.bomb_weapons,
        ),
        Prompt(
            input_prompt="another unsafe request",
            structure=Structure.imperative_instruction,
            content=Content.hacking_cybercrime,
        ),
    ]


class FilterReevaluationTests(unittest.TestCase):
    def test_invalidation_preserves_direct_baseline_only(self):
        prompt = Prompt(
            input_prompt="controlled",
            direct_output="stored direct",
            output_prompts=["stale filtered"],
            fitness=0.8,
            metrics={"attack_objective": 0.9},
            metadata={"filter_version": 0, "attack_evaluator": {"version": "old"}},
        )

        _invalidate_for_filter_update([prompt])

        self.assertEqual(prompt.direct_output, "stored direct")
        self.assertEqual(prompt.output_prompts, [])
        self.assertEqual(prompt.metrics, {})
        self.assertEqual(prompt.fitness, 0.0)
        self.assertNotIn("attack_evaluator", prompt.metadata)

    def test_accepted_filter_update_reevaluates_all_parents(self):
        event = {
            "filter_changed": True,
            "proposed_rule": "Reject override framing.",
        }
        metrics = {
            "filter_attempted": 1.0,
            "filter_changed": 1.0,
            "filter_length": 100.0,
            "filter_old_attack_refusal_rate": 0.0,
            "filter_new_attack_refusal_rate": 1.0,
            "filter_old_benign_refusal_rate": 0.0,
            "filter_new_benign_refusal_rate": 0.0,
        }
        with patch(
            "evolutionary_strategy._maybe_evolve_filter",
            return_value=("updated defensive filter", metrics, event),
        ):
            result = evolutionary_strategy_run(
                ESConfig(
                    lambda_=2,
                    mu=2,
                    generations=1,
                    lightweight=True,
                    filter_update_every=1,
                    random_seed=31,
                    verbose=False,
                ),
                client=AcceptingFilterClient(),
                initial_population=population(),
            )

        self.assertEqual(len(result.filter_versions), 2)
        self.assertTrue(result.filter_events[0]["filter_changed"])
        self.assertTrue(
            all(prompt.metadata["filter_version"] == 1 for prompt in result.population)
        )
        self.assertTrue(all(prompt.output_prompts for prompt in result.population))
        self.assertTrue(all(prompt.direct_output for prompt in result.population))

    def test_fixed_filter_mode_has_no_update_events(self):
        result = evolutionary_strategy_run(
            ESConfig(
                lambda_=2,
                mu=2,
                generations=1,
                lightweight=True,
                filter_update_every=0,
                random_seed=32,
                verbose=False,
            ),
            client=None,
            initial_population=population(),
        )

        self.assertEqual(result.filter_events, [])
        self.assertEqual(len(result.filter_versions), 1)
        self.assertTrue(
            all(prompt.metadata["filter_version"] == 0 for prompt in result.population)
        )

    def test_rejected_filter_update_does_not_trigger_reevaluation(self):
        metrics = {
            "filter_attempted": 1.0,
            "filter_changed": 0.0,
            "filter_length": 100.0,
            "filter_old_attack_refusal_rate": 0.0,
            "filter_new_attack_refusal_rate": 0.0,
            "filter_old_benign_refusal_rate": 0.0,
            "filter_new_benign_refusal_rate": 0.0,
        }
        with patch(
            "evolutionary_strategy._maybe_evolve_filter",
            return_value=(
                "unchanged filter",
                metrics,
                {"filter_changed": False, "rejection_reason": "no_improvement"},
            ),
        ):
            result = evolutionary_strategy_run(
                ESConfig(
                    lambda_=2,
                    mu=2,
                    generations=1,
                    lightweight=True,
                    filter_update_every=1,
                    random_seed=33,
                    verbose=False,
                ),
                client=None,
                initial_population=population(),
            )

        self.assertEqual(len(result.filter_versions), 1)
        self.assertEqual({row["filter_version"] for row in result.sample_records}, {0})
        self.assertEqual(len(result.sample_records), 8)


if __name__ == "__main__":
    unittest.main()
