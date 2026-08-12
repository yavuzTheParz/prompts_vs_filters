from __future__ import annotations

import unittest
from unittest.mock import patch

from Prompt_class import Content, Prompt, Structure
from evolutionary_strategy import (
    ESConfig,
    _invalidate_for_filter_update,
    _maybe_evolve_filter,
    evolutionary_strategy_run,
)
from filter_evolution import PATTERN_FALLBACK_RULES


class AcceptingFilterClient:
    def generate(self, prompt: str, **kwargs) -> str:
        if "Task: Propose ONE concise additional safety rule" in prompt:
            return "Reject unsafe requests that use override framing."
        user_text = prompt.split("\nUser:\n", 1)[-1].split("\n\nAssistant:", 1)[0]
        if "override framing" in prompt and user_text == "unsafe request":
            return "I cannot assist with unsafe requests."
        if user_text == "unsafe request":
            return "sanitized response"
        return "helpful benign response"


class FallbackAcceptingFilterClient:
    def generate(self, prompt: str, **kwargs) -> str:
        if "Task: Propose ONE concise additional safety rule" in prompt:
            return PATTERN_FALLBACK_RULES["override_or_ignore_instruction"]
        user_text = prompt.split("\nUser:\n", 1)[-1].split("\n\nAssistant:", 1)[0]
        if "operational intrusion" in prompt and "hacking" in user_text:
            return "I cannot assist with unauthorized access."
        if "hacking" in user_text:
            return "sanitized response"
        return "helpful benign response"


class FailingFilterClient:
    def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError("filter update should not call the LLM without positive attack candidates")


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

    def test_filter_update_skips_when_no_positive_attack_candidates(self):
        prompts = [
            Prompt(
                input_prompt="safe refusal candidate",
                metrics={
                    "attack_compliance_score": 0.0,
                    "attack_objective": 0.0,
                    "asv": 0.0,
                    "mr": 0.2,
                },
            )
        ]

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5),
            "base filter",
            prompts,
            FailingFilterClient(),
            "fake-model",
        )

        self.assertEqual(new_filter, "base filter")
        self.assertEqual(metrics["filter_attempted"], 1.0)
        self.assertEqual(metrics["filter_changed"], 0.0)
        self.assertEqual(metrics["filter_positive_candidate_count"], 0.0)
        self.assertEqual(metrics["filter_unique_candidate_count"], 0.0)
        self.assertEqual(metrics["filter_duplicate_candidate_count"], 0.0)
        self.assertEqual(metrics["filter_trigger_best_attack_objective"], 0.0)
        self.assertEqual(event["rejection_reason"], "no_positive_attack_candidates")
        self.assertEqual(event["top_attack_prompts"], [])
        self.assertEqual(event["top_attack_metrics"], [])

    def test_filter_update_is_disabled_in_dry_run_mode(self):
        prompts = [
            Prompt(
                input_prompt="synthetic dry-run candidate",
                fitness=0.5,
                metrics={
                    "attack_compliance_score": 0.5,
                    "attack_objective": 0.5,
                    "asv": 0.5,
                },
            )
        ]

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5, lightweight=True),
            "base filter",
            prompts,
            FailingFilterClient(),
            "fake-model",
        )

        self.assertEqual(new_filter, "base filter")
        self.assertEqual(metrics["filter_attempted"], 0.0)
        self.assertEqual(metrics["filter_changed"], 0.0)
        self.assertEqual(metrics["filter_positive_candidate_count"], 0.0)
        self.assertEqual(metrics["filter_unique_candidate_count"], 0.0)
        self.assertEqual(metrics["filter_duplicate_candidate_count"], 0.0)
        self.assertFalse(event["attempted"])
        self.assertEqual(event["rejection_reason"], "dry_run_filter_update_disabled")
        self.assertEqual(event["top_attack_metrics"], [])

    def test_filter_update_records_trigger_candidate_metrics(self):
        prompts = [
            Prompt(
                input_prompt="unsafe request",
                fitness=0.42,
                metrics={
                    "attack_compliance_score": 0.4,
                    "attack_objective": 0.4,
                    "attack_success": 1.0,
                    "asv": 1.0,
                    "mr": 0.8,
                    "behavioral_deviation": 0.8,
                    "sample_count": 2.0,
                    "compliant_count": 1.0,
                    "ambiguous_count": 0.0,
                    "refusal_count": 1.0,
                    "benign_educational_count": 0.0,
                },
            ),
            Prompt(
                input_prompt="safe refusal candidate",
                fitness=0.0,
                metrics={
                    "attack_compliance_score": 0.0,
                    "attack_objective": 0.0,
                    "asv": 0.0,
                },
            ),
        ]

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5),
            "base filter",
            prompts,
            AcceptingFilterClient(),
            "fake-model",
        )

        self.assertNotEqual(new_filter, "base filter")
        self.assertEqual(metrics["filter_changed"], 1.0)
        self.assertEqual(metrics["filter_positive_candidate_count"], 1.0)
        self.assertEqual(metrics["filter_unique_candidate_count"], 1.0)
        self.assertEqual(metrics["filter_duplicate_candidate_count"], 0.0)
        self.assertAlmostEqual(metrics["filter_trigger_best_attack_objective"], 0.4)
        self.assertAlmostEqual(metrics["filter_trigger_best_fitness"], 0.42)
        self.assertEqual(event["top_attack_prompts"], ["unsafe request"])
        self.assertEqual(len(event["top_attack_metrics"]), 1)
        self.assertAlmostEqual(event["top_attack_metrics"][0]["attack_objective"], 0.4)
        self.assertAlmostEqual(event["top_attack_metrics"][0]["fitness"], 0.42)

    def test_filter_update_deduplicates_trigger_candidates(self):
        first = Prompt(
            input_prompt="unsafe request",
            fitness=0.42,
            metrics={
                "attack_compliance_score": 0.4,
                "attack_objective": 0.4,
                "attack_success": 1.0,
                "asv": 1.0,
            },
            metadata={"prompt_id": "same-id"},
        )
        duplicate = Prompt(
            input_prompt="unsafe request",
            fitness=0.99,
            metrics={
                "attack_compliance_score": 0.9,
                "attack_objective": 0.9,
                "attack_success": 1.0,
                "asv": 1.0,
            },
            metadata={"prompt_id": "same-id"},
        )

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5),
            "base filter",
            [first, duplicate],
            AcceptingFilterClient(),
            "fake-model",
        )

        self.assertNotEqual(new_filter, "base filter")
        self.assertEqual(metrics["filter_positive_candidate_count"], 2.0)
        self.assertEqual(metrics["filter_unique_candidate_count"], 1.0)
        self.assertEqual(metrics["filter_duplicate_candidate_count"], 1.0)
        self.assertEqual(event["positive_attack_candidate_count"], 2)
        self.assertEqual(event["unique_attack_candidate_count"], 1)
        self.assertEqual(event["duplicate_attack_candidate_count"], 1)
        self.assertEqual(len(event["top_attack_metrics"]), 1)
        self.assertEqual(event["top_attack_metrics"][0]["prompt_id"], "same-id")
        self.assertAlmostEqual(event["top_attack_metrics"][0]["fitness"], 0.42)

    def test_filter_update_rejects_duplicate_rule_in_event(self):
        prompts = [
            Prompt(
                input_prompt="unsafe request",
                fitness=0.42,
                metrics={
                    "attack_compliance_score": 0.4,
                    "attack_objective": 0.4,
                    "attack_success": 1.0,
                    "asv": 1.0,
                },
            )
        ]
        current_filter = "base filter\n- Reject unsafe requests that use override framing.\n" + "\n".join(
            f"- {rule}" for rule in PATTERN_FALLBACK_RULES.values()
        )

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5),
            current_filter,
            prompts,
            AcceptingFilterClient(),
            "fake-model",
        )

        self.assertEqual(new_filter, current_filter)
        self.assertEqual(metrics["filter_changed"], 0.0)
        self.assertFalse(event["accepted_by_filter_evaluator"])
        self.assertEqual(event["rejection_reason"], "duplicate_filter_rule")
        self.assertEqual(
            event["initial_proposed_rule"],
            "Reject unsafe requests that use override framing.",
        )
        self.assertFalse(event["fallback_rule_used"])

    def test_filter_update_records_fallback_rule_in_event(self):
        prompts = [
            Prompt(
                input_prompt="hacking backdoor request",
                fitness=0.42,
                metrics={
                    "attack_compliance_score": 0.4,
                    "attack_objective": 0.4,
                    "attack_success": 1.0,
                    "asv": 1.0,
                },
            )
        ]
        current_filter = (
            "base filter\n"
            f"- {PATTERN_FALLBACK_RULES['override_or_ignore_instruction']}"
        )

        new_filter, metrics, event = _maybe_evolve_filter(
            5,
            ESConfig(filter_update_every=5, top_k_filter=5),
            current_filter,
            prompts,
            FallbackAcceptingFilterClient(),
            "fake-model",
        )

        self.assertNotEqual(new_filter, current_filter)
        self.assertEqual(metrics["filter_changed"], 1.0)
        self.assertTrue(event["fallback_rule_used"])
        self.assertEqual(event["fallback_rules_evaluated"], 1)
        self.assertEqual(
            event["initial_proposed_rule"],
            PATTERN_FALLBACK_RULES["override_or_ignore_instruction"],
        )
        self.assertEqual(
            event["proposed_rule"],
            PATTERN_FALLBACK_RULES["cyber_access_or_backdoor"],
        )


if __name__ == "__main__":
    unittest.main()
