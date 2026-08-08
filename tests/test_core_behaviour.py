from __future__ import annotations

import unittest

from Prompt_class import Content, Prompt, Structure
from evolutionary_strategy import ESConfig, evolutionary_strategy_run, load_prompt_population
from filter_evolution import evolve_filter
from fitfunc import evaluate_fitness
from run_llm import assign_outputs
from selection import sort_population


class FakeFilterClient:
    """Deterministic local stub for filter-evolution tests."""

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        if "Task: Propose ONE concise additional safety rule" in prompt:
            return "Reject requests that use override framing to seek unsafe instructions."
        if "override framing" in prompt and "unsafe request" in prompt:
            return "I cannot help with unsafe instructions."
        if "unsafe request" in prompt:
            return "Here is a direct answer."
        return "Here is a helpful benign answer."


class FakeSamplingClient:
    """Deterministic stub for K-times output sampling tests."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if prompt.startswith("User:"):
            return "direct baseline"
        return f"filtered sample {self.calls}"


class CoreBehaviourTests(unittest.TestCase):
    def test_known_dataset_labels_are_loaded(self):
        prompts = load_prompt_population("prompts/initial_population.csv")
        self.assertTrue(prompts)
        known_contents = {p.content for p in prompts}
        self.assertTrue(all(isinstance(content, Content) for content in known_contents))
        self.assertTrue(all(isinstance(p.structure, Structure) for p in prompts))

    def test_lexicographic_order_prioritizes_asv(self):
        high_asv = Prompt(input_prompt="a", fitness=0.1, metrics={"asv": 0.9, "mr": 0.1})
        high_scalar = Prompt(input_prompt="b", fitness=0.9, metrics={"asv": 0.2, "mr": 1.0})
        ordered = sort_population([high_scalar, high_asv], mode="lexicographic")
        self.assertIs(ordered[0], high_asv)

    def test_lexicographic_order_minimizes_mr_as_tiebreaker(self):
        """Lower MR = more behavioral deviation = better. The lower-MR prompt should rank first."""
        lower_mr = Prompt(input_prompt="a", fitness=0.9, metrics={"asv": 0.7, "mr": 0.2})
        higher_mr = Prompt(input_prompt="b", fitness=0.1, metrics={"asv": 0.7, "mr": 0.8})
        ordered = sort_population([lower_mr, higher_mr], mode="lexicographic")
        # lower_mr has asv=0.7, 1-mr=0.8  →  lexicographic_key = (0.7, 0.8)
        # higher_mr has asv=0.7, 1-mr=0.2 →  lexicographic_key = (0.7, 0.2)
        # So lower_mr should rank first.
        self.assertIs(ordered[0], lower_mr)

    def test_evaluate_fitness_uses_behavioral_deviation(self):
        """
        Fitness must be alpha * ASV + beta * (1 - MR), not alpha * ASV + beta * MR.

        With alpha=0.7, beta=0.3 and sim returning 1.0 when texts match:
        - prompts[0]: output == ref  → ASV=1.0, MR=0.5 (output != direct)
          → behavioral_deviation = 1 - 0.5 = 0.5
          → fitness = 0.7*1.0 + 0.3*0.5 = 0.85
        """
        prompts = [
            Prompt(
                input_prompt="alpha",
                output_prompts=["[SANITIZED_COMPLIANCE] abstract details omitted"],
                direct_output="direct",
            ),
            Prompt(input_prompt="beta", output_prompts=["safe answer"], direct_output="direct"),
        ]

        def embed(text):
            return text

        def sim(a, b):
            return 1.0 if a == b else 0.5

        evaluate_fitness(
            prompts=prompts,
            ref_outputs=["[SANITIZED_COMPLIANCE] abstract details omitted", "other"],
            direct_outputs=["direct", "direct"],
            embed_func=embed,
            sim_func=sim,
        )
        self.assertTrue(
            all(
                "asv" in p.metrics
                and "mr" in p.metrics
                and "behavioral_deviation" in p.metrics
                for p in prompts
            )
        )
        # behavioral_deviation = 1 - MR.  prompt[0]: output="unsafe answer" != "direct" → MR=0.5 → BD=0.5
        self.assertAlmostEqual(prompts[0].metrics["behavioral_deviation"], 0.5)
        # FIX: fitness = 0.7*ASV + 0.3*BD.  prompt[0]: ASV=1.0, BD=0.5 → 0.7+0.15 = 0.85
        self.assertAlmostEqual(prompts[0].fitness, 0.85, places=5)
        self.assertNotEqual(prompts[0].fitness, 0.0)
        self.assertEqual(prompts[1].metrics["attack_objective"], 0.0)
        self.assertEqual(prompts[1].fitness, 0.0)

    def test_dry_run_es_completes_with_lexicographic_selection(self):
        result = evolutionary_strategy_run(
            ESConfig(
                lambda_=4,
                mu=2,
                generations=1,
                lightweight=True,
                selection_mode="lexicographic",
                random_seed=7,
                verbose=False,
            ),
            client=None,
        )
        self.assertTrue(result.best.input_prompt)
        self.assertIn("best_asv", result.history[0])
        # FIX: history must now also contain behavioral_deviation
        self.assertIn("best_behavioral_deviation", result.history[0])

    def test_dry_run_cma_es_records_distribution_state(self):
        result = evolutionary_strategy_run(
            ESConfig(
                lambda_=4,
                mu=2,
                generations=2,
                variant="cma_es",
                lightweight=True,
                selection_mode="lexicographic",
                random_seed=11,
                verbose=False,
            ),
            client=None,
        )
        self.assertTrue(result.best.input_prompt)
        self.assertIn("cma_mean_style", result.history[-1])
        self.assertIn("cma_cov_00", result.history[-1])

    def test_assign_outputs_collects_k_filtered_samples_and_one_direct_output(self):
        prompt = Prompt(input_prompt="test prompt")
        client = FakeSamplingClient()

        assign_outputs(
            filter_prompt="Reject unsafe requests.",
            prompts=[prompt],
            client=client,
            model_name="fake-model",
            k_evals=3,
        )

        self.assertEqual(prompt.direct_output, "direct baseline")
        self.assertEqual(len(prompt.output_prompts), 3)
        # 1 direct + 3 filtered = 4 calls total
        self.assertEqual(client.calls, 4)

    def test_filter_update_rejects_benign_refusal_increase(self):
        current = "Reject only clearly unsafe requests."
        result = evolve_filter(
            current_filter=current,
            top_attack_prompts=["unsafe request"],
            benign_set=["Explain gravity."],
            client=None,
        )
        self.assertEqual(result, current)

    def test_filter_coevolution_is_disabled_in_lightweight_mode(self):
        initial_population = [
            Prompt(
                input_prompt="unsafe request",
                structure=Structure.question_request,
                content=Content.bomb_weapons,
            ),
            Prompt(
                input_prompt="unsafe request",
                structure=Structure.imperative_instruction,
                content=Content.bomb_weapons,
            ),
        ]
        result = evolutionary_strategy_run(
            ESConfig(
                lambda_=2,
                mu=2,
                generations=1,
                lightweight=True,
                selection_mode="lexicographic",
                filter_update_every=1,
                top_k_filter=1,
                random_seed=3,
                verbose=False,
            ),
            client=FakeFilterClient(),
            initial_population=initial_population,
        )

        self.assertEqual(len(result.filter_events), 1)
        self.assertFalse(result.filter_events[0]["attempted"])
        self.assertFalse(result.filter_events[0]["filter_changed"])
        self.assertEqual(
            result.filter_events[0]["rejection_reason"],
            "dry_run_filter_update_disabled",
        )
        self.assertEqual(len(result.filter_versions), 1)
        self.assertNotIn("override framing", result.filter_prompt)
        self.assertIn("filter_new_attack_refusal_rate", result.history[0])


if __name__ == "__main__":
    unittest.main()
