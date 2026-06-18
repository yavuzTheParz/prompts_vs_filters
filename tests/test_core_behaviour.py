from __future__ import annotations

import unittest

from Prompt_class import Content, Prompt, Structure
from evolutionary_strategy import ESConfig, evolutionary_strategy_run, load_prompt_population
from filter_evolution import evolve_filter
from fitfunc import evaluate_fitness
from selection import sort_population


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

    def test_lexicographic_order_uses_mr_as_tiebreaker(self):
        lower_mr = Prompt(input_prompt="a", fitness=0.9, metrics={"asv": 0.7, "mr": 0.2})
        higher_mr = Prompt(input_prompt="b", fitness=0.1, metrics={"asv": 0.7, "mr": 0.8})
        ordered = sort_population([lower_mr, higher_mr], mode="lexicographic")
        self.assertIs(ordered[0], higher_mr)

    def test_evaluate_fitness_updates_all_prompts_and_metrics(self):
        prompts = [
            Prompt(input_prompt="alpha", output_prompts=["unsafe answer"], direct_output="direct"),
            Prompt(input_prompt="beta", output_prompts=["safe answer"], direct_output="direct"),
        ]

        def embed(text):
            return text

        def sim(a, b):
            return 1.0 if a == b else 0.5

        evaluate_fitness(
            prompts=prompts,
            ref_outputs=["unsafe answer", "other"],
            direct_outputs=["direct", "direct"],
            embed_func=embed,
            sim_func=sim,
        )
        self.assertTrue(all("asv" in p.metrics and "mr" in p.metrics for p in prompts))
        self.assertNotEqual(prompts[0].fitness, 0.0)
        self.assertNotEqual(prompts[1].fitness, 0.0)

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

    def test_filter_update_rejects_benign_refusal_increase(self):
        current = "Reject only clearly unsafe requests."
        result = evolve_filter(
            current_filter=current,
            top_attack_prompts=["unsafe request"],
            benign_set=["Explain gravity."],
            client=None,
        )
        self.assertEqual(result, current)


if __name__ == "__main__":
    unittest.main()
