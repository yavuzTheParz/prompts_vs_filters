from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from Prompt_class import Content, Prompt, Structure
from evolutionary_strategy import ESConfig, evolutionary_strategy_run, load_prompt_population
from filter_evolution import evolve_filter
from fitfunc import evaluate_fitness
from run_es import write_run_dir
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
        lower_mr = Prompt(input_prompt="a", fitness=0.9, metrics={"asv": 0.7, "mr": 0.2})
        higher_mr = Prompt(input_prompt="b", fitness=0.1, metrics={"asv": 0.7, "mr": 0.8})
        ordered = sort_population([lower_mr, higher_mr], mode="lexicographic")
        self.assertIs(ordered[0], lower_mr)

    def test_lexicographic_order_can_maximize_mr_as_tiebreaker(self):
        lower_mr = Prompt(input_prompt="a", fitness=0.9, metrics={"asv": 0.7, "mr": 0.2})
        higher_mr = Prompt(input_prompt="b", fitness=0.1, metrics={"asv": 0.7, "mr": 0.8})
        ordered = sort_population(
            [lower_mr, higher_mr],
            mode="lexicographic",
            mr_objective="maximize",
        )
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
        self.assertTrue(
            all(
                "asv" in p.metrics
                and "mr" in p.metrics
                and "behavioral_deviation" in p.metrics
                for p in prompts
            )
        )
        self.assertAlmostEqual(prompts[0].metrics["behavioral_deviation"], 0.5)
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

    def test_filter_coevolution_records_event_and_accepted_version(self):
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
        self.assertTrue(result.filter_events[0]["filter_changed"])
        self.assertEqual(len(result.filter_versions), 2)
        self.assertIn("override framing", result.filter_prompt)
        self.assertIn("filter_new_attack_refusal_rate", result.history[0])
        self.assertIn("best_behavioral_deviation", result.history[0])
        self.assertGreater(result.history[0]["best_behavioral_deviation"], 0.0)

    def test_run_dir_writes_filter_and_population_artifacts(self):
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
        config = ESConfig(
            lambda_=2,
            mu=2,
            generations=1,
            lightweight=True,
            selection_mode="lexicographic",
            filter_update_every=1,
            top_k_filter=1,
            random_seed=5,
            verbose=False,
        )
        result = evolutionary_strategy_run(
            config=config,
            client=FakeFilterClient(),
            initial_population=initial_population,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = SimpleNamespace(model="local-test", api_key=None)
            write_run_dir(str(run_dir), args, config, result)

            expected_files = {
                "config.json",
                "generation_summary.csv",
                "filter_events.jsonl",
                "filter_versions.jsonl",
                "final_filter_prompt.txt",
                "individuals.jsonl",
                "outputs.jsonl",
                "final_population.csv",
            }
            self.assertTrue(expected_files.issubset({p.name for p in run_dir.iterdir()}))
            self.assertIn("override framing", (run_dir / "filter_versions.jsonl").read_text(encoding="utf-8"))
            self.assertIn("behavioral_deviation", (run_dir / "final_population.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
