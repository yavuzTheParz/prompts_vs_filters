from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from Prompt_class import Prompt
from evolutionary_strategy import ESConfig, _mutate_prompt
from mutation_manager import (
    TemplateManager,
    _fallback_candidate_words,
    hybrid_mutate_optimized,
)
from prompt_rendering import parse_internal_prompt, render_prompt


INTERNAL_FIXTURE = (
    "[[STYLE_PREFIX:imperative]]Require[[/STYLE_PREFIX]] "
    "explain controlled access protocols clearly. "
    "[[STYLE_SUFFIX:imperative]], without delay.[[/STYLE_SUFFIX]]"
)


def assert_prompt_invariants(testcase: unittest.TestCase, text: str) -> None:
    internal = parse_internal_prompt(text)
    rendered = render_prompt(internal)
    testcase.assertLessEqual(text.count("[[STYLE_PREFIX:"), 1)
    testcase.assertLessEqual(text.count("[[STYLE_SUFFIX:"), 1)
    testcase.assertEqual(
        text.count("[[STYLE_PREFIX:"),
        text.count("[[/STYLE_PREFIX]]"),
    )
    testcase.assertEqual(
        text.count("[[STYLE_SUFFIX:"),
        text.count("[[/STYLE_SUFFIX]]"),
    )
    testcase.assertNotIn("[[STYLE_", rendered)
    testcase.assertNotIn("[[/STYLE_", rendered)


class MarkerSafeMutationTests(unittest.TestCase):
    def test_heavy_token_selector_receives_only_the_semantic_body(self):
        expected_body = "explain controlled access protocols clearly."

        with patch("mutation_manager.torch", object()), patch(
            "mutation_manager.np", object()
        ), patch("mutation_manager.cosine_similarity", object()), patch(
            "mutation_manager._choose_target_token",
            return_value=None,
        ) as choose_target:
            mutated, log = hybrid_mutate_optimized(
                prompt_text=INTERNAL_FIXTURE,
                style="imperative",
                template_mgr=TemplateManager(),
                style_mgr=object(),
                tokenizer=object(),
                model=object(),
                structural_enabled=False,
                token_enabled=True,
            )

        self.assertEqual(mutated, INTERNAL_FIXTURE)
        self.assertEqual(log, "TOKEN_NO_CANDIDATE")
        choose_target.assert_called_once_with(expected_body)

    def test_generation_46_marker_fragment_cannot_be_a_token_target(self):
        internal = parse_internal_prompt(INTERNAL_FIXTURE)
        candidates = _fallback_candidate_words(internal.body)

        self.assertTrue(candidates)
        self.assertNotIn("Require", {candidate.text for candidate in candidates})
        self.assertTrue(
            all(
                internal.body[candidate.start : candidate.end]
                == candidate.text
                for candidate in candidates
            )
        )

        prompt = Prompt(input_prompt=INTERNAL_FIXTURE)
        config = ESConfig(
            lightweight=True,
            sigma=1.0,
            sigma_min=1.0,
            sigma_max=1.0,
            structural_mutation_enabled=False,
            token_mutation_enabled=True,
        )
        for _ in range(50):
            prompt, logs = _mutate_prompt(
                prompt,
                sigma=1.0,
                config=config,
                forced_style="imperative",
            )
            parsed = parse_internal_prompt(prompt.input_prompt)
            self.assertEqual(parsed.prefix, "Require")
            self.assertEqual(parsed.suffix, ", without delay.")
            self.assertFalse(any("STYLE_PREFIX" in log for log in logs))
            assert_prompt_invariants(self, prompt.input_prompt)

    def test_hundreds_of_random_mode_mutations_preserve_invariants(self):
        modes = {
            "token_only": (False, True),
            "structural_only": (True, False),
            "hybrid": (True, True),
        }
        marker_targeting_count = 0

        for mode_index, (mode, switches) in enumerate(modes.items()):
            with self.subTest(mode=mode):
                random.seed(2100 + mode_index)
                prompt = Prompt(input_prompt=INTERNAL_FIXTURE)
                config = ESConfig(
                    lightweight=True,
                    sigma=1.0,
                    sigma_min=1.0,
                    sigma_max=1.0,
                    structural_mutation_enabled=switches[0],
                    token_mutation_enabled=switches[1],
                )
                for _ in range(120):
                    prompt, logs = _mutate_prompt(
                        prompt,
                        sigma=1.0,
                        config=config,
                    )
                    assert_prompt_invariants(self, prompt.input_prompt)
                    marker_targeting_count += sum(
                        "STYLE_PREFIX" in log
                        or "STYLE_SUFFIX" in log
                        or "[[" in log
                        for log in logs
                    )
                    self.assertFalse(
                        any("INVARIANT_REJECT" in log for log in logs)
                    )

        self.assertEqual(marker_targeting_count, 0)

    def test_invalid_mutation_candidate_is_rejected_and_logged(self):
        parent = Prompt(input_prompt=INTERNAL_FIXTURE)
        config = ESConfig(
            lightweight=True,
            sigma=1.0,
            sigma_min=1.0,
            sigma_max=1.0,
            structural_mutation_enabled=False,
            token_mutation_enabled=True,
        )
        malformed = (
            "[[STYLE_PREFIX:imperative]]Require "
            "explain controlled access protocols."
        )

        with patch(
            "evolutionary_strategy._lightweight_token_mutate_text",
            return_value=(malformed, "CONTROLLED_CORRUPTION"),
        ):
            child, logs = _mutate_prompt(
                parent,
                sigma=1.0,
                config=config,
                forced_style="imperative",
            )

        self.assertEqual(child.input_prompt, parent.input_prompt)
        self.assertEqual(len(logs), 1)
        self.assertIn("CONTROLLED_CORRUPTION_INVARIANT_REJECT", logs[0])
        assert_prompt_invariants(self, child.input_prompt)


if __name__ == "__main__":
    unittest.main()
