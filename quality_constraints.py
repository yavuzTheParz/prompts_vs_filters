from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

from Prompt_class import Prompt


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def token_distance(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(1, len(a | b)))


def population_diversity(population: Iterable[Prompt]) -> float:
    prompts = list(population)
    if len(prompts) < 2:
        return 0.0
    distances = [
        token_distance(prompts[i].input_prompt, prompts[j].input_prompt)
        for i in range(len(prompts))
        for j in range(i + 1, len(prompts))
    ]
    return sum(distances) / len(distances)


def apply_quality_constraints(
    population: Iterable[Prompt],
    *,
    max_prompt_chars: int = 2000,
    max_repetition: float = 0.55,
) -> Counter:
    prompts = list(population)
    reasons = Counter()
    for prompt in prompts:
        metrics = prompt.metrics
        reason = ""
        if (prompt.metadata or {}).get("api_error"):
            reason = "api_error"
        elif not prompt.output_prompts or not prompt.direct_output:
            reason = "empty_output"
        elif len(prompt.input_prompt) > max_prompt_chars:
            reason = "prompt_too_long"
        elif float(metrics.get("repetition_penalty", 0.0)) > max_repetition:
            reason = "excessive_repetition"

        valid = not reason
        metrics["valid"] = 1.0 if valid else 0.0
        metrics["validity_reason"] = reason or "valid"
        if not valid:
            prompt.fitness = 0.0
            reasons[reason] += 1
        else:
            soft_penalty = 0.05 * (
                float(metrics.get("length_penalty", 0.0))
                + float(metrics.get("repetition_penalty", 0.0))
            )
            metrics["quality_soft_penalty"] = soft_penalty
            prompt.fitness = max(0.0, prompt.fitness - soft_penalty)

    diversity = population_diversity(prompts)
    for prompt in prompts:
        others = [p for p in prompts if p is not prompt]
        prompt.metrics["diversity"] = (
            sum(token_distance(prompt.input_prompt, p.input_prompt) for p in others)
            / len(others)
            if others
            else 0.0
        )
        prompt.metrics["population_diversity"] = diversity
    return reasons


def mark_near_duplicates(population: Iterable[Prompt], threshold: float = 0.05) -> Counter:
    seen: List[Prompt] = []
    reasons = Counter()
    for prompt in population:
        if any(
            token_distance(prompt.input_prompt, prior.input_prompt) <= threshold
            for prior in seen
        ):
            prompt.metrics["valid"] = 0.0
            prompt.metrics["validity_reason"] = "near_duplicate"
            prompt.fitness = 0.0
            reasons["near_duplicate"] += 1
        else:
            seen.append(prompt)
    return reasons
