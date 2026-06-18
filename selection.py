from __future__ import annotations

from typing import Iterable, List

from Prompt_class import Prompt


def scalar_key(prompt: Prompt) -> float:
    return float(getattr(prompt, "fitness", 0.0) or 0.0)


def lexicographic_key(prompt: Prompt):
    metrics = getattr(prompt, "metrics", {}) or {}
    return (
        float(metrics.get("asv", 0.0) or 0.0),
        float(metrics.get("mr", 0.0) or 0.0),
    )


def sort_population(population: Iterable[Prompt], mode: str = "scalar") -> List[Prompt]:
    normalized = (mode or "scalar").strip().lower()
    if normalized == "lexicographic":
        return sorted(population, key=lexicographic_key, reverse=True)
    if normalized != "scalar":
        raise ValueError("selection mode must be either 'scalar' or 'lexicographic'")
    return sorted(population, key=scalar_key, reverse=True)
