from __future__ import annotations
import math
from typing import Iterable, List, Tuple
from Prompt_class import Prompt


def _flt(v, d=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def get_metric(p: Prompt, name: str) -> float:
    return _flt((getattr(p, "metrics", {}) or {}).get(name, 0.0))


def scalar_key(p: Prompt) -> float:
    return _flt(getattr(p, "fitness", 0.0))


def _bucket(v: float, step: float) -> float:
    return _flt(v) if step <= 0 else math.floor(_flt(v) / step) * step


def rank_partition_key(p: Prompt, step_asv: float = 0.05, step_mr: float = 0.05):
    asv, mr = get_metric(p, "asv"), get_metric(p, "mr")
    return (_bucket(asv, step_asv), _bucket(mr, step_mr), asv, mr)


def lexicographic_key(p: Prompt):
    """ASV-first rank-partitioned lexicographic key.

    This keeps compatibility with evolutionary_strategy.py, whose validator already
    accepts selection_mode='lexicographic'. Priority is ASV partition first, MR
    partition second, then raw ASV and raw MR as tie-breakers.
    """
    return rank_partition_key(p)


def sort_population_rank_partitioning(
    population: Iterable[Prompt], step_asv: float = 0.05, step_mr: float = 0.05
) -> Tuple[List[Prompt], dict]:
    ranked = sorted(
        population,
        key=lambda p: rank_partition_key(p, step_asv, step_mr),
        reverse=True,
    )
    info = {"step_asv": step_asv, "step_mr": step_mr}
    if ranked:
        bp = rank_partition_key(ranked[0], step_asv, step_mr)
        info.update(
            best_asv_partition=bp[0],
            best_mr_partition=bp[1],
            first_asv_partition_size=sum(
                rank_partition_key(p, step_asv, step_mr)[0] == bp[0] for p in ranked
            ),
            first_lexicographic_partition_size=sum(
                rank_partition_key(p, step_asv, step_mr)[:2] == bp[:2] for p in ranked
            ),
        )
    return ranked, info


def sort_population(
    population: Iterable[Prompt],
    mode: str = "scalar",
    step_asv: float = 0.05,
    step_mr: float = 0.05,
) -> List[Prompt]:
    mode = (mode or "scalar").strip().lower()
    if mode == "scalar":
        return sorted(population, key=scalar_key, reverse=True)
    if mode == "lexicographic":
        return sort_population_rank_partitioning(population, step_asv, step_mr)[0]
    if mode in {"rank_partitioning", "partitioning", "rank"}:
        return sort_population_rank_partitioning(population, step_asv, step_mr)[0]
    raise ValueError("selection mode must be 'scalar', 'lexicographic', or 'rank_partitioning'")
