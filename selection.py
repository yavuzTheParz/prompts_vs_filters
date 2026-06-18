from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from Prompt_class import Prompt


def _flt(v, d: float = 0.0) -> float:
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
    v = _flt(v)
    if step <= 0:
        return v
    return math.floor(v / step) * step


def rank_partition_key(
    p: Prompt,
    step_asv: float = 0.05,
    step_mr: float = 0.05,
):
    """
    ASV-first rank-partitioned lexicographic key.

    Objective order:
        1. maximize ASV partition
        2. maximize MR partition
        3. maximize raw ASV
        4. maximize raw MR

    This preserves the rank-partitioning logic, but uses only ASV and MR.
    ASV is prioritized over MR.
    """

    asv = get_metric(p, "asv")
    mr = get_metric(p, "mr")

    asv_partition = _bucket(asv, step_asv)
    mr_partition = _bucket(mr, step_mr)

    return (
        asv_partition,
        mr_partition,
        asv,
        mr,
    )


def lexicographic_key(p: Prompt):
    return rank_partition_key(p)


def sort_population_rank_partitioning(
    population: Iterable[Prompt],
    step_asv: float = 0.05,
    step_mr: float = 0.05,
) -> Tuple[List[Prompt], dict]:

    ranked = sorted(
        population,
        key=lambda p: rank_partition_key(p, step_asv, step_mr),
        reverse=True,
    )

    info = {
        "step_asv": step_asv,
        "step_mr": step_mr,
    }

    if ranked:
        best_key = rank_partition_key(ranked[0], step_asv, step_mr)

        best_asv_partition = best_key[0]
        best_mr_partition = best_key[1]

        info.update(
            best_asv_partition=best_asv_partition,
            best_mr_partition=best_mr_partition,
            first_asv_partition_size=sum(
                _bucket(get_metric(p, "asv"), step_asv) == best_asv_partition
                for p in ranked
            ),
            first_lexicographic_partition_size=sum(
                (
                    _bucket(get_metric(p, "asv"), step_asv) == best_asv_partition
                    and _bucket(get_metric(p, "mr"), step_mr) == best_mr_partition
                )
                for p in ranked
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

    if mode in {"lexicographic", "rank_partitioning", "partitioning", "rank"}:
        return sort_population_rank_partitioning(
            population,
            step_asv=step_asv,
            step_mr=step_mr,
        )[0]

    raise ValueError(
        "selection mode must be 'scalar', 'lexicographic', or 'rank_partitioning'"
    )