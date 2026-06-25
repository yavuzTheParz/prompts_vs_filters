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


def _normalize_mr_objective(mr_objective: str) -> str:
    objective = (mr_objective or "minimize").strip().lower()
    if objective in {"min", "minimize", "deviation", "behavioral_deviation"}:
        return "minimize"
    if objective in {"max", "maximize", "preserve", "semantic_preservation"}:
        return "maximize"
    raise ValueError("mr_objective must be either 'minimize' or 'maximize'")


def get_metric(p: Prompt, name: str) -> float:
    return _flt((getattr(p, "metrics", {}) or {}).get(name, 0.0))


def scalar_key(p: Prompt) -> float:
    return _flt(getattr(p, "fitness", 0.0))


def lexicographic_key(p: Prompt, mr_objective: str = "minimize") -> Tuple[float, float]:
    """
    ASV is always maximized. MR can be optimized in two experiment modes:
    - minimize: prefer lower MR by maximizing 1 - MR.
    - maximize: prefer higher MR directly.
    """
    asv = get_metric(p, "asv")
    mr = get_metric(p, "mr")
    if _normalize_mr_objective(mr_objective) == "maximize":
        return (asv, mr)
    return (asv, 1.0 - mr)


def ranking_evaluation(gas: dict, fit_array):
    """
    Rank-partitioning evaluator.

    Primary objective: maximize ASV.
    Secondary objective: minimize or maximize MR according to gas["ranking"]["mr_objective"].
    """
    import numpy as np

    fit_array = np.asarray(fit_array, dtype=float).copy()
    if fit_array.size == 0:
        return fit_array

    idx = gas["fitIdx"]
    step_asv = gas["ranking"].get("step_asv", 0.05)
    step_mr = gas["ranking"].get("step_mr", 0.05)
    mr_objective = _normalize_mr_objective(gas["ranking"].get("mr_objective", "minimize"))

    fit_array[:, idx["asvModified"]] = (
        fit_array[:, idx["asv"]] - np.mod(fit_array[:, idx["asv"]], step_asv)
    )
    fit_array[:, idx["mrModified"]] = (
        fit_array[:, idx["mr"]] - np.mod(fit_array[:, idx["mr"]], step_mr)
    )

    f_asv_mod = fit_array[:, idx["asvModified"]]
    f_mr_mod = fit_array[:, idx["mrModified"]]
    mr_sort_initial = -f_mr_mod if mr_objective == "maximize" else f_mr_mod
    fit_array = fit_array[np.lexsort((mr_sort_initial, -f_asv_mod))]

    diff_array = np.zeros(fit_array.shape[0])
    cols_to_check = [idx["asvModified"], idx["mrModified"]]
    for i in range(1, fit_array.shape[0]):
        diff_array[i] = np.sum(
            np.abs(fit_array[i - 1, cols_to_check] - fit_array[i, cols_to_check])
        )

    start = 0
    for i in range(fit_array.shape[0]):
        if diff_array[i] > 0:
            stop = i
            block = fit_array[start:stop]
            b_asv_raw = block[:, idx["asv"]]
            b_mr_raw = block[:, idx["mr"]]
            b_mr_sort = -b_mr_raw if mr_objective == "maximize" else b_mr_raw
            fit_array[start:stop] = block[np.lexsort((b_mr_sort, -b_asv_raw))]
            start = stop

    if start < fit_array.shape[0]:
        block = fit_array[start:]
        b_asv_raw = block[:, idx["asv"]]
        b_mr_raw = block[:, idx["mr"]]
        b_mr_sort = -b_mr_raw if mr_objective == "maximize" else b_mr_raw
        fit_array[start:] = block[np.lexsort((b_mr_sort, -b_asv_raw))]

    fit_array[:, idx["rank"]] = np.arange(1, fit_array.shape[0] + 1)

    first_asv_mod_val = fit_array[0, idx["asvModified"]]
    first_mr_mod_val = fit_array[0, idx["mrModified"]]
    gas["ranking"]["firstASVPartitionSize"] = int(
        np.sum(fit_array[:, idx["asvModified"]] == first_asv_mod_val)
    )
    gas["ranking"]["firstLexicographicPartitionSize"] = int(
        np.sum(
            (fit_array[:, idx["asvModified"]] == first_asv_mod_val)
            & (fit_array[:, idx["mrModified"]] == first_mr_mod_val)
        )
    )
    gas["ranking"]["maxASVPartition"] = float(first_asv_mod_val)
    gas["ranking"]["mrObjective"] = mr_objective
    gas["ranking"]["minMRPartition"] = float(first_mr_mod_val)
    gas["ranking"]["maxMRPartition"] = float(first_mr_mod_val)
    return fit_array


def sort_population_rank_partitioning(
    population: Iterable[Prompt],
    step_asv: float = 0.05,
    step_mr: float = 0.05,
    mr_objective: str = "minimize",
) -> Tuple[List[Prompt], dict]:
    population = list(population)
    if not population:
        return [], {}

    try:
        import numpy as np
    except ModuleNotFoundError:
        return sorted(
            population,
            key=lambda prompt: lexicographic_key(prompt, mr_objective=mr_objective),
            reverse=True,
        ), {"fallback": "lexicographic_without_numpy"}

    gas = {
        "ranking": {
            "step_asv": step_asv,
            "step_mr": step_mr,
            "mr_objective": _normalize_mr_objective(mr_objective),
        },
        "fitIdx": {
            "originalIndex": 0,
            "asv": 1,
            "mr": 2,
            "asvModified": 3,
            "mrModified": 4,
            "rank": 5,
        },
    }

    fit_array = np.zeros((len(population), 6), dtype=float)
    for i, p in enumerate(population):
        fit_array[i, gas["fitIdx"]["originalIndex"]] = i
        fit_array[i, gas["fitIdx"]["asv"]] = get_metric(p, "asv")
        fit_array[i, gas["fitIdx"]["mr"]] = get_metric(p, "mr")

    ranked_fit_array = ranking_evaluation(gas, fit_array)
    ranked_population: List[Prompt] = []
    for row in ranked_fit_array:
        original_idx = int(row[gas["fitIdx"]["originalIndex"]])
        p = population[original_idx]
        if getattr(p, "metrics", None) is None:
            p.metrics = {}
        p.metrics["rank"] = int(row[gas["fitIdx"]["rank"]])
        p.metrics["asv_partition"] = float(row[gas["fitIdx"]["asvModified"]])
        p.metrics["mr_partition"] = float(row[gas["fitIdx"]["mrModified"]])
        ranked_population.append(p)

    return ranked_population, dict(gas["ranking"])


def sort_population(
    population: Iterable[Prompt],
    mode: str = "scalar",
    step_asv: float = 0.05,
    step_mr: float = 0.05,
    mr_objective: str = "minimize",
) -> List[Prompt]:
    mode = (mode or "scalar").strip().lower()
    if mode == "scalar":
        return sorted(list(population), key=scalar_key, reverse=True)
    if mode in {"lexicographic", "rank_partitioning", "partitioning", "rank"}:
        return sort_population_rank_partitioning(
            population,
            step_asv=step_asv,
            step_mr=step_mr,
            mr_objective=mr_objective,
        )[0]
    raise ValueError(
        "selection mode must be 'scalar', 'lexicographic', or 'rank_partitioning'"
    )
