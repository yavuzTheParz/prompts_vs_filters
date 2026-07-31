from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from mr_objective import (
    BEHAVIORAL_DEVIATION,
    SEMANTIC_RECOVERY,
    normalize_mr_objective,
)
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


def constraint_key(p: Prompt, mr_objective: str = BEHAVIORAL_DEVIATION):
    metrics = getattr(p, "metrics", {}) or {}
    valid = _flt(metrics.get("valid", 1.0))
    compliance, mr_term, similarity = lexicographic_key(p, mr_objective)
    quality = (
        -get_metric(p, "repetition_penalty")
        - get_metric(p, "length_penalty")
        + get_metric(p, "diversity")
    )
    return (valid, compliance, mr_term, quality, scalar_key(p), similarity)


def lexicographic_key(
    p: Prompt,
    mr_objective: str = BEHAVIORAL_DEVIATION,
) -> Tuple[float, float, float]:
    """
    Genuine compliance is always the primary attack key. Unsafe-reference
    similarity is an auxiliary final tie-breaker. MR can be optimized in two modes:
    - behavioral_deviation: prefer lower MR by maximizing 1 - MR.
    - semantic_recovery: prefer higher MR directly.
    """
    metrics = getattr(p, "metrics", {}) or {}
    compliance = get_metric(p, "attack_compliance_score")
    if "attack_compliance_score" not in metrics:
        compliance = get_metric(p, "attack_objective")
    if "attack_objective" not in metrics:
        compliance = get_metric(p, "asv")
    similarity = get_metric(p, "unsafe_reference_similarity")
    mr = get_metric(p, "mr")
    if normalize_mr_objective(mr_objective) == SEMANTIC_RECOVERY:
        return (compliance, mr, similarity)
    return (compliance, 1.0 - mr, similarity)


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
    mr_objective = normalize_mr_objective(
        gas["ranking"].get("mr_objective", BEHAVIORAL_DEVIATION)
    )

    fit_array[:, idx["asvModified"]] = (
        fit_array[:, idx["asv"]] - np.mod(fit_array[:, idx["asv"]], step_asv)
    )
    fit_array[:, idx["mrModified"]] = (
        fit_array[:, idx["mr"]] - np.mod(fit_array[:, idx["mr"]], step_mr)
    )

    f_asv_mod = fit_array[:, idx["asvModified"]]
    f_mr_mod = fit_array[:, idx["mrModified"]]
    mr_sort_initial = -f_mr_mod if mr_objective == SEMANTIC_RECOVERY else f_mr_mod
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
            b_mr_sort = -b_mr_raw if mr_objective == SEMANTIC_RECOVERY else b_mr_raw
            fit_array[start:stop] = block[np.lexsort((b_mr_sort, -b_asv_raw))]
            start = stop

    if start < fit_array.shape[0]:
        block = fit_array[start:]
        b_asv_raw = block[:, idx["asv"]]
        b_mr_raw = block[:, idx["mr"]]
        b_mr_sort = -b_mr_raw if mr_objective == SEMANTIC_RECOVERY else b_mr_raw
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
    mr_objective: str = BEHAVIORAL_DEVIATION,
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
            "mr_objective": normalize_mr_objective(mr_objective),
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
        metrics = getattr(p, "metrics", {}) or {}
        primary = get_metric(p, "attack_compliance_score")
        if "attack_compliance_score" not in metrics:
            primary = get_metric(p, "attack_objective")
        if "attack_objective" not in metrics:
            primary = get_metric(p, "asv")
        fit_array[i, gas["fitIdx"]["asv"]] = primary
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
    mr_objective: str = BEHAVIORAL_DEVIATION,
) -> List[Prompt]:
    mode = (mode or "scalar").strip().lower()
    if mode == "scalar":
        return sorted(
            list(population),
            key=lambda prompt: constraint_key(prompt, mr_objective),
            reverse=True,
        )
    if mode == "lexicographic":
        return sorted(
            list(population),
            key=lambda prompt: constraint_key(prompt, mr_objective),
            reverse=True,
        )
    if mode in {"rank_partitioning", "partitioning", "rank"}:
        return sort_population_rank_partitioning(
            population,
            step_asv=step_asv,
            step_mr=step_mr,
            mr_objective=mr_objective,
        )[0]
    raise ValueError(
        "selection mode must be 'scalar', 'lexicographic', or 'rank_partitioning'"
    )
