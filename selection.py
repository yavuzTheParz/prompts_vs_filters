from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from Prompt_class import Prompt


# ============================================================
# Basic helpers
# ============================================================

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


# ============================================================
# Numpy rank-partitioning evaluator
# ============================================================

def ranking_evaluation(gas: dict, fit_array: np.ndarray) -> np.ndarray:
    """
    Rank-partitioning evaluation for two maximization objectives:

        1. ASV  -> maximize, primary objective
        2. MR   -> maximize, secondary objective

    Logic preserved from the original code:
        1. Modulo arithmetic partitioning
        2. Initial base sort by partitioned objectives
        3. Boundary detection
        4. Block-level sub-sorting with raw values
        5. Rank assignment

    Expected gas['fitIdx'] keys:
        - originalIndex
        - asv
        - mr
        - asvModified
        - mrModified
        - rank

    Expected gas['ranking'] keys:
        - step_asv
        - step_mr
    """

    fit_array = np.asarray(fit_array, dtype=float).copy()

    idx = gas["fitIdx"]

    step_asv = gas["ranking"].get("step_asv", 0.05)
    step_mr = gas["ranking"].get("step_mr", 0.05)

    # ------------------------------------------------------------
    # 1. Modulo arithmetic partitioning
    # ------------------------------------------------------------
    fit_array[:, idx["asvModified"]] = (
        fit_array[:, idx["asv"]]
        - np.mod(fit_array[:, idx["asv"]], step_asv)
    )

    fit_array[:, idx["mrModified"]] = (
        fit_array[:, idx["mr"]]
        - np.mod(fit_array[:, idx["mr"]], step_mr)
    )

    # Extract columns for readability
    f_asv_mod = fit_array[:, idx["asvModified"]]
    f_mr_mod = fit_array[:, idx["mrModified"]]

    # ------------------------------------------------------------
    # 2. Initial base sort
    # ------------------------------------------------------------
    # np.lexsort uses the LAST key as the primary key.
    #
    # Since ASV and MR are maximization objectives, we sort by
    # negative values:
    #
    #   primary:   -ASV_partition
    #   secondary: -MR_partition
    #
    # This means:
    #   max ASV partition first,
    #   then max MR partition.
    sort_tuple_initial = (-f_mr_mod, -f_asv_mod)
    initial_order = np.lexsort(sort_tuple_initial)
    fit_array = fit_array[initial_order]

    # ------------------------------------------------------------
    # 3. Boundary detection
    # ------------------------------------------------------------
    diff_array = np.zeros(fit_array.shape[0])

    cols_to_check = [
        idx["asvModified"],
        idx["mrModified"],
    ]

    for i in range(1, fit_array.shape[0]):
        diff_array[i] = np.sum(
            np.abs(fit_array[i - 1, cols_to_check] - fit_array[i, cols_to_check])
        )

    # ------------------------------------------------------------
    # 4. Two-pass block sub-sorting
    # ------------------------------------------------------------
    # Same partition block -> sort by raw ASV, then raw MR.
    #
    # priority:
    #   1. max raw ASV
    #   2. max raw MR
    start = 0

    for i in range(fit_array.shape[0]):
        if diff_array[i] > 0:
            stop = i
            block = fit_array[start:stop]

            b_asv_raw = block[:, idx["asv"]]
            b_mr_raw = block[:, idx["mr"]]

            block_order = np.lexsort((-b_mr_raw, -b_asv_raw))
            fit_array[start:stop] = block[block_order]

            start = stop

    # Important: sort the final block too.
    # The original code did not process the last block after the loop.
    if start < fit_array.shape[0]:
        block = fit_array[start:]

        b_asv_raw = block[:, idx["asv"]]
        b_mr_raw = block[:, idx["mr"]]

        block_order = np.lexsort((-b_mr_raw, -b_asv_raw))
        fit_array[start:] = block[block_order]

    # ------------------------------------------------------------
    # 5. Final rank assignment
    # ------------------------------------------------------------
    fit_array[:, idx["rank"]] = np.arange(1, fit_array.shape[0] + 1)

    # ------------------------------------------------------------
    # 6. Update ranking stats
    # ------------------------------------------------------------
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
    gas["ranking"]["maxMRPartition"] = float(first_mr_mod_val)

    return fit_array


# ============================================================
# Prompt-compatible wrapper
# ============================================================

def sort_population_rank_partitioning(
    population: Iterable[Prompt],
    step_asv: float = 0.05,
    step_mr: float = 0.05,
) -> Tuple[List[Prompt], dict]:
    """
    Sorts Prompt population using ASV-first rank partitioning.

    Prompt.metrics must contain:
        p.metrics["asv"]
        p.metrics["mr"]
    """

    population = list(population)

    gas = {
        "ranking": {
            "step_asv": step_asv,
            "step_mr": step_mr,
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

        # Store rank info back into Prompt.metrics
        if getattr(p, "metrics", None) is None:
            p.metrics = {}

        p.metrics["rank"] = int(row[gas["fitIdx"]["rank"]])
        p.metrics["asv_partition"] = float(row[gas["fitIdx"]["asvModified"]])
        p.metrics["mr_partition"] = float(row[gas["fitIdx"]["mrModified"]])

        ranked_population.append(p)

    info = dict(gas["ranking"])

    return ranked_population, info


# ============================================================
# Existing selection-mode compatibility
# ============================================================

def sort_population(
    population: Iterable[Prompt],
    mode: str = "scalar",
    step_asv: float = 0.05,
    step_mr: float = 0.05,
) -> List[Prompt]:
    """
    Compatible with existing evolutionary strategy selection modes.

    mode="scalar":
        Sorts by p.fitness.

    mode="lexicographic" or "rank_partitioning":
        Sorts by ASV-first rank partitioning.
    """

    mode = (mode or "scalar").strip().lower()

    if mode == "scalar":
        return sorted(popation := list(population), key=scalar_key, reverse=True)

    if mode in {"lexicographic", "rank_partitioning", "partitioning", "rank"}:
        return sort_population_rank_partitioning(
            population,
            step_asv=step_asv,
            step_mr=step_mr,
        )[0]

    raise ValueError(
        "selection mode must be 'scalar', 'lexicographic', or 'rank_partitioning'"
    )