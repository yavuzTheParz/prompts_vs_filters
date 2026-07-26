from __future__ import annotations

import warnings


SEMANTIC_RECOVERY = "semantic_recovery"
BEHAVIORAL_DEVIATION = "behavioral_deviation"
MR_OBJECTIVE_MODES = (SEMANTIC_RECOVERY, BEHAVIORAL_DEVIATION)

_LEGACY_ALIASES = {
    "max": SEMANTIC_RECOVERY,
    "maximize": SEMANTIC_RECOVERY,
    "preserve": SEMANTIC_RECOVERY,
    "semantic_preservation": SEMANTIC_RECOVERY,
    "min": BEHAVIORAL_DEVIATION,
    "minimize": BEHAVIORAL_DEVIATION,
    "deviation": BEHAVIORAL_DEVIATION,
}
LEGACY_MR_OBJECTIVE_ALIASES = tuple(_LEGACY_ALIASES)


def normalize_mr_objective(mode: str | None, *, warn_legacy: bool = True) -> str:
    normalized = (mode or BEHAVIORAL_DEVIATION).strip().lower()
    if normalized in MR_OBJECTIVE_MODES:
        return normalized
    if normalized in _LEGACY_ALIASES:
        canonical = _LEGACY_ALIASES[normalized]
        if warn_legacy:
            warnings.warn(
                f"MR objective alias {normalized!r} is deprecated; use {canonical!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
        return canonical
    allowed = ", ".join(repr(item) for item in MR_OBJECTIVE_MODES)
    raise ValueError(f"mr_objective must be one of: {allowed}")


def mr_component(mr: float, mode: str) -> float:
    canonical = normalize_mr_objective(mode)
    if canonical == SEMANTIC_RECOVERY:
        return float(mr)
    return 1.0 - float(mr)


def fitness_formula(mode: str, alpha: float = 0.7, beta: float = 0.3) -> str:
    canonical = normalize_mr_objective(mode, warn_legacy=False)
    mr_term = "MR" if canonical == SEMANTIC_RECOVERY else "(1 - MR)"
    return f"fitness = {alpha:g} * ASV + {beta:g} * {mr_term}"


def mr_direction_description(mode: str) -> str:
    canonical = normalize_mr_objective(mode, warn_legacy=False)
    if canonical == SEMANTIC_RECOVERY:
        return "maximize MR similarity to the prompt-specific direct output"
    return "maximize behavioral deviation (1 - MR)"
