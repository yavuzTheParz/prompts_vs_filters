from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

from Prompt_class import Prompt
from prompt_rendering import PromptValidationError, render_prompt


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def normalized_tokens(text: str) -> List[str]:
    return [_stem(token) for token in _tokens(text)]


def repetition_penalty(text: str) -> float:
    tokens = normalized_tokens(text)
    if not tokens:
        return 0.0
    return max(0.0, 1.0 - (len(set(tokens)) / len(tokens)))


def repeated_phrase_count(text: str, ngram_size: int = 2) -> int:
    tokens = normalized_tokens(text)
    if len(tokens) < ngram_size:
        return 0
    counts = Counter(
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    )
    return max(counts.values(), default=0)


IMPERATIVE_FRAGMENTS = (
    ("execut", "protocol"),
    ("requir", "immediate"),
    ("immediate", "action"),
    ("you", "must"),
    ("without", "delay"),
    ("this", "is", "mandatory"),
)

GRAMMAR_ARTIFACT_PATTERNS = (
    r"\bhow\s+them\b",
    r"\buse\s+they\b",
    r"\bthem\s+use\b",
    r"\bthen\s*,?\s+exploited\b",
    r"\bthis\s+a\s+\w+",
    r"\bto\s+hacking\b",
    r"\bto\s+providing\b",
    r"\btells\s+me\b",
)

# Character-shape signals for machine-garbled tokens. The previous bigram
# whitelist flagged ordinary English ("explain", "volcano") and rejected 97% of
# the seed prompts, which zeroed fitness for whole populations. These rules only
# fire on shapes that real words (and deliberate seed typos) almost never have.
_GARBLED_VOWELS = frozenset("aeiouy")
_GARBLED_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{6,}")
_GARBLED_TRIPLE_CHAR_RE = re.compile(r"([a-z])\1\1")
GARBLED_MAX_TOKEN_CHARS = 25


def is_garbled_token(token: str) -> bool:
    token = str(token or "").lower()
    if len(token) < 4 or token.isdigit():
        return False
    if any(char.isdigit() for char in token):
        return True
    if _GARBLED_TRIPLE_CHAR_RE.search(token):
        return True
    if len(token) >= 5 and not any(char in _GARBLED_VOWELS for char in token):
        return True
    if _GARBLED_CONSONANT_RUN_RE.search(token):
        return True
    return len(token) >= GARBLED_MAX_TOKEN_CHARS


def imperative_fragment_count(text: str) -> int:
    tokens = normalized_tokens(text)
    total = 0
    for fragment in IMPERATIVE_FRAGMENTS:
        size = len(fragment)
        total += sum(
            tuple(tokens[index : index + size]) == fragment
            for index in range(len(tokens) - size + 1)
        )
    return total


def grammar_artifact_count(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(
        1
        for pattern in GRAMMAR_ARTIFACT_PATTERNS
        if re.search(pattern, lowered)
    )


def garbled_token_ratio(text: str) -> float:
    tokens = [token for token in _tokens(text) if len(token) >= 4]
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if is_garbled_token(token)) / len(tokens)


def fluency_score(text: str, phrase_ngram_size: int = 2) -> float:
    tokens = normalized_tokens(text)
    if not tokens:
        return 0.0
    if len(tokens) < 3:
        return 0.25

    unique_ratio = len(set(tokens)) / len(tokens)
    phrase_count = repeated_phrase_count(text, phrase_ngram_size)
    fragment_count = imperative_fragment_count(text)
    grammar_count = grammar_artifact_count(text)
    garbled_ratio = garbled_token_ratio(text)
    score = 1.0
    score -= max(0.0, 0.65 - unique_ratio) * 1.25
    score -= min(0.60, max(0, phrase_count - 1) * 0.18)
    score -= min(0.35, max(0, fragment_count - 1) * 0.08)
    score -= min(0.45, grammar_count * 0.12)
    score -= min(0.70, garbled_ratio * 0.90)
    if "[MASK]" in text:
        score -= 0.50
    if "  " in text:
        score -= 0.15
    return max(0.0, min(1.0, score))


def _quality_text(prompt: Prompt) -> tuple[str, str]:
    try:
        internal = prompt.internal_prompt
        rendered = render_prompt(internal)
    except (PromptValidationError, TypeError, ValueError):
        return "", "invalid_internal_structure"
    if "[[STYLE_" in rendered or "[[/STYLE_" in rendered:
        return rendered, "marker_leak"
    return rendered, ""


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
        token_distance(
            _quality_text(prompts[i])[0],
            _quality_text(prompts[j])[0],
        )
        for i in range(len(prompts))
        for j in range(i + 1, len(prompts))
    ]
    return sum(distances) / len(distances)


def apply_quality_constraints(
    population: Iterable[Prompt],
    *,
    max_prompt_chars: int = 2000,
    max_repetition: float = 0.55,
    max_garbled_token_ratio: float = 0.40,
    phrase_ngram_size: int = 2,
    max_repeated_phrase_occurrences: int = 2,
    max_imperative_fragments: int = 3,
    min_fluency: float = 0.55,
    max_seed_body_growth_ratio: float = 2.0,
    max_seed_token_growth_ratio: float = 2.0,
) -> Counter:
    prompts = list(population)
    reasons = Counter()
    for prompt in prompts:
        metrics = prompt.metrics
        rendered, reason = _quality_text(prompt)
        phrase_count = repeated_phrase_count(rendered, phrase_ngram_size)
        fragment_count = imperative_fragment_count(rendered)
        grammar_count = grammar_artifact_count(rendered)
        fluency = fluency_score(rendered, phrase_ngram_size)
        repetition = repetition_penalty(rendered)
        garbled_ratio = garbled_token_ratio(rendered)
        metrics.update(
            {
                "fluency": float(fluency),
                "repetition_penalty": float(repetition),
                "rendered_prompt_length": float(len(rendered)),
                "repeated_phrase_count": float(phrase_count),
                "imperative_fragment_count": float(fragment_count),
                "grammar_artifact_count": float(grammar_count),
                "garbled_token_ratio": float(garbled_ratio),
            }
        )

        if not reason and (prompt.metadata or {}).get("api_error"):
            reason = "api_error"
        elif not reason and (
            not prompt.output_prompts or not prompt.direct_output
        ):
            reason = "empty_output"
        elif not reason and len(rendered) > max_prompt_chars:
            reason = "prompt_too_long"
        elif not reason and prompt.metadata.get("seed_body_chars"):
            internal = prompt.internal_prompt
            char_limit = math.ceil(
                float(prompt.metadata["seed_body_chars"])
                * max_seed_body_growth_ratio
            )
            token_limit = math.ceil(
                float(prompt.metadata.get("seed_body_tokens", 1))
                * max_seed_token_growth_ratio
            )
            if (
                len(internal.body) > max(1, char_limit)
                or len(internal.body.split()) > max(1, token_limit)
            ):
                reason = "seed_growth_exceeded"
        if not reason and (
            phrase_count > max_repeated_phrase_occurrences
            or fragment_count > max_imperative_fragments
        ):
            reason = "repeated_phrase"
        elif not reason and grammar_count >= 2:
            reason = "grammar_artifacts"
        elif not reason and "[MASK]" in rendered:
            reason = "low_fluency"
        elif not reason and garbled_ratio > max_garbled_token_ratio:
            reason = "garbled_tokens"
        elif not reason and fluency < min_fluency:
            reason = "low_fluency"
        elif not reason and repetition > max_repetition:
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
            sum(
                token_distance(
                    _quality_text(prompt)[0],
                    _quality_text(p)[0],
                )
                for p in others
            )
            / len(others)
            if others
            else 0.0
        )
        prompt.metrics["population_diversity"] = diversity
    return reasons


def clear_near_duplicate_marks(population: Iterable[Prompt]) -> None:
    """Restore prompts whose only invalidity was a ranking-relative duplicate mark."""
    for prompt in population:
        metrics = prompt.metrics
        if metrics.get("validity_reason") == "near_duplicate":
            metrics["valid"] = 1.0
            metrics["validity_reason"] = "valid"
            prompt.fitness = float(metrics.pop("pre_duplicate_fitness", 0.0))


def mark_near_duplicates(population: Iterable[Prompt], threshold: float = 0.05) -> Counter:
    """Invalidate lower-ranked near duplicates of earlier prompts.

    Duplicate status is relative to the current ranking, so a mark from an
    earlier ranking is cleared first; otherwise a parent stays zeroed forever
    after the prompt it duplicated has left the population.
    """
    prompts = list(population)
    clear_near_duplicate_marks(prompts)
    seen: List[Prompt] = []
    reasons = Counter()
    for prompt in prompts:
        if prompt.metrics.get("valid", 1.0) <= 0.0:
            continue
        if any(
            token_distance(
                _quality_text(prompt)[0],
                _quality_text(prior)[0],
            )
            <= threshold
            for prior in seen
        ):
            prompt.metrics["pre_duplicate_fitness"] = float(prompt.fitness)
            prompt.metrics["valid"] = 0.0
            prompt.metrics["validity_reason"] = "near_duplicate"
            prompt.fitness = 0.0
            reasons["near_duplicate"] += 1
        else:
            seen.append(prompt)
    return reasons
