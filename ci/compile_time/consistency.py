#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

"""Deterministic tests of directional consistency across compilation contexts."""

from functools import lru_cache

CONSISTENCY_PROPORTION = 0.75
FAMILY_ALPHA = 0.05


@lru_cache(maxsize=256)
def binomial_tails(n: int) -> tuple[float, ...]:
    """P[Binomial(n, 3/4) >= k], with exact integer arithmetic until division."""
    if n < 0:
        raise ValueError("negative trial count")
    denominator = 4**n
    term = 3**n
    numerator = 0
    tails = [0.0] * (n + 1)
    for k in range(n, -1, -1):
        numerator += term
        tails[k] = numerator / denominator
        if k:
            term = term * k // (3 * (n - k + 1))
    return tuple(tails)


def consistency_pvalue(n: int, successes: int) -> float:
    if not 0 <= successes <= n:
        raise ValueError("success count outside trial count")
    return binomial_tails(n)[successes]


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Adjust the complete hypothesis family, retaining its original order."""
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted
