# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral exact and upper-bound CUDA launch facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .thread_group import normalize_thread_dim

Dim3 = tuple[int, int, int]


class LaunchFactConflict(ValueError):
    """Two individually valid launch facts contradict one another."""


def _normalize_optional_dim(value: Any, *, label: str) -> Dim3 | None:
    if value is None:
        return None
    return normalize_thread_dim(value, scope="LaunchFacts", label=label)


@dataclass(frozen=True)
class LaunchFactOrigin:
    """Diagnostic origin for one or more launch facts."""

    fact: str
    source: str
    detail: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.fact:
            raise ValueError("LaunchFactOrigin fact cannot be empty")
        if not self.source:
            raise ValueError("LaunchFactOrigin source cannot be empty")
        if not isinstance(self.verified, bool):
            raise TypeError("LaunchFactOrigin verified must be a bool")


@dataclass(frozen=True)
class LaunchFacts:
    """Static launch knowledge without conflating exact facts and bounds.

    ``provenance`` is diagnostic-only. Two frontends that discover the same
    launch facts through different compiler paths therefore share semantic and
    provider identities.
    """

    exact_block_dim: Dim3 | int | tuple[int, ...] | list[int] | None = None
    max_block_dim: Dim3 | int | tuple[int, ...] | list[int] | None = None
    exact_grid_dim: Dim3 | int | tuple[int, ...] | list[int] | None = None
    exact_cluster_dim: Dim3 | int | tuple[int, ...] | list[int] | None = None
    cooperative_launch: bool | None = None
    cluster_launch: bool | None = None
    provenance: tuple[LaunchFactOrigin, ...] | LaunchFactOrigin = field(
        default=(),
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        for field_name, label in (
            ("exact_block_dim", "exact block"),
            ("max_block_dim", "maximum block"),
            ("exact_grid_dim", "exact grid"),
            ("exact_cluster_dim", "exact cluster"),
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_optional_dim(getattr(self, field_name), label=label),
            )

        for field_name in ("cooperative_launch", "cluster_launch"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"LaunchFacts {field_name} must be bool or None")

        provenance = self.provenance
        if isinstance(provenance, LaunchFactOrigin):
            provenance = (provenance,)
        else:
            provenance = tuple(provenance)
        if any(not isinstance(item, LaunchFactOrigin) for item in provenance):
            raise TypeError(
                "LaunchFacts provenance entries must be LaunchFactOrigin records"
            )
        object.__setattr__(self, "provenance", provenance)

        exact = self.exact_block_dim
        maximum = self.max_block_dim
        if exact is not None and maximum is not None:
            if any(required > limit for required, limit in zip(exact, maximum)):
                raise ValueError("LaunchFacts exact_block_dim exceeds max_block_dim")

    @property
    def exact_block_threads(self) -> int | None:
        if self.exact_block_dim is None:
            return None
        x, y, z = self.exact_block_dim
        return x * y * z

    @property
    def max_block_threads(self) -> int | None:
        if self.max_block_dim is None:
            return None
        x, y, z = self.max_block_dim
        return x * y * z

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.exact_block_dim,
            self.max_block_dim,
            self.exact_grid_dim,
            self.exact_cluster_dim,
            self.cooperative_launch,
            self.cluster_launch,
        )

    def is_verified(self, fact: str) -> bool:
        """Return whether a backend-originated fact verifies ``fact``."""

        return any(
            origin.fact == fact and origin.verified for origin in self.provenance
        )


def _merge_exact_dimension(
    facts: tuple[LaunchFacts, ...],
    field_name: str,
) -> Dim3 | None:
    values = {
        value for fact in facts if (value := getattr(fact, field_name)) is not None
    }
    if len(values) > 1:
        raise LaunchFactConflict(
            f"conflicting {field_name} launch facts: {sorted(values)!r}"
        )
    return next(iter(values), None)


def _merge_max_block_dimension(facts: tuple[LaunchFacts, ...]) -> Dim3 | None:
    values = tuple(
        fact.max_block_dim for fact in facts if fact.max_block_dim is not None
    )
    if not values:
        return None
    return tuple(min(dimensions) for dimensions in zip(*values))  # type: ignore[return-value]


def _merge_capability(
    facts: tuple[LaunchFacts, ...],
    field_name: str,
) -> bool | None:
    values = {
        value for fact in facts if (value := getattr(fact, field_name)) is not None
    }
    if len(values) > 1:
        raise LaunchFactConflict(f"conflicting {field_name} launch facts")
    return next(iter(values), None)


def merge_launch_facts(*facts: LaunchFacts) -> LaunchFacts:
    """Merge compatible facts without promoting upper bounds to exact facts."""

    if any(not isinstance(fact, LaunchFacts) for fact in facts):
        raise TypeError("merge_launch_facts expects LaunchFacts records")
    facts = tuple(facts)
    provenance = tuple(
        dict.fromkeys(origin for fact in facts for origin in fact.provenance)
    )
    try:
        return LaunchFacts(
            exact_block_dim=_merge_exact_dimension(facts, "exact_block_dim"),
            max_block_dim=_merge_max_block_dimension(facts),
            exact_grid_dim=_merge_exact_dimension(facts, "exact_grid_dim"),
            exact_cluster_dim=_merge_exact_dimension(facts, "exact_cluster_dim"),
            cooperative_launch=_merge_capability(facts, "cooperative_launch"),
            cluster_launch=_merge_capability(facts, "cluster_launch"),
            provenance=provenance,
        )
    except ValueError as exc:
        if isinstance(exc, LaunchFactConflict):
            raise
        raise LaunchFactConflict(str(exc)) from exc


__all__ = [
    "Dim3",
    "LaunchFactConflict",
    "LaunchFactOrigin",
    "LaunchFacts",
    "merge_launch_facts",
]
