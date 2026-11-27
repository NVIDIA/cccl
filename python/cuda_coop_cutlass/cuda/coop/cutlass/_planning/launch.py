# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Record the launch information that group planning can use.

A compiler can know an exact block shape, an upper bound on that shape,
or only part of a launch configuration. ``LaunchFacts`` keeps these
cases separate. A bound can reject an impossible launch, but it cannot
supply the exact thread count needed to choose a group implementation.

Origins record where each fact came from and whether its producer
verified it. Group planning uses this evidence for capabilities such as
cluster and cooperative launches. Merging facts keeps compatible values
and reports contradictions before an operation is compiled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .thread_group import normalize_thread_dim

Dim3 = tuple[int, int, int]

_LAUNCH_FACT_VALUE_FIELDS = frozenset(
    {
        "exact_block_dim",
        "max_block_dim",
        "exact_grid_dim",
        "exact_cluster_dim",
        "cooperative_launch",
        "cluster_launch",
    }
)


class LaunchFactConflict(ValueError):
    """Two individually valid launch facts contradict one another."""


def _normalize_optional_dim(value: Any, *, label: str) -> Dim3 | None:
    if value is None:
        return None
    return normalize_thread_dim(value, scope="LaunchFacts", label=label)


@dataclass(frozen=True)
class LaunchFactOrigin:
    """Record the source of a launch fact and its verification status.

    A backend uses this record to explain how it obtained a value, such as
    an exact block shape from the configured kernel launch. ``LaunchFacts``
    retains the record for diagnostics and tracks verified fact names for
    planning. The record itself does not inspect or verify a launch.

    Parameters
    ----------
    fact : str
        Name of the fact being described. A verified origin must name a
        value field in the containing ``LaunchFacts`` record.
    source : str
        Nonempty name identifying the producer of the fact.
    detail : str, optional
        Extra diagnostic text about how the producer obtained the value.
    verified : bool, optional
        Whether the producer has verified this fact. The default is false.
        A verified origin requires a value in the same ``LaunchFacts``.
    """

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


@dataclass(frozen=True, eq=False)
class LaunchFacts:
    """Keep exact launch dimensions, bounds, and capability evidence.

    Group planners need exact dimensions to count participants and allocate
    storage for each group. ``max_block_dim`` only limits a possible block;
    it does not identify the block that will run. Capability values use
    ``None`` for unknown, which differs from a known false value.

    Construction normalizes dimensions to three positive integers. It checks
    that an exact block fits its bound and that each verified origin names a
    fact with a value. It records the producer's verification claim; it does
    not query the driver or inspect a kernel launch.

    Parameters
    ----------
    exact_block_dim : int or sequence of int, optional
        Exact thread dimensions of one block.
    max_block_dim : int or sequence of int, optional
        Upper bound for each block dimension.
    exact_grid_dim : int or sequence of int, optional
        Exact number of blocks along each grid axis. The group
        resolver converts these to cluster counts when needed.
    exact_cluster_dim : int or sequence of int, optional
        Exact number of blocks along each axis of one cluster.
    cooperative_launch : bool, optional
        Whether the launch supports cooperative grid participation.
    cluster_launch : bool, optional
        Whether the launch uses clusters.
    provenance : LaunchFactOrigin or tuple of LaunchFactOrigin, optional
        Sources for these facts. Sources and diagnostic details do not
        affect equality or hashing. Verified fact names do, because a
        planner can require verified capability evidence.

    Notes
    -----
    Two frontends that supply the same values and verify the same fact names
    share an identity, even if they describe different ways of obtaining
    those facts.
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
    _verified_facts: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize dimensions and check the supplied evidence."""

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
                raise TypeError(
                    f"LaunchFacts {field_name} must be bool or None"
                )

        provenance = self.provenance
        if isinstance(provenance, LaunchFactOrigin):
            provenance = (provenance,)
        else:
            provenance = tuple(provenance)
        if any(not isinstance(item, LaunchFactOrigin) for item in provenance):
            raise TypeError(
                "LaunchFacts provenance entries must be "
                "LaunchFactOrigin records"
            )
        object.__setattr__(self, "provenance", provenance)
        verified_facts = set()
        for origin in provenance:
            if not origin.verified:
                continue
            if origin.fact not in _LAUNCH_FACT_VALUE_FIELDS:
                raise ValueError(
                    "verified LaunchFactOrigin fact must name a LaunchFacts "
                    f"value field; got {origin.fact!r}"
                )
            if getattr(self, origin.fact) is None:
                raise ValueError(
                    f"verified LaunchFactOrigin for {origin.fact!r} requires "
                    "the same LaunchFacts record to carry its value"
                )
            verified_facts.add(origin.fact)
        object.__setattr__(
            self,
            "_verified_facts",
            tuple(sorted(verified_facts)),
        )

        exact = self.exact_block_dim
        maximum = self.max_block_dim
        if (
            exact is not None
            and maximum is not None
            and any(required > limit for required, limit in zip(exact, maximum))
        ):
            raise ValueError(
                "LaunchFacts exact_block_dim exceeds max_block_dim"
            )

    @property
    def exact_block_threads(self) -> int | None:
        """Return the exact block thread count, or ``None`` if it is unknown.

        An upper bound does not contribute to this count.
        """

        if self.exact_block_dim is None:
            return None
        x, y, z = self.exact_block_dim
        return x * y * z

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the values and verified fact names that affect planning.

        Exclude source labels and diagnostic details so equivalent backend
        facts can share a cache entry.
        """

        return (
            self.exact_block_dim,
            self.max_block_dim,
            self.exact_grid_dim,
            self.exact_cluster_dim,
            self.cooperative_launch,
            self.cluster_launch,
            self._verified_facts,
        )

    def is_verified(self, fact: str) -> bool:
        """Return whether a producer marked this fact's value as verified.

        Construction checks that each verified name has a value in the record.
        This query does not perform another verification.
        """

        return fact in self._verified_facts

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaunchFacts):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def _merge_exact_dimension(
    facts: tuple[LaunchFacts, ...],
    field_name: str,
) -> Dim3 | None:
    """Keep one known exact shape; reject different known shapes."""

    values = {
        value
        for fact in facts
        if (value := getattr(fact, field_name)) is not None
    }
    if len(values) > 1:
        raise LaunchFactConflict(
            f"conflicting {field_name} launch facts: {sorted(values)!r}"
        )
    return next(iter(values), None)


def _merge_max_block_dimension(facts: tuple[LaunchFacts, ...]) -> Dim3 | None:
    """Intersect known bounds by taking the minimum along each axis."""

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
    """Combine known capability values and reject a true/false conflict."""

    values = {
        value
        for fact in facts
        if (value := getattr(fact, field_name)) is not None
    }
    if len(values) > 1:
        raise LaunchFactConflict(f"conflicting {field_name} launch facts")
    return next(iter(values), None)


def merge_launch_facts(*facts: LaunchFacts) -> LaunchFacts:
    """Combine compatible launch knowledge from several producers.

    Exact dimensions and known capability flags must agree. Block bounds
    combine by taking the smallest limit on each axis. An empty input gives
    an empty record. Bounds remain bounds; merging never uses one as an
    exact launch shape.

    Keep distinct origin records for diagnostics. If a merged value differs
    from the value an origin verified, clear that origin's verification
    flag. This matters when the intersection of several block bounds
    produces a new bound that no one producer supplied.

    Parameters
    ----------
    *facts : LaunchFacts
        Records to combine. Unknown values leave known values unchanged.

    Returns
    -------
    LaunchFacts
        Combined values, bounds, and origin records.

    Raises
    ------
    TypeError
        An input is not a ``LaunchFacts`` record.
    LaunchFactConflict
        Exact dimensions or capability flags disagree, or the merged exact
        block exceeds the merged bound.
    """

    if any(not isinstance(fact, LaunchFacts) for fact in facts):
        raise TypeError("merge_launch_facts expects LaunchFacts records")
    facts = tuple(facts)
    try:
        merged_values = {
            "exact_block_dim": _merge_exact_dimension(facts, "exact_block_dim"),
            "max_block_dim": _merge_max_block_dimension(facts),
            "exact_grid_dim": _merge_exact_dimension(facts, "exact_grid_dim"),
            "exact_cluster_dim": _merge_exact_dimension(
                facts, "exact_cluster_dim"
            ),
            "cooperative_launch": _merge_capability(
                facts, "cooperative_launch"
            ),
            "cluster_launch": _merge_capability(facts, "cluster_launch"),
        }
        provenance = tuple(
            dict.fromkeys(
                replace(origin, verified=False)
                if origin.verified
                and getattr(fact, origin.fact) != merged_values[origin.fact]
                else origin
                for fact in facts
                for origin in fact.provenance
            )
        )
        return LaunchFacts(**merged_values, provenance=provenance)
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
