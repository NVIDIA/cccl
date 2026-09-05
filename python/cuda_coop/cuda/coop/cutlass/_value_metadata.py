# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time value-availability metadata for cooperative operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cuda.coop._core import ResultVisibility, ThreadGroup

_SCALAR_METADATA_LOOKUP: Callable[[Any], ValueGroupMetadata | None] | None = None


class DefinedThreadDomainKind(str, Enum):
    MEMBERS = "members"
    ROOTS = "roots"


@dataclass(frozen=True, eq=False)
class ResolvedGroupIdentity:
    """The semantic identity and lineage of a resolved static group."""

    group: ThreadGroup

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("ResolvedGroupIdentity group must be a ThreadGroup")
        if not self.group.is_static:
            raise ValueError("ResolvedGroupIdentity requires a static resolved group")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.group.semantic_key

    @property
    def lineage_keys(self) -> tuple[tuple[Any, ...], ...]:
        keys = []
        group: ThreadGroup | None = self.group
        while group is not None:
            keys.append(group.semantic_key)
            group = group.parent
        return tuple(keys)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResolvedGroupIdentity):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class DefinedThreadConstraint:
    kind: DefinedThreadDomainKind
    group_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DefinedThreadDomainKind(self.kind))
        if not self.group_key:
            raise ValueError("defined-thread constraint requires a group key")


@dataclass(frozen=True)
class DefinedThreadDomain:
    """Intersection of constraints describing where a value is defined."""

    constraints: frozenset[DefinedThreadConstraint] = frozenset()

    def __post_init__(self) -> None:
        constraints = frozenset(self.constraints)
        if any(
            not isinstance(constraint, DefinedThreadConstraint)
            for constraint in constraints
        ):
            raise TypeError(
                "DefinedThreadDomain constraints must be DefinedThreadConstraint"
            )
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def all_callers(cls) -> DefinedThreadDomain:
        return cls()

    @classmethod
    def members(cls, group: ResolvedGroupIdentity) -> DefinedThreadDomain:
        return cls(
            frozenset(
                {
                    DefinedThreadConstraint(
                        DefinedThreadDomainKind.MEMBERS,
                        group.semantic_key,
                    )
                }
            )
        )

    @classmethod
    def roots(cls, group: ResolvedGroupIdentity) -> DefinedThreadDomain:
        return cls(
            frozenset(
                {
                    DefinedThreadConstraint(
                        DefinedThreadDomainKind.ROOTS,
                        group.semantic_key,
                    )
                }
            )
        )

    def intersect(self, other: DefinedThreadDomain) -> DefinedThreadDomain:
        if not isinstance(other, DefinedThreadDomain):
            raise TypeError("defined-thread domains can intersect only each other")
        return DefinedThreadDomain(self.constraints | other.constraints)

    @property
    def contains_roots_only(self) -> bool:
        return any(
            constraint.kind is DefinedThreadDomainKind.ROOTS
            for constraint in self.constraints
        )

    def covers(self, target: ResolvedGroupIdentity) -> bool:
        lineage = frozenset(target.lineage_keys)
        for constraint in self.constraints:
            if constraint.kind is DefinedThreadDomainKind.ROOTS:
                return False
            if constraint.group_key not in lineage:
                return False
        return True


@dataclass(frozen=True)
class ValueGroupMetadata:
    """Compile-time facts describing where a value is available."""

    defined_domain: DefinedThreadDomain
    visibility: ResultVisibility

    def __post_init__(self) -> None:
        if not isinstance(self.defined_domain, DefinedThreadDomain):
            raise TypeError("defined_domain must be a DefinedThreadDomain")
        object.__setattr__(self, "visibility", ResultVisibility(self.visibility))


def metadata_for_group(
    group: ThreadGroup,
    *,
    visibility: ResultVisibility,
) -> ValueGroupMetadata:
    visibility = ResultVisibility(visibility)
    if visibility is ResultVisibility.GROUP_ROOT:
        domain = DefinedThreadDomain.roots(ResolvedGroupIdentity(group))
    elif group.complete_membership is False:
        domain = DefinedThreadDomain.members(ResolvedGroupIdentity(group))
    else:
        domain = DefinedThreadDomain.all_callers()
    return ValueGroupMetadata(domain, visibility)


def merge_value_metadata(
    metadata: Iterable[ValueGroupMetadata | None],
) -> ValueGroupMetadata | None:
    values = tuple(value for value in metadata if value is not None)
    if not values:
        return None
    domain = DefinedThreadDomain.all_callers()
    for value in values:
        domain = domain.intersect(value.defined_domain)
    if domain.contains_roots_only:
        visibility = ResultVisibility.GROUP_ROOT
    elif any(value.visibility is ResultVisibility.PER_MEMBER for value in values):
        visibility = ResultVisibility.PER_MEMBER
    else:
        visibility = ResultVisibility.ALL_MEMBERS
    return ValueGroupMetadata(domain, visibility)


def attach_thread_data_metadata(value: Any, metadata: ValueGroupMetadata | None) -> Any:
    if metadata is not None and not isinstance(metadata, ValueGroupMetadata):
        raise TypeError("thread-data metadata must be ValueGroupMetadata or None")
    setter = getattr(value, "_set_group_metadata", None)
    if callable(setter):
        setter(metadata)
    else:
        value._group_metadata = metadata
    return value


def thread_data_metadata(value: Any) -> ValueGroupMetadata | None:
    metadata = getattr(value, "_group_metadata", None)
    return metadata if isinstance(metadata, ValueGroupMetadata) else None


def register_scalar_metadata_lookup(
    lookup: Callable[[Any], ValueGroupMetadata | None],
) -> None:
    if not callable(lookup):
        raise TypeError("scalar metadata lookup must be callable")
    global _SCALAR_METADATA_LOOKUP
    _SCALAR_METADATA_LOOKUP = lookup


def value_group_metadata(value: Any) -> ValueGroupMetadata | None:
    metadata = thread_data_metadata(value)
    if metadata is not None or _SCALAR_METADATA_LOOKUP is None:
        return metadata
    scalar_metadata = _SCALAR_METADATA_LOOKUP(value)
    return scalar_metadata if isinstance(scalar_metadata, ValueGroupMetadata) else None


def validate_operand_domains(
    group: ThreadGroup,
    operands: dict[str, Any],
    *,
    scope: str,
    primitive_name: str,
) -> None:
    """Prove every tagged operand is defined for all target participants."""

    target = ResolvedGroupIdentity(group)
    for name, value in operands.items():
        metadata = value_group_metadata(value)
        if metadata is None:
            continue
        if metadata.defined_domain.contains_roots_only:
            raise ValueError(
                f"{scope}.{primitive_name} operand {name!r} is defined only at "
                "group roots; use broadcast=True or consume it under rank-0 "
                "control flow with non-cooperative scalar operations"
            )
        if not metadata.defined_domain.covers(target):
            raise ValueError(
                f"{scope}.{primitive_name} operand {name!r} is not defined for "
                f"every member of target group {group.symbol_suffix}; pass a "
                "compatible group or rebuild the value for the wider domain"
            )


__all__ = [
    "DefinedThreadConstraint",
    "DefinedThreadDomain",
    "DefinedThreadDomainKind",
    "ResolvedGroupIdentity",
    "ValueGroupMetadata",
    "attach_thread_data_metadata",
    "merge_value_metadata",
    "metadata_for_group",
    "register_scalar_metadata_lookup",
    "thread_data_metadata",
    "validate_operand_domains",
    "value_group_metadata",
]
