# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register and dispatch backend-neutral primitive families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .._types import ParameterClassification
from ..launch import LaunchFacts
from ..thread_group import THREAD_GROUP_KINDS, ThreadGroup
from ._contracts import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupOperationSemantics,
    GroupPrimitiveCall,
    UnsupportedReasonCode,
)
from ._resolution import _resolve_group


@dataclass(frozen=True)
class _GroupOperationFamily:
    classifications: Callable[
        [GroupOperationSemantics], tuple[ParameterClassification, ...]
    ]
    planner: Callable[
        [GroupPrimitiveCall, ThreadGroup, LaunchFacts, GroupOperationSemantics],
        GroupLoweringPlan,
    ]
    group_kinds: frozenset[str]
    unsupported_group_message: str

    def __post_init__(self) -> None:
        if not callable(self.classifications):
            raise TypeError("classifications must be callable")
        if not callable(self.planner):
            raise TypeError("planner must be callable")
        object.__setattr__(self, "group_kinds", frozenset(self.group_kinds))
        if not self.group_kinds:
            raise ValueError("group_kinds must not be empty")
        invalid_group_kinds = {
            kind
            for kind in self.group_kinds
            if not isinstance(kind, str) or kind not in THREAD_GROUP_KINDS
        }
        if invalid_group_kinds:
            names = ", ".join(sorted((repr(kind) for kind in invalid_group_kinds)))
            raise ValueError(f"group_kinds contains unsupported values: {names}")
        if (
            not isinstance(self.unsupported_group_message, str)
            or not self.unsupported_group_message.strip()
        ):
            raise ValueError("unsupported_group_message must be a non-empty string")


_GROUP_OPERATION_FAMILIES: dict[type, _GroupOperationFamily] = {}


def _register_group_operation_family(
    semantics_type: type,
    *,
    classifications: Callable[
        [GroupOperationSemantics], tuple[ParameterClassification, ...]
    ],
    planner: Callable[
        [GroupPrimitiveCall, ThreadGroup, LaunchFacts, GroupOperationSemantics],
        GroupLoweringPlan,
    ],
    group_kinds: frozenset[str],
    unsupported_group_message: str,
) -> None:
    """Register one semantic family without changing the neutral dispatcher."""

    if not isinstance(semantics_type, type):
        raise TypeError("semantics_type must be a type")
    registration = _GroupOperationFamily(
        classifications=classifications,
        planner=planner,
        group_kinds=frozenset(group_kinds),
        unsupported_group_message=unsupported_group_message,
    )
    existing = _GROUP_OPERATION_FAMILIES.get(semantics_type)
    if existing is not None and existing != registration:
        raise RuntimeError(
            f"group operation semantics {semantics_type!r} are already registered"
        )
    _GROUP_OPERATION_FAMILIES[semantics_type] = registration


def _group_operation_family(operation: object) -> _GroupOperationFamily | None:
    return _GROUP_OPERATION_FAMILIES.get(type(operation))


def _is_group_operation(operation: object) -> bool:
    return _group_operation_family(operation) is not None


def _call_classifications(
    operation: GroupOperationSemantics,
) -> tuple[ParameterClassification, ...]:
    family = _group_operation_family(operation)
    if family is None:
        raise TypeError("unsupported GroupPrimitiveCall operation")
    return family.classifications(operation)


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupOperationSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
    del source
    return GroupPrimitiveCall(group=group, operation=operation)


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve a compile-time group call to an official CUDAX/CUB target."""

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    family = _group_operation_family(call.operation)
    if family is None:
        raise TypeError("unsupported GroupPrimitiveCall operation")
    if call.group.kind not in family.group_kinds:
        return _unsupported(
            call,
            call.group,
            UnsupportedReasonCode.GROUP_KIND,
            family.unsupported_group_message,
        )
    cluster_dim = launch.exact_cluster_dim
    uses_multi_block_cluster = cluster_dim is not None and cluster_dim != (1, 1, 1)
    if call.group.kind in {"cluster", "grid"} and uses_multi_block_cluster:
        if launch.cluster_launch is not True or not launch.is_verified(
            "cluster_launch"
        ):
            return _unsupported(
                call,
                call.group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "multi-block cluster lowering requires verified cluster launch "
                f"capability; observed {launch.cluster_launch!r} with verified="
                f"{launch.is_verified('cluster_launch')!r}",
            )
    if call.group.kind == "grid":
        if launch.cooperative_launch is not True or not launch.is_verified(
            "cooperative_launch"
        ):
            return _unsupported(
                call,
                call.group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "grid group lowering requires verified cooperative launch "
                f"capability; observed {launch.cooperative_launch!r} with "
                f"verified={launch.is_verified('cooperative_launch')!r}",
            )
    resolved, failure = _resolve_group(call, launch)
    if failure is not None:
        return failure
    return family.planner(call, resolved, launch, call.operation)


__all__ = [
    "GroupOperationSemantics",
    "make_group_primitive_call",
    "plan_group_primitive",
]
