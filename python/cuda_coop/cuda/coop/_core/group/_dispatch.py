# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register and dispatch backend-neutral primitive families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .._types import ParameterClassification
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
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
        classifications,
        planner,
        frozenset(group_kinds),
        unsupported_group_message,
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
    resolved, failure = _resolve_group(call, launch)
    if failure is not None:
        return failure
    return family.planner(call, resolved, launch, call.operation)


__all__ = [
    "GroupOperationSemantics",
    "make_group_primitive_call",
    "plan_group_primitive",
]
