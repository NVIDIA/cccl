# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dispatch portable group operations to their family planners."""

from __future__ import annotations

from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _unsupported_plan, _validate_static_valid_items
from ._model import GroupLoweringPlan, GroupPrimitiveCall
from ._resolution import resolve_thread_group
from .reduce import GroupReduceSemantics, _plan_reduce

GroupOperationSemantics = GroupReduceSemantics


def _is_group_operation(operation: object) -> bool:
    return isinstance(operation, GroupReduceSemantics)


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupOperationSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
    """Build one canonical block or warp reduction call."""

    return GroupPrimitiveCall(group=group, operation=operation, source=source)


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve one scalar block or warp reduction to a typed CUB plan."""

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    operation = call.operation
    _validate_static_valid_items(operation.valid_items)
    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is not None:
        return _unsupported_plan(call, resolution)
    return _plan_reduce(call, resolution.group, launch, operation)


__all__ = [
    "GroupOperationSemantics",
    "make_group_primitive_call",
    "plan_group_primitive",
]
