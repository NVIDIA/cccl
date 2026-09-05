# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Classify and dispatch the initial portable group operations."""

from __future__ import annotations

from .._bindings import BindingKind
from .._types import (
    ArgumentKind,
    ParameterClassification,
    ParameterRole,
    classify_parameter,
)
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupPrimitiveCall,
    UnsupportedReasonCode,
)
from ._resolution import _resolve_group
from .load_store import (
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    _plan_load_store,
)
from .reduce import GroupReduceSemantics, _plan_reduce
from .scan import GroupScanSemantics, _plan_scan

GroupOperationSemantics = (
    GroupReduceSemantics | GroupScanSemantics | GroupLoadStoreSemantics
)
_GROUP_OPERATION_TYPES = (
    GroupReduceSemantics,
    GroupScanSemantics,
    GroupLoadStoreSemantics,
)


def _is_group_operation(operation: object) -> bool:
    return isinstance(operation, _GROUP_OPERATION_TYPES)


def _call_classifications(
    operation: GroupOperationSemantics,
) -> tuple[ParameterClassification, ...]:
    if isinstance(operation, GroupLoadStoreSemantics):
        classifications = [
            ParameterClassification(
                "source"
                if operation.kind is GroupLoadStoreKind.LOAD
                else "destination",
                ArgumentKind.RUNTIME,
                (
                    ParameterRole.INPUT
                    if operation.kind is GroupLoadStoreKind.LOAD
                    else ParameterRole.OUTPUT
                ),
            )
        ]
        if operation.kind is GroupLoadStoreKind.LOAD:
            classifications.append(
                ParameterClassification(
                    "output",
                    ArgumentKind.RUNTIME,
                    ParameterRole.OUTPUT,
                )
            )
        else:
            classifications.append(
                ParameterClassification(
                    "value",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        for name, binding in (
            ("valid_items", operation.valid_items),
            ("oob_default", operation.oob_default),
            ("offset", operation.offset),
        ):
            if binding.argument_kind is None:
                continue
            classifications.append(
                ParameterClassification(
                    name,
                    binding.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if binding.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        classifications.append(
            ParameterClassification(
                "algorithm",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            )
        )
        return tuple(classifications)

    classifications = [
        ParameterClassification(
            "value",
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT,
        )
    ]
    if isinstance(operation, GroupReduceSemantics):
        if operation.valid_items.argument_kind is not None:
            classifications.append(
                ParameterClassification(
                    "valid_items",
                    operation.valid_items.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if operation.valid_items.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        if operation.reduce_operator is not None:
            classification = classify_parameter(operation.reduce_operator)
            classifications.append(
                ParameterClassification(
                    "operation",
                    classification.kind,
                    classification.role,
                )
            )
        else:
            classifications.append(
                ParameterClassification(
                    "operation",
                    ArgumentKind.STATIC,
                    ParameterRole.CONSTANT,
                )
            )
        return tuple(classifications)

    classifications.append(
        ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
    )
    for name, parameter in (
        ("initial_value", operation.initial_value),
        ("operation", operation.scan_operator),
        ("prefix_callback", operation.prefix_callback),
    ):
        if parameter is None:
            continue
        classification = classify_parameter(parameter)
        classifications.append(
            ParameterClassification(
                name,
                classification.kind,
                classification.role,
            )
        )
    if operation.valid_items.argument_kind is not None:
        classifications.append(
            ParameterClassification(
                "valid_items",
                operation.valid_items.argument_kind,
                (
                    ParameterRole.CONSTANT
                    if operation.valid_items.kind is BindingKind.STATIC
                    else ParameterRole.INPUT
                ),
            )
        )
    return tuple(classifications)


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
    operation = call.operation
    if isinstance(operation, GroupReduceSemantics):
        return _plan_reduce(call, resolved, launch, operation)
    if isinstance(operation, GroupScanSemantics):
        return _plan_scan(call, resolved, launch, operation)
    return _plan_load_store(call, resolved, launch, operation)


__all__ = [
    "GroupOperationSemantics",
    "make_group_primitive_call",
    "plan_group_primitive",
]
