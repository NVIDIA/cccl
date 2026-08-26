# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Classify and dispatch portable group calls to family planners.

This is the single cross-family routing table after group resolution. It keeps
runtime ABI classification and planner selection explicit while leaving each
primitive's semantic and lowering rules in its adjacent family module.
"""

from __future__ import annotations

from .._bindings import BindingKind
from .._types import (
    ArgumentKind,
    ParameterClassification,
    ParameterRole,
    classify_parameter,
)
from ..block.adjacent_difference import BlockAdjacentDifferenceBoundary
from ..launch import LaunchFacts
from ..thread_group import ThreadGroup
from ._contracts import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupPrimitiveCall,
    UnsupportedReasonCode,
)
from ._resolution import _resolve_group
from .adjacent_difference import (
    GroupAdjacentDifferenceSemantics,
    _plan_adjacent_difference,
)
from .discontinuity import GroupDiscontinuitySemantics, _plan_discontinuity
from .exchange import GroupExchangeSemantics, _plan_exchange
from .histogram import GroupHistogramSemantics, _plan_histogram
from .load_store import (
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    _plan_load_store,
)
from .reduce import GroupReduceSemantics, _plan_reduce
from .run_length_decode import (
    GroupRunLengthDecodeSemantics,
    _plan_run_length_decode,
)
from .scan import GroupScanSemantics, _plan_scan
from .shuffle import GroupShuffleSemantics, _plan_shuffle

GroupOperationSemantics = (
    GroupReduceSemantics
    | GroupScanSemantics
    | GroupAdjacentDifferenceSemantics
    | GroupDiscontinuitySemantics
    | GroupShuffleSemantics
    | GroupHistogramSemantics
    | GroupRunLengthDecodeSemantics
    | GroupExchangeSemantics
    | GroupLoadStoreSemantics
)
_GROUP_OPERATION_TYPES = (
    GroupReduceSemantics,
    GroupScanSemantics,
    GroupAdjacentDifferenceSemantics,
    GroupDiscontinuitySemantics,
    GroupShuffleSemantics,
    GroupHistogramSemantics,
    GroupRunLengthDecodeSemantics,
    GroupExchangeSemantics,
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

    if isinstance(operation, GroupRunLengthDecodeSemantics):
        primitive = operation.primitive
        classifications = [
            ParameterClassification(
                "run_values", ArgumentKind.RUNTIME, ParameterRole.INPUT
            ),
            ParameterClassification(
                "run_lengths", ArgumentKind.RUNTIME, ParameterRole.INPUT
            ),
            ParameterClassification(
                "decoded_items_per_thread",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
            ParameterClassification(
                "decoded_window_offset",
                ArgumentKind.RUNTIME,
                ParameterRole.INPUT,
            ),
        ]
        if primitive.has_relative_offsets:
            classifications.append(
                ParameterClassification(
                    "relative_offsets",
                    ArgumentKind.RUNTIME,
                    ParameterRole.OUTPUT,
                )
            )
        classifications.append(
            ParameterClassification(
                "total_decoded_size",
                ArgumentKind.RUNTIME,
                ParameterRole.OUTPUT,
            )
        )
        return tuple(classifications)

    primary_argument = (
        "samples" if isinstance(operation, GroupHistogramSemantics) else "value"
    )
    classifications = [
        ParameterClassification(
            primary_argument,
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT,
        )
    ]
    if isinstance(operation, GroupHistogramSemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "bins", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "bins_per_thread", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "algorithm", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
            )
        )
    elif isinstance(operation, GroupAdjacentDifferenceSemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "direction", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "operation", ArgumentKind.STATIC, ParameterRole.OPERATOR
                ),
            )
        )
        if operation.primitive.has_partial_tile:
            classifications.append(
                ParameterClassification(
                    "valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
        if operation.primitive.boundary is not BlockAdjacentDifferenceBoundary.NONE:
            boundary_name = (
                "tile_predecessor_item"
                if operation.primitive.boundary
                is BlockAdjacentDifferenceBoundary.PREDECESSOR
                else "tile_successor_item"
            )
            classifications.append(
                ParameterClassification(
                    boundary_name, ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
    elif isinstance(operation, GroupDiscontinuitySemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "mode", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "operation", ArgumentKind.STATIC, ParameterRole.OPERATOR
                ),
            )
        )
        if operation.primitive.has_tile_predecessor:
            classifications.append(
                ParameterClassification(
                    "tile_predecessor_item",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        if operation.primitive.has_tile_successor:
            classifications.append(
                ParameterClassification(
                    "tile_successor_item",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
    elif isinstance(operation, GroupShuffleSemantics):
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
        if operation.primitive.distance.argument_kind is not None:
            classifications.append(
                ParameterClassification(
                    "distance",
                    operation.primitive.distance.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if operation.primitive.distance.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        for boundary_name, enabled in (
            ("block_prefix", operation.primitive.block_prefix),
            ("block_suffix", operation.primitive.block_suffix),
        ):
            if enabled:
                classifications.append(
                    ParameterClassification(
                        boundary_name,
                        ArgumentKind.RUNTIME,
                        ParameterRole.OUTPUT,
                    )
                )
    elif isinstance(operation, GroupExchangeSemantics):
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
        if operation.primitive.uses_ranks:
            classifications.append(
                ParameterClassification(
                    "ranks",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        if operation.primitive.uses_valid_flags:
            classifications.append(
                ParameterClassification(
                    "valid_flags",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
    elif isinstance(operation, GroupReduceSemantics):
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
                    "operation", ArgumentKind.STATIC, ParameterRole.CONSTANT
                )
            )
    elif isinstance(operation, GroupScanSemantics):
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
    else:
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
    return tuple(classifications)


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupOperationSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
    del source
    return GroupPrimitiveCall(
        group=group,
        operation=operation,
    )


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
    if isinstance(operation, GroupAdjacentDifferenceSemantics):
        return _plan_adjacent_difference(call, resolved, launch, operation)
    if isinstance(operation, GroupDiscontinuitySemantics):
        return _plan_discontinuity(call, resolved, launch, operation)
    if isinstance(operation, GroupShuffleSemantics):
        return _plan_shuffle(call, resolved, launch, operation)
    if isinstance(operation, GroupHistogramSemantics):
        return _plan_histogram(call, resolved, launch, operation)
    if isinstance(operation, GroupRunLengthDecodeSemantics):
        return _plan_run_length_decode(call, resolved, launch, operation)
    if isinstance(operation, GroupExchangeSemantics):
        return _plan_exchange(call, resolved, launch, operation)
    return _plan_load_store(call, resolved, launch, operation)


__all__ = [
    "GroupOperationSemantics",
    "make_group_primitive_call",
    "plan_group_primitive",
]
